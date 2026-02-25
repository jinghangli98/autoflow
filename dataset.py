import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
import os
from PIL import Image
import glob
import numpy as np
import torchio as tio
import matplotlib.pyplot as plt
import random
from pathlib import Path
from natsort import natsorted
from torch.utils.data.distributed import DistributedSampler
import re
import torch.nn.functional as F


class CenterCrop3D:
    """Center crop for 3D volumes (crops H and W only, keeps all slices)"""
    def __init__(self, crop_h, crop_w):
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __call__(self, img):
        """
        Args:
            img: tensor of shape (D, H, W) or (C, D, H, W)
        """
        h, w = img.shape[-2], img.shape[-1]
        new_h, new_w = self.crop_h, self.crop_w

        # Center crop offsets (0 if image smaller than crop)
        top = max(0, (h - new_h) // 2)
        left = max(0, (w - new_w) // 2)

        if img.ndim == 3:
            cropped = img[:, top:top+new_h, left:left+new_w]
        else:
            cropped = img[:, :, top:top+new_h, left:left+new_w]

        # Center pad if image was smaller than crop size
        pad_h = new_h - cropped.shape[-2]
        pad_w = new_w - cropped.shape[-1]
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            # F.pad order: (left, right, top, bottom)
            cropped = F.pad(cropped, (pad_left, pad_right, pad_top, pad_bottom))

        return cropped


class tseDataset(Dataset):
    def __init__(self, image_groups, num_slices=5, crop_size=448, artifact_transform=None):
        self.image_groups = image_groups
        self.num_slices = num_slices
        self.crop = CenterCrop3D(crop_size, crop_size)
        self.artifact_transform = artifact_transform
        self.context_radius = num_slices // 2

    def __len__(self):
        return len(self.image_groups)

    def __getitem__(self, index):
        group_info = self.image_groups[index]
        center_slice_idx = group_info['center_idx']
        available_slices = group_info['available_slices']

        # Determine which slices to load around the center slice
        slice_indices = []
        for offset in range(-self.context_radius, self.context_radius + 1):
            target_idx = center_slice_idx + offset
            if target_idx < 0:
                target_idx = abs(target_idx)
            elif target_idx >= len(available_slices):
                target_idx = 2 * len(available_slices) - target_idx - 2
                target_idx = max(0, target_idx)
            target_idx = min(target_idx, len(available_slices) - 1)
            slice_indices.append(target_idx)

        # Load mprage slices
        mprage_slices = []
        for slice_idx in slice_indices:
            slice_path = available_slices[slice_idx].replace('PLACEHOLDER', 'mprage')
            try:
                img = Image.open(slice_path)
                mprage_slices.append(torch.tensor(np.array(img), dtype=torch.float32))
            except:
                center_path = available_slices[center_slice_idx].replace('PLACEHOLDER', 'mprage')
                img = Image.open(center_path)
                mprage_slices.append(torch.tensor(np.array(img), dtype=torch.float32))

        # Load tse slices
        tse_slices = []
        for slice_idx in slice_indices:
            slice_path = available_slices[slice_idx].replace('PLACEHOLDER', 'tse')
            try:
                img = Image.open(slice_path)
                tse_slices.append(torch.tensor(np.array(img), dtype=torch.float32))
            except:
                center_path = available_slices[center_slice_idx].replace('PLACEHOLDER', 'tse')
                img = Image.open(center_path)
                tse_slices.append(torch.tensor(np.array(img), dtype=torch.float32))

        # Stack slices: (num_slices, H, W)
        mprage_volume = torch.stack(mprage_slices)
        tse_volume = torch.stack(tse_slices)

        # Center crop both to 448x448
        mprage_volume = self.crop(mprage_volume)
        tse_volume = self.crop(tse_volume)

        # Normalize to [0, 1] BEFORE artifact transforms so that
        # torchio's multiplicative augmentations (RandomBiasField etc.)
        # operate in [0, 1] space and don't push values above 1.
        mprage_volume = mprage_volume / 255.0
        tse_volume = tse_volume / 255.0

        # Apply artifact transforms only to mprage (in [0,1] space)
        if self.artifact_transform:
            mprage_volume = self.artifact_transform(mprage_volume.unsqueeze(0)).squeeze()
            mprage_volume = mprage_volume.clamp(0.0, 1.0)  # guarantee [0, 1] after augmentation

        return mprage_volume, tse_volume


def group_slices_by_volume(all_files):
    volume_groups = {}

    for file_path in all_files:
        filename = Path(file_path).stem
        match = re.search(r'slice(\d+)$', filename)
        if match:
            slice_num = int(match.group(1))
            volume_name = filename[:match.start()]
        else:
            continue

        volume_key = file_path.replace('/mprage/', '/PLACEHOLDER/').replace('/tse/', '/PLACEHOLDER/')
        volume_key = str(Path(volume_key).parent / volume_name)

        if volume_key not in volume_groups:
            volume_groups[volume_key] = []

        placeholder_path = file_path.replace('/mprage/', '/PLACEHOLDER/').replace('/tse/', '/PLACEHOLDER/')
        volume_groups[volume_key].append((slice_num, placeholder_path))

    image_groups = []
    for volume_key, slices in volume_groups.items():
        slices.sort(key=lambda x: x[0])
        slice_paths = [x[1] for x in slices]

        for i in range(len(slice_paths)):
            image_groups.append({
                'center_idx': i,
                'available_slices': slice_paths,
                'base_path': volume_key,
                'volume_name': Path(volume_key).name
            })

    return image_groups


def get_dataset_3d(data_root, crop_size=448, num_slices=5, sample=1, split_ratio=0.95):
    random_effects = {
        tio.transforms.RandomBiasField(0.3, 3): 0.1,
        tio.transforms.RandomGhosting(intensity=(0.1, 0.5)): 0.1,
    }

    all_files = natsorted(glob.glob(f'{data_root}/*.tif'))
    image_groups = group_slices_by_volume(all_files)

    sample_size = max(1, int(len(image_groups) * sample / 100))
    random.seed(42)
    sampled_groups = random.sample(image_groups, sample_size)
    random.seed(None)

    dataset = tseDataset(
        image_groups=sampled_groups,
        num_slices=num_slices,
        crop_size=crop_size,
        artifact_transform=tio.OneOf(random_effects)
    )

    generator = torch.Generator().manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [int(len(dataset) * split_ratio), len(dataset) - int(len(dataset) * split_ratio)],
        generator=generator
    )

    return train_set, val_set


def getloader_3d(batch_size, data_root, crop_size=448, num_slices=5, sample=1, num_workers=4,
                 distributed=False, rank=0, world_size=1, split_ratio=0.99, train_shuffle=True):
    train_set, val_set = get_dataset_3d(data_root, crop_size, num_slices, sample, split_ratio)

    if distributed:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=train_shuffle, seed=42)
        val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False, seed=42)

        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler,
                                num_workers=num_workers, pin_memory=True, drop_last=False)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=train_shuffle,
                                  num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


if __name__ == "__main__":
    data_root = '/ix3/tibrahim/jil202/cfg_gen/qc_image_tif/mprage_2_tse/tse/coronal/'

    crop_size = 512
    batch_size = 1
    sample = 100
    num_slices = 20

    print("Testing 3D context dataset:")
    train_loader, val_loader = getloader_3d(batch_size, data_root, crop_size, num_slices, sample)

    for batch in train_loader:
        mprage_volume, tse_volume = batch
        print(f"3D - mprage shape: {mprage_volume.shape}, tse shape: {tse_volume.shape}")
        for idx in range(len(mprage_volume.squeeze())):
            m_slice = mprage_volume.squeeze()[idx]
            t_slice = tse_volume.squeeze()[idx]
            print(f"Slice {idx:03d} | mprage min: {m_slice.min():.4f}, max: {m_slice.max():.4f} | "
                        f"tse min: {t_slice.min():.4f}, max: {t_slice.max():.4f}")
            vis = torch.hstack((m_slice, t_slice))
            plt.imshow(vis, cmap='gray')
            plt.savefig(f'{idx}.png')
            plt.close()
        break
