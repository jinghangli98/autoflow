# autoflow

3D NIfTI flow matching for accelerated MRI reconstruction (mprage / tse).

## Training

Per-contrast model. Pick `--contrast mprage` or `--contrast tse`. Data root
must contain `train/<contrast>/` and `test/<contrast>/` (test/ is used as the
validation set).

```bash
python -m torch.distributed.run --nproc_per_node=3 train_flow.py \
    --contrast mprage \
    --data_root /home/rflab/jil202/grappa-recon/dataset_grappa_nii \
    --distributed --fp16 --save_model --compile \
    --batch_size 4 --max_epochs 100 --sample 3 \
    --num_sampling_steps 2 \
    --checkpoint_path ./checkpoints/flow_matching_3d_mprage_best_epoch_0.pt --contrast mprage
```

## Inference (un-patched whole volume)

```bash
python enhance_flow_3d.py \
    --checkpoint_path ./checkpoints/flow_matching_3d_mprage.pt \
    --input_path /home/rflab/jil202/grappa-recon/nii/mprage/532_5_R5.nii.gz \
    --output_path ./outputs/532_5_R5_recon.nii.gz \
    --num_sampling_steps 1 --euler --auto --fp16
```

Inference loads the un-patched volume, normalizes `[0, 255] → [0, 1]`, pads
each spatial dim to a multiple of 16, slides 16-thick chunks along Z with
autoregressive `prev_chunk`, stitches along Z, unpads, and saves.
