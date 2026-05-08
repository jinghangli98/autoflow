# autoflow

3D NIfTI flow matching for accelerated MRI reconstruction (mprage / tse).

## Training

`--contrast` accepts one or more contrasts (e.g. `--contrast mprage`,
`--contrast mprage tse`). Data root must contain `train/<contrast>/` and
`test/<contrast>/` for each contrast (test/ is used as the validation set).
Checkpoint filenames are tagged with all contrasts, e.g.
`flow_matching_3d_mprage_tse_best_epoch_*.pt`.

Single contrast:

```bash
python -m torch.distributed.run --nproc_per_node=3 train_flow.py \
    --contrast mprage \
    --data_root /home/rflab/jil202/grappa-recon/dataset_grappa_nii \
    --distributed --fp16 --save_model --compile \
    --batch_size 4 --max_epochs 100 --sample 3 \
    --num_sampling_steps 2
```

Joint training over multiple contrasts with balanced per-epoch sampling:

```bash
python -m torch.distributed.run --nproc_per_node=3 train_flow.py \
    --contrast mprage tse \
    --data_root /home/rflab/jil202/grappa-recon/dataset_grappa_nii \
    --distributed --fp16 --save_model --compile \
    --batch_size 4 --max_epochs 100 --sample 100 \
    --num_sampling_steps 2 \
    --samples_per_contrast 0
```

### Balancing contrasts (`--samples_per_contrast`)

When training over multiple contrasts, dataset sizes are usually unequal and
the model can get dominated by whichever contrast has more patches. Use
`--samples_per_contrast` to draw an equal number of training samples from each
contrast every epoch:

- `--samples_per_contrast 0` — auto-balance to the smallest contrast's pool.
- `--samples_per_contrast N` — each contrast contributes exactly `N` samples
  per epoch (contrasts with fewer than `N` are oversampled with replacement).
- omit the flag — naive concatenation (the larger contrast dominates).

Each epoch reseeds, so the larger contrast cycles through a fresh random
subset every epoch — no data is permanently discarded, it just appears in a
different epoch. Works in both single-GPU and DDP mode.

### Other notable flags

- `--sample P` — keep only `P%` of the (undersampled, GT) pairs per contrast
  (useful for quick sanity runs).
- `--prev_chunk_dropout`, `--prev_chunk_noise`, `--noise_prob` — robustness
  perturbations on the autoregressive `prev_chunk` input.
- `--ema_decay`, `--ema_start_epoch` — EMA weights for evaluation/checkpoint.
- `--checkpoint_path` — path to load from on startup (resumes if present).

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
