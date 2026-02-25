# autoflow
```
python -m torch.distributed.run --nproc_per_node=2 train_flow.py \
  --distributed --fp16 --save_model --compile \
  --batch_size 2 --max_epochs 100 --save_model --fp16 --sample 1 --compile --checkpoint_path /ix3/tibrahim/jil202/cfg_gen/src/autoregressive_t1_2_tse/checkpoints/flow_matching_best_epoch_final.pt

```
