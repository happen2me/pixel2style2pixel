# Notes

## Train

Sanity Check:
```bash
python scripts/train_pix2pix.py \
--dataset_type=ioct_overfit \
--exp_dir=/home/extra/micheal/pixel2style2pixel/experiments/ioct_overfit1 \
--stylegan_weights=/home/extra/micheal/pixel2style2pixel/pretrained_models/psp_celebs_seg_to_face.pt \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=2500 \
--save_interval=5000 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0 \
--w_norm_lambda=0.005 \
--label_nc=5 \
--input_nc=5 \
--output_nc=1 \
--load_partial_weights
```

Train on iOCT:
```bash
python scripts/train_pix2pix.py \
--dataset_type=ioct_seg_to_bscan \
--exp_dir=experiments/ioct_seg2bscan3 \
--stylegan_weights=pretrained_models/psp_celebs_seg_to_face.pt \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=2500 \
--save_interval=5000 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0 \
--w_norm_lambda=0.005 \
--label_nc=5 \
--input_nc=5 \
--output_nc=1 \
--load_partial_weights
```