cd /path

CUDA_VISIBLE_DEVICES=0 python ffnn.py \
    --hidden_dim 2048 \
    --epochs 15 \
    --train_data training.json \
    --val_data validation.json



# dim: 10, 128, 512, 1024, 2048