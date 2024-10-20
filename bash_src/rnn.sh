

cd /path

for dim in 8 16 32 64 128 256;
do
CUDA_VISIBLE_DEVICES=2 python rnn.py \
    --hidden_dim $dim \
    --n_layers 1 \
    --epochs 25 \
    --train_data training.json \
    --val_data validation.json
done
