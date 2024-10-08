# CUDA_VISIBLE_DEVICES=0 python main.py --dataset=library --model=LightGCN --experiment=add_noise  --noise_pct=10 --lrate=0.001 --embedding_size=64 --weight_decay=0
CUDA_VISIBLE_DEVICES=0 python main.py --dataset=library --model=LightGCN --experiment=full  --noise_pct=10 --lrate=0.001 --embedding_size=64 --weight_decay=0
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset=library --model=LightGCN --experiment=cold_start  --noise_pct=10 --lrate=0.001 --embedding_size=64 --weight_decay=0
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset=library --model=LightGCN --experiment=missing  --noise_pct=10 --lrate=0.001 --embedding_size=64 --weight_decay=0
