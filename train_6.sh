
CUDA_VISIBLE_DEVICES=4 python detective_active_model_3.py --dataset Pheme --output_dir result/EADA/10step2 \
--train_epochs 30 --active_type EADA  --active_round 10 12 14 16 18


CUDA_VISIBLE_DEVICES=4 python detective_active_model_3.py --dataset Weibo --output_dir result/EADA/10step2 \
--train_epochs 30 --active_type EADA  --active_round 10 12 14 16 18


CUDA_VISIBLE_DEVICES=4 python detective_active_model_3.py --dataset Pheme --output_dir result/EADA/10step4 \
--train_epochs 30 --active_type EADA  --active_round 10 14 18 22 26


CUDA_VISIBLE_DEVICES=4 python detective_active_model_3.py --dataset Weibo --output_dir result/EADA/10step4 \
--train_epochs 30 --active_type EADA  --active_round 10 14 18 22 26