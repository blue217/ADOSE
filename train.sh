

MODEL_FILE="multimodal_active_model_3.py"
GPU_ID=3
DATASET='Pheme'
DIR='model3'
ACTIVE_DIR='active_1'
MAX_ITER=20
ROUND="6 8 10 12 14"

CUDA_VISIBLE_DEVICES=${GPU_ID} python ${MODEL_FILE} --output_dir result/${DIR}/${ACTIVE_DIR}/ldms12 \
--dataset ${DATASET} --active_type ldms12 --active_round ${ROUND} --max_iter ${MAX_ITER}



## EADA ADA:
#
#CUDA_VISIBLE_DEVICES=3 python detective_active_model_3.py --dataset Pheme --output_dir result/EADA/6step2 \
#--train_epochs 30 --active_type EADA  --active_round 6 8 10 12 14
#
#
#CUDA_VISIBLE_DEVICES=3 python detective_active_model_3.py --dataset Weibo --output_dir result/EADA/6step2 \
#--train_epochs 30 --active_type EADA  --active_round 6 8 10 12 14
#
#
#CUDA_VISIBLE_DEVICES=3 python detective_active_model_3.py --dataset Pheme --output_dir result/EADA/8step2 \
#--train_epochs 30 --active_type EADA  --active_round 8 10 12 14 16
#
#
#CUDA_VISIBLE_DEVICES=3 python detective_active_model_3.py --dataset Weibo --output_dir result/EADA/8step2 \
#--train_epochs 30 --active_type EADA  --active_round 8 10 12 14 16


#CUDA_VISIBLE_DEVICES=6 python detective_active_model_3.py --dataset Pheme --output_dir result/EADA/10step2 \
#--train_epochs 30 --active_type EADA  --active_round 10 12 14 16 18
#
#
#CUDA_VISIBLE_DEVICES=7 python detective_active_model_3.py --dataset Weibo --output_dir result/EADA/10step2 \
#--train_epochs 30 --active_type EADA  --active_round 10 12 14 16 18
#
#
#CUDA_VISIBLE_DEVICES=6 python detective_active_model_3.py --dataset Pheme --output_dir result/EADA/10step4 \
#--train_epochs 30 --active_type EADA  --active_round 10 14 18 22 26
#
#
#CUDA_VISIBLE_DEVICES=7 python detective_active_model_3.py --dataset Weibo --output_dir result/EADA/10step4 \
#--train_epochs 30 --active_type EADA  --active_round 10 14 18 22 26