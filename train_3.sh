
MODEL_FILE="multimodal_active_model_3.py"
GPU_ID=2
DATASET='Pheme'
DIR='model3'
ACTIVE_DIR='active_1'
MAX_ITER=20
ROUND="6 8 10 12 14"

CUDA_VISIBLE_DEVICES=${GPU_ID} python ${MODEL_FILE} --output_dir result/${DIR}/${ACTIVE_DIR}/ldms10 \
--dataset ${DATASET} --active_type ldms10 --active_round ${ROUND} --max_iter ${MAX_ITER}

## Detective ADA:
#
#CUDA_VISIBLE_DEVICES=3 python detective_active_model_3.py --dataset Pheme --output_dir result/detective/6step2 \
#--train_epochs 30 --active_type detective --active_round 6 8 10 12 14
#
#
#CUDA_VISIBLE_DEVICES=3 python detective_active_model_3.py --dataset Weibo --output_dir result/detective/6step2 \
#--train_epochs 30 --active_type detective --active_round 6 8 10 12 14
#
#
#CUDA_VISIBLE_DEVICES=3 python detective_active_model_3.py --dataset Pheme --output_dir result/detective/8step2 \
#--train_epochs 30 --active_type detective --active_round 8 10 12 14 16
#
#
#CUDA_VISIBLE_DEVICES=3 python detective_active_model_3.py --dataset Weibo --output_dir result/detective/8step2 \
#--train_epochs 30 --active_type detective --active_round 8 10 12 14 16


#CUDA_VISIBLE_DEVICES=4 python detective_active_model_3.py --dataset Pheme --output_dir result/detective/10step2 \
#--train_epochs 30 --active_type detective --active_round 10 12 14 16 18
#
#
#CUDA_VISIBLE_DEVICES=4 python detective_active_model_3.py --dataset Weibo --output_dir result/detective/10step2 \
#--train_epochs 30 --active_type detective --active_round 10 12 14 16 18
#
#
#CUDA_VISIBLE_DEVICES=4 python detective_active_model_3.py --dataset Pheme --output_dir result/detective/10step4 \
#--train_epochs 30 --active_type detective --active_round 10 14 18 22 26
#
#
#CUDA_VISIBLE_DEVICES=4 python detective_active_model_3.py --dataset Weibo --output_dir result/detective/10step4 \
#--train_epochs 30 --active_type detective --active_round 10 14 18 22 26


