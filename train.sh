## active DA ADOSE:


MODEL_FILE="multimodal_active_model_3.py"
GPU_ID=0
DATASET='Pheme'
DIR='model3'
ACTIVE_DIR='active_1'
MAX_ITER=20
ROUND="6 8 10 12 14"
#
#
#
#CUDA_VISIBLE_DEVICES=${GPU_ID} python ${MODEL_FILE} --output_dir result/${DIR}/${ACTIVE_DIR}/random \
#--dataset ${DATASET} --active_type random --active_round ${ROUND} --max_iter ${MAX_ITER}
#
#
#CUDA_VISIBLE_DEVICES=${GPU_ID} python ${MODEL_FILE} --output_dir result/${DIR}/${ACTIVE_DIR}/entropy \
#--dataset ${DATASET} --active_type entropy --active_round ${ROUND} --max_iter ${MAX_ITER}
#
#
##CUDA_VISIBLE_DEVICES=${GPU_ID} python ${MODEL_FILE} --output_dir result/${DIR}/${ACTIVE_DIR}/ldms \
##--dataset ${DATASET} --active_type ldms --active_round ${ROUND} --max_iter ${MAX_ITER}
#
#
CUDA_VISIBLE_DEVICES=${GPU_ID} python ${MODEL_FILE} --output_dir result/${DIR}/${ACTIVE_DIR}/ldms8 \
--dataset ${DATASET} --active_type ldms8 --active_round ${ROUND} --max_iter ${MAX_ITER}






## RDCM UDA:
#
#CUDA_VISIBLE_DEVICES=0 python drive_ourmodel.py --data='Pheme' --tag='DA'  --da='True' --log='result/RDCM-UDA/pheme'
#
##CUDA_VISIBLE_DEVICES=0 python drive_ourmodel.py --data='Weibo' --tag='DA'  --da='True' --log='result/RDCM-UDA/weibo'
#
## ADOSE UDA:
#
##CUDA_VISIBLE_DEVICES=0 python multimodal_active_model_3.py --output_dir result/model3-UDA/ --dataset Pheme \
##--active_round 100 101 --max_iter 20
#
#
#CUDA_VISIBLE_DEVICES=0 python multimodal_active_model_3.py --output_dir result/model3-UDA/ --dataset Weibo \
#--active_round 100 101 --max_iter 45



## Detective ADA:
#
#CUDA_VISIBLE_DEVICES=4 python detective_active_model_3.py --dataset Pheme --output_dir result/detective \
#--train_epochs 30 --active_type detective
#
#
#CUDA_VISIBLE_DEVICES=5 python detective_active_model_3.py --dataset Weibo --output_dir result/detective \
#--train_epochs 30 --active_type detective
#
#
## EADA ADA:
#
#CUDA_VISIBLE_DEVICES=6 python detective_active_model_3.py --dataset Pheme --output_dir result/EADA \
#--train_epochs 30 --active_type EADA
#
#
#CUDA_VISIBLE_DEVICES=7 python detective_active_model_3.py --dataset Weibo --output_dir result/EADA \
#--train_epochs 30 --active_type EADA
#
#
## CLUE ADA:
#CUDA_VISIBLE_DEVICES=4 python baseline_clue_model.py --dataset Pheme --output_dir result/CLUE
#
#CUDA_VISIBLE_DEVICES=5 python baseline_clue_model.py --dataset Weibo --output_dir result/CLUE

## ablation:
## Pheme:
#CUDA_VISIBLE_DEVICES=0 python multimodal_active_model_3.py --output_dir result/ablation/abla1 \
#--dataset Pheme --active_type ablation1 --active_round 6 8 10 12 14 --max_iter 20
#
#
#CUDA_VISIBLE_DEVICES=0 python multimodal_active_model_3.py --output_dir result/ablation/abla2 \
#--dataset Pheme --active_type ablation2 --active_round 6 8 10 12 14 --max_iter 20
#
#
#CUDA_VISIBLE_DEVICES=0 python ablation_model_3.py --output_dir result/ablation/abla3 \
#--dataset Pheme --active_round 6 8 10 12 14 --max_iter 20









