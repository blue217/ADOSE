#CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py --data='Twitter' --tag='DA'  --da='True'  --lr=0.001 --log='twitter-da'   --phase='train' --epochs=30  --max_iter=10 --lambda1=1 --tsigma="2#4#8#16" --vsigma="2#4#8#16"  --lambda2=1 --temperature=0.5 --threshold=0.5 --ctsize=64
#
##  ADOSE:
#
MODEL_FILE="multimodal_active_model_3.py"
GPU_ID=1
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
#CUDA_VISIBLE_DEVICES=${GPU_ID} python ${MODEL_FILE} --output_dir result/${DIR}/${ACTIVE_DIR}/ldms \
#--dataset ${DATASET} --active_type ldms --active_round ${ROUND} --max_iter ${MAX_ITER}
#
#
CUDA_VISIBLE_DEVICES=${GPU_ID} python ${MODEL_FILE} --output_dir result/${DIR}/${ACTIVE_DIR}/ldms9 \
--dataset ${DATASET} --active_type ldms9 --active_round ${ROUND} --max_iter ${MAX_ITER}


# RDCM UDA:

#CUDA_VISIBLE_DEVICES=1 python drive_ourmodel.py --data='Pheme' --tag='DA'  --da='True' --log='result/RDCM-UDA/pheme'
#
#CUDA_VISIBLE_DEVICES=1 python drive_ourmodel.py --data='Weibo' --tag='DA'  --da='True' --log='result/RDCM-UDA/weibo'




## ADOSE UDA:
#
#CUDA_VISIBLE_DEVICES=2 python multimodal_active_model_3.py --output_dir result/model3-UDA/ --dataset Pheme \
#--active_round 100 101 --max_iter 20
#
#
#CUDA_VISIBLE_DEVICES=2 python multimodal_active_model_3.py --output_dir result/model3-UDA/ --dataset Weibo \
#--active_round 100 101 --max_iter 45

## ablation:
## Pheme:
#CUDA_VISIBLE_DEVICES=5 python multimodal_active_model_3.py --output_dir result/ablation/abla1 \
#--dataset Pheme --active_type ablation1 --active_round 6 8 10 12 14 --max_iter 20
#
#
#CUDA_VISIBLE_DEVICES=5 python multimodal_active_model_3.py --output_dir result/ablation/abla2 \
#--dataset Pheme --active_type ablation2 --active_round 6 8 10 12 14 --max_iter 20
#
#
#CUDA_VISIBLE_DEVICES=5 python ablation_model_3.py --output_dir result/ablation/abla3 \
#--dataset Pheme --active_round 6 8 10 12 14 --max_iter 20
#
# Weibo:

#CUDA_VISIBLE_DEVICES=1 python multimodal_active_model_3.py --output_dir result/ablation/abla1 \
#--dataset Weibo --active_type ablation1 --active_round 6 8 10 12 14 --max_iter 45
#
#
#CUDA_VISIBLE_DEVICES=1 python multimodal_active_model_3.py --output_dir result/ablation/abla2 \
#--dataset Weibo --active_type ablation2 --active_round 6 8 10 12 14 --max_iter 45
#
#
#CUDA_VISIBLE_DEVICES=1 python ablation_model_3.py --output_dir result/ablation/abla3 \
#--dataset Weibo --active_round 6 8 10 12 14 --max_iter 45




