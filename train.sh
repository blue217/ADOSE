

MODEL_FILE="multimodal_active_model.py"
GPU_ID=3
DATASET='Pheme'
DIR='model3'
ACTIVE_DIR='active_1'
MAX_ITER=20
ROUND="6 8 10 12 14"

CUDA_VISIBLE_DEVICES=${GPU_ID} python ${MODEL_FILE} --output_dir result/${DIR}/${ACTIVE_DIR}/ldms12 \
--dataset ${DATASET} --active_type ldms12 --active_round ${ROUND} --max_iter ${MAX_ITER}



