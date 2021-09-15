# 我的样本挑选实验


#############################################################
##                       uniform nosiy                     ##
#############################################################
for i in 0.2 0.4 0.6 0.8
do
    export CUDA_VISIBLE_DEVICES=0

    PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
    ALGORITHM="sample_selection"
    SOURCE="amazon"         # amazon, dslr, webcam
    NOISY_TYPE="uniform"    # uniform, pair, ood, feature, feature_pair, feature_uniform, ood_feature, ood_uniform, ood_feature_uniform...
    NOISY_RATE=$i
    DATASET="Office-31"

    # LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
    STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"
    # echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
    python trainer/sample_selection.py \
        --config ${PROJ_ROOT}/config/dann.yml \
        --dataset ${DATASET} \
        --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
        --stats_file ${STATS_FILE} \
        --noisy_rate ${NOISY_RATE} \
        --noisy_type ${NOISY_TYPE} \
        # >> ${LOG_FILE}  2>&1
done