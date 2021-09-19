# Pcyc 实验


#############################################################
##                       uniform_A2W                       ##
#############################################################
# for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
for i in 0.2
do
    export CUDA_VISIBLE_DEVICES=0

    PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
    ALGORITHM="PCYC"
    PROJ_NAME="uniform_A2W"
    SOURCE="amazon"         # amazon, dslr, webcam
    TARGET="webcam"         
    NOISY_TYPE="uniform"    # uniform, ood, feature, feature_uniform, ood_feature, ood_uniform, ood_feature_uniform...
    NOISY_RATE=$i
    DATASET="Office-31"

    LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
    STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"
    # echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
    python trainer/Periodic_cycle_train.py \
        --config ${PROJ_ROOT}/config/dann.yml \
        --dataset ${DATASET} \
        --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
        --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
        --stats_file ${STATS_FILE} \
        --noisy_rate ${NOISY_RATE} \
        --noisy_type ${NOISY_TYPE} \
        >> ${LOG_FILE}  2>&1
done


#############################################################
##                          amazon                         ##
##                         ood nosiy                       ##
#############################################################
# for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
# do
#     export CUDA_VISIBLE_DEVICES=0

#     PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
#     ALGORITHM="sample_selection"
#     SOURCE="amazon"         # amazon, dslr, webcam
#     NOISY_TYPE="ood"    # uniform, ood, feature, feature_uniform, ood_feature, ood_uniform, ood_feature_uniform...
#     NOISY_RATE=$i
#     DATASET="Office-31"

#     LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
#     STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"
#     # echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
#     python trainer/sample_selection.py \
#         --config ${PROJ_ROOT}/config/dann.yml \
#         --dataset ${DATASET} \
#         --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#         --stats_file ${STATS_FILE} \
#         --noisy_rate ${NOISY_RATE} \
#         --noisy_type ${NOISY_TYPE} \
#         >> ${LOG_FILE}  2>&1
# done


#############################################################
##                          amazon                         ##
##                   feature_uniform nosiy                 ##
#############################################################
# for i in 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6
# do
#     export CUDA_VISIBLE_DEVICES=0

#     PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
#     ALGORITHM="sample_selection"
#     SOURCE="amazon"         # amazon, dslr, webcam
#     NOISY_TYPE="feature_uniform"    # uniform, ood, feature, feature_uniform, ood_feature, ood_uniform, ood_feature_uniform...
#     NOISY_RATE=$i
#     DATASET="Office-31"

#     LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
#     STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"
#     # echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
#     python trainer/sample_selection.py \
#         --config ${PROJ_ROOT}/config/dann.yml \
#         --dataset ${DATASET} \
#         --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#         --stats_file ${STATS_FILE} \
#         --noisy_rate ${NOISY_RATE} \
#         --noisy_type ${NOISY_TYPE} \
#         >> ${LOG_FILE}  2>&1
# done


#############################################################
##                          amazon                         ##
##                     ood_uniform nosiy                   ##
#############################################################
# for i in 0.2 0.4 0.6 0.8 1.0
# do
#     export CUDA_VISIBLE_DEVICES=0

#     PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
#     ALGORITHM="sample_selection"
#     SOURCE="amazon"         # amazon, dslr, webcam
#     NOISY_TYPE="ood_uniform"    # uniform, ood, feature, feature_uniform, ood_feature, ood_uniform, ood_feature_uniform...
#     NOISY_RATE=$i
#     DATASET="Office-31"

#     LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
#     STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"
#     # echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
#     python trainer/sample_selection.py \
#         --config ${PROJ_ROOT}/config/dann.yml \
#         --dataset ${DATASET} \
#         --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#         --stats_file ${STATS_FILE} \
#         --noisy_rate ${NOISY_RATE} \
#         --noisy_type ${NOISY_TYPE} \
#         >> ${LOG_FILE}  2>&1
# done


#############################################################
##                          amazon                         ##
##                     ood_feature nosiy                   ##
#############################################################
# for i in 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6
# do
#     export CUDA_VISIBLE_DEVICES=0

#     PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
#     ALGORITHM="sample_selection"
#     SOURCE="amazon"         # amazon, dslr, webcam
#     NOISY_TYPE="ood_feature"    # uniform, ood, feature, feature_uniform, ood_feature, ood_uniform, ood_feature_uniform...
#     NOISY_RATE=$i
#     DATASET="Office-31"

#     LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
#     STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"
#     # echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
#     python trainer/sample_selection.py \
#         --config ${PROJ_ROOT}/config/dann.yml \
#         --dataset ${DATASET} \
#         --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#         --stats_file ${STATS_FILE} \
#         --noisy_rate ${NOISY_RATE} \
#         --noisy_type ${NOISY_TYPE} \
#         >> ${LOG_FILE}  2>&1
# done


#############################################################
##                          amazon                         ##
##                 ood_feature_uniform nosiy               ##
#############################################################
# for i in 0.3 0.6 0.9 1.2 1.5
# do
#     export CUDA_VISIBLE_DEVICES=0

#     PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
#     ALGORITHM="sample_selection"
#     SOURCE="amazon"         # amazon, dslr, webcam
#     NOISY_TYPE="ood_feature_uniform"    # uniform, ood, feature, feature_uniform, ood_feature, ood_uniform, ood_feature_uniform...
#     NOISY_RATE=$i
#     DATASET="Office-31"

#     LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
#     STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"
#     # echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
#     python trainer/sample_selection.py \
#         --config ${PROJ_ROOT}/config/dann.yml \
#         --dataset ${DATASET} \
#         --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#         --stats_file ${STATS_FILE} \
#         --noisy_rate ${NOISY_RATE} \
#         --noisy_type ${NOISY_TYPE} \
#         >> ${LOG_FILE}  2>&1
# done