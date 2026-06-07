#!/bin/bash

#cd ../..

# custom config
DATA="/datasets/"
TRAINER=TasTPT

DATASET=$1
#SEED=$2
WEIGHTSPATH='output/imagenet/SaVTP/vit_b16_c2_ep5_batch4_4ctx_cross_datasets_16shots'
CFG=DG_rn50_c2_ep5_batch4_4ctx_cross_datasets
SHOTS=16
LOADEP=5
for SEED in 1
do
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/SaVTPTPT/${CFG}.yaml \
    --output-dir output/evaluation/SaVPT_${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED} \
    --load-epoch 5 \
    --tpt \
    DATASET.NUM_SHOTS ${SHOTS} 
done
#--model-dir ${WEIGHTSPATH}/seed${SEED} \