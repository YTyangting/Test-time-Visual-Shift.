#!/bin/bash

#cd ../..

# custom config
DATA=".../datasets/"
TRAINER=TVS

DATASET=$1
DEVICE=7
WEIGHTSPATH='output/imagenet/TVS/vit_b16_c2_ep5_batch4_4ctx_cross_datasets_16shots'
CFG=DG_vit_b16_c2_ep5_batch4_4ctx_cross_datasets
SHOTS=16
LOADEP=5

for SEED in 1
do
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --device ${DEVICE}\
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/TVS/${CFG}.yaml \
    --output-dir output/evaluation/TVS_${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED} \
    --model-dir ${WEIGHTSPATH}/seed${SEED} \
    --load-epoch 5 \
    --tpt \
    DATASET.NUM_SHOTS ${SHOTS} 
done
