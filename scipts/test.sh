#!/usr/bin/env bash
set -e
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

CFG='configs/qdtrack-frcnn_r50_fpn_12e_bdd100k.py'
WORKDIR='ckpt'
CKPT="${WORKDIR}/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed.pth"
PKL="${WORKDIR}/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed.pkl"

GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) \
    tools/test.py $CFG $CKPT --launcher pytorch \
    --out $PKL --eval bbox track --fuse-conv-bn
