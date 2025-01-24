#!/bin/bash

exec > infer_results.log 2>&1

# r101
torchrun --master_port=9909 --nproc_per_node=16 /colon_workspace/RT-DETR/rtdetrv2_pytorch/tools/train.py \
-c /colon_workspace/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml \
-r /colon_workspace/RT-DETR/rtdetrv2_pytorch/output/rtdetrv2_r101vd_6x_coco_20_epoch_no_amp/checkpoint0010.pth \
--test-only --use-amp

# r101, no pretrain
torchrun --master_port=9909 --nproc_per_node=16 /colon_workspace/RT-DETR/rtdetrv2_pytorch/tools/train.py \
-c /colon_workspace/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml \
-r /colon_workspace/RT-DETR/rtdetrv2_pytorch/output/rtdetrv2_r101vd_6x_coco_20_epoch_no_amp_no_pretrain/checkpoint0019.pth \
--test-only --use-amp

# r50
torchrun --master_port=9909 --nproc_per_node=16 /colon_workspace/RT-DETR/rtdetrv2_pytorch/tools/train.py \
-c /colon_workspace/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \
-r /colon_workspace/RT-DETR/rtdetrv2_pytorch/output/rtdetrv2_r50vd_6x_coco_20_epoch/checkpoint0009.pth \
--test-only --use-amp

# bash infer_rt_detr.sh > infer_results.log 2>&1