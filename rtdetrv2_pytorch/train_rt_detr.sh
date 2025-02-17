# resnet101 train, fp 32
torchrun --master_port=9909 --nproc_per_node=16 tools/train.py \
-c /colon_workspace/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml \
 --seed=0 &> log_rtdetrv2_r101vd_6x_coco_20_epoch_no_amp.txt 2>&1 &

# resnet101 train, fp 32 no pretrain

torchrun --master_port=9909 --nproc_per_node=16 tools/train.py \
-c /colon_workspace/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml \
 --seed=0 &> log_rtdetrv2_r101vd_6x_coco_20_epoch_no_amp_no_pretrain.txt 2>&1 &

# resnet101 train, use coco pretrained
torchrun --master_port=9909 --nproc_per_node=16 tools/train.py \
-c /colon_workspace/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml \
-t /colon_workspace/rtdetrv2_r101vd_6x_coco_from_paddle.pth \
 --seed=0 &> log_rtdetrv2_r50vd_6x_coco_20_epoch_no_amp_coco_pretrain.txt 2>&1 &

# timm resnet50 nv imagenet pretrain, fp 32
# scratch
torchrun --master_port=9909 --nproc_per_node=16 tools/train.py \
-c /colon_workspace/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_timm_r50vd_6x_colon.yml \
 --seed=0 &> log_640_rtdetrv2_timm_r50vd_6x_colon_20_epoch_no_freeze_bn_queries_300_all_data_weighted_sample.txt 2>&1 &

 # tune with all neg images, freeze bn, conv1
CUDA_VISIBLE_DEVICES=4,5,6,7,8,9,10,11,12,13,14,15 torchrun --master_port=9909 --nproc_per_node=12 tools/train.py \
-c /colon_workspace/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_timm_r50vd_6x_colon.yml \
--tuning /colon_workspace/RT-DETR/rtdetrv2_pytorch/output/rtdetrv2_timm_r50vd_6x_colon_20_epoch_no_freeze_bn_lr1e-4/checkpoint0010.pth \
 --seed=0 &> log_640_rtdetrv2_timm_r50vd_6x_colon_20_epoch_finetune_pos_backbone_pos1_neg1.txt 2>&1 &

# test
CUDA_VISIBLE_DEVICES=4,5,6,7,8,9,10,11,12,13,14,15 torchrun --master_port=9909 --nproc_per_node=12 tools/train.py \
-c /colon_workspace/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_timm_r50vd_6x_colon.yml \
-r /colon_workspace/RT-DETR/rtdetrv2_pytorch/output/log_640_rtdetrv2_timm_r50vd_6x_colon_20_epoch_finetune_pos_backbone_pos1_neg1/best.pth \
--test-only --use-amp

# tune with all neg images

# rtdetrv2_timm_r50vd_6x_colon_20_epoch_no_freeze_bn_lr1e-4_all_data_pos1_neg3/checkpoint0002.pth

torchrun --master_port=9909 --nproc_per_node=16 tools/train.py \
-c /colon_workspace/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_timm_r50vd_6x_colon.yml \
--tuning /colon_workspace/RT-DETR/rtdetrv2_pytorch/output/rtdetrv2_timm_r50vd_6x_colon_20_epoch_no_freeze_bn_lr1e-4/checkpoint0010.pth \
 --seed=0 &> log_640_rtdetrv2_timm_r50vd_6x_colon_20_epoch_no_freeze_bn_lr1e-1_all_data_pos1_neg3_freeze_backbone.txt 2>&1 &

torchrun --master_port=9909 --nproc_per_node=16 tools/train.py \
-c /colon_workspace/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_timm_r50vd_6x_colon.yml \
-r /colon_workspace/RT-DETR/rtdetrv2_pytorch/output/rtdetrv2_timm_r50vd_6x_colon_20_epoch_no_freeze_bn_lr1e-4_all_data_pos1_neg3/checkpoint0001.pth \
--test-only --use-amp


# finetune
torchrun --master_port=9909 --nproc_per_node=16 tools/train.py \
-c /colon_workspace/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml \
 --use-amp --seed=0 --tuning /colon_workspace/RT-DETR/rtdetrv2_pytorch/output/rtdetrv2_r101vd_6x_coco_640_20_epochs/best.pth &> log_640_epoch20_lr_fix_1e-5backbone_finetune.txt 2>&1 &

# pip install numpy==1.26.4

# onnx
python tools/export_onnx.py \
-c /colon_workspace/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_timm_r50vd_6x_colon.yml \
-r /colon_workspace/RT-DETR/rtdetrv2_pytorch/output/rtdetrv2_timm_r50vd_6x_colon_20_epoch_no_freeze_bn_lr1e-4_all_data_pos1_neg3/checkpoint0002.pth \
--check --output_file rtdetrv2_timm_r50_nvimagenet_pretrained_neg_finetune.onnx

# switch to bhwc
pip install onnx_graphsurgeon
python tools/graph_surgeon.py rtdetrv2_timm_r50_nvimagenet_pretrained_neg_finetune.onnx rtdetrv2_timm_r50_nvimagenet_pretrained_neg_finetune_bhwc.onnx

# onnx inference
pip install onnxruntime

python references/deploy/rtdetrv2_onnxruntime.py \
--onnx-file=rtdetrv2_timm_r50_nvimagenet_pretrained_neg_finetune.onnx \
--im-file /colon_workspace/real-colon-dataset/real_colon_dataset_coco_fmt_3subsets_poslesion1000_negratio0/test_images/001-014_21220.jpg

# docker
docker run --rm -it --gpus=all --ipc=host -v /raid/colon_reproduce:/colon_workspace holoscan:polyp-det
docker run --rm -it --gpus=all --ipc=host -v /raid/colon_reproduce:/colon_workspace nvcr.io/nvidia/clara-holoscan/holoscan:v2.7.0-dgpu-polyp-det
# onnx to trt with dynamic shapes
trtexec --onnx=rtdetrv2_timm_r50_nvimagenet_pretrained_neg_finetune.onnx --saveEngine=rt_detrv2_timm_r50_nvimagenet_pretrained_neg_finetune.trt \
--minShapes=images:1x3x640x640,orig_target_sizes:1x2 \
--optShapes=images:32x3x640x640,orig_target_sizes:32x2 \
--maxShapes=images:32x3x640x640,orig_target_sizes:32x2 \
--allowGPUFallback

# trt inference
python references/deploy/rtdetrv2_tensorrt.py \
-trt /colon_workspace/RT-DETR/rtdetrv2_pytorch/rt_detrv2_timm_r50_nvimagenet_pretrained_neg_finetune.trt \
--im-file /colon_workspace/real-colon-dataset/real_colon_dataset_coco_fmt_3subsets_poslesion1000_negratio0/test_images/001-014_21220.jpg

# trt inference a folder
CUDA_VISIBLE_DEVICES=0 python references/deploy/trt_inference_all_data.py \
-trt /colon_workspace/RT-DETR/rtdetrv2_pytorch/rt_detrv2_timm_r50_nvimagenet_pretrained_neg_finetune.trt \
-f /colon_workspace/real-colon-dataset/real_colon_dataset_coco_fmt_3subsets_poslesion1000_negratio0/test_images \
-a /colon_workspace/real-colon-dataset/real_colon_dataset_coco_fmt_3subsets_poslesion1000_negratio0/test_ann.json \
-b 128

# jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root


## holoscan
# convert video
ffmpeg -i /colon_workspace/colon_videos/004-015_first_900_original.mp4 -pix_fmt rgb24 -f rawvideo pipe:1 | python3 scripts/convert_video_to_gxf_entities.py --width 1164 --height 1034 --channels 3 --framerate 30


