#!/usr/bin/env bash

model_dir=$(dirname "$0")/../models/
model_name=$1

mkdir -p ${model_dir}

# Mask R-CNN
echo "Using model: ${model_name}"
echo ""
if [ -f "${model_dir}/${model_name}.pth" ]; then
  echo "-> Model found locally."
else
  echo "-> Pulling remote model."
  case ${model_name} in
    "mask_rcnn_coco")
      wget https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth -O ${model_dir}/${model_name}.pth
      ;;
    *)
      echo "--> This model is not supported."
      ;;
  esac
fi
