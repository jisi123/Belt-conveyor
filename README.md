# Idler Fault Detection for Belt Conveyors: Visual and Auditory Dual-Mode Fusion

## Overview

This repository contains the code and instructions for an accurate fault location method for idlers in belt conveyors using a fusion of visual and auditory dual-modal information. The system utilizes ResNet for image feature extraction and VGGish for audio feature extraction, followed by an audiovisual fusion module to align and merge the features. Pixel classification is then used for pixel-level segmentation of faulty idlers.

## Key Features

- Belt Conveyor
- Idler Fault Detection
- Multimodal Detection

## Data and Model Preparation

### Dataset

The dataset was collected from a self-built experimental platform. You can download the public dataset from the following links:

- [Training and Testing Dataset](https://pan.baidu.com/s/1qMD511uPA6ULsuBPvvhOqQ?pwd=6v6q)
- [Idler Fault Dataset for Training and Testing](https://pan.baidu.com/s/11zCxpBRsnf5P44pGaQUDEQ?pwd=t4tu)

Please download the dataset and place it in the `data` directory.

### Backbones

Pretrained backbones for ResNet50 (vision) and VGGish (audio) can be downloaded from:

- [Pretrained Models](https://pan.baidu.com/s/1XCxI3THok2Kdne-t3o5OnA?pwd=22qt)

Place the downloaded models in the `pretrained` directory.

### Train Weights

If you do not wish to train the weights from scratch, you can use the provided training weights:

- [Training Weights](https://pan.baidu.com/s/1GJlftVfbeqALWjDvtnLh3A?pwd=mdtr)

Place the weights in the specified location in `test.sh` with the `--weights` flag.

## Usage

### Train Model

To train the model, navigate to the `network/avs_s3` directory and run the following commands:

```bash
cd network/avs_s3
bash train.sh
```

or

```bash
python train.py \
  --session_name "MS3_resnet" \
  --visual_backbone "resnet" \
  --max_epoches 30 \
  --train_batch_size 1 \
  --lr 0.0001 \
  --tpavi_stages 0 1 2 3 \
  --tpavi_va_flag \
  --masked_av_flag \
  --masked_av_stages 0 1 2 3 \
  --lambda_1 0.5\
  --kl_flag\
```

### Test Model

To test the model, navigate to the `network/avs_s3` directory and run the following commands:

```bash
cd network/avs_s3
bash test.sh
```

or

```bash
python test.py \
  --session_name "MS3_resnet" \
  --visual_backbone "resnet" \
  --weights "MS3_resnet_best.pth" \
  --test_batch_size 1 \
  --tpavi_stages 0 1 2 3 \
  --tpavi_va_flag \
  --save_pred_mask\
```

## Results

The test results can be found in the following path:

```bash
cd ../network/avs_ms3/test_logs
```

## Documentation

For detailed documentation, please refer to the [Project Documentation](https://github.com/your-username/your-repository-name/blob/main/Documentation.md).

## Contact

If you have any questions, please feel free to contact us at:

- Email: 15303717926@163.com

---

