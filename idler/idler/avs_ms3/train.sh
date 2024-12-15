
python train.py --session_name "MS3_resnet" --visual_backbone "resnet" --max_epoches 30  --train_batch_size 1  --lr 0.0001  --tpavi_stages 0 1 2 3  --tpavi_va_flag --masked_av_flag  --masked_av_stages 0 1 2 3 --lambda_1 0.5 --kl_flag
  