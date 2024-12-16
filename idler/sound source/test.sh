setting='MS3'
visual_backbone="resnet" # "resnet" or "pvt"
# visual_backbone="pvt" # "resnet" or "pvt"

spring.submit arun --gpu -n1 --gres=gpu:1 --quotatype=auto --job-name="test_${setting}_${visual_backbone}" \
"
python test.py \
    --session_name ${setting}_${visual_backbone} \
    --visual_backbone ${visual_backbone} \
    --weights  "E:/AVSBench-maina/vs_scripts/avs_ms3/test_logs/V2_resnet_miou_and_fscore_best.pth" \
    --test_batch_size 4 \
    --tpavi_stages 0 1 2 3 \
    --tpavi_va_flag \
    --save_pred_mask \
"
python test9.py --session_name "MS3_resnet" --visual_backbone "resnet" --weights "F:\AVSBench-main\avs_scripts\avs_ms3\train_logs\MS3_resnet_20240427-213841\checkpoints\MS3_resnet_best.pth"  --test_batch_size 1 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask

python test.py --session_name "MS3_resnet" --visual_backbone "resnet" --weights "E:\AVSBench-gai\avs_scripts\avs_ms3\train_logs\MS3_resnet_20240427-213841\checkpoints\MS3_resnet_best.pth"  --test_batch_size 1 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask

python test.py --session_name "MS3_resnet" --visual_backbone "resnet" --weights "F:\AVSBench-main\avs_scripts\avs_ms3\train_logs\MS3_resnet_20241101-212920\checkpoints\MS3_resnet_best.pth"  --test_batch_size 1 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask
