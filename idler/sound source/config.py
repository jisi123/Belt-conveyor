from easydict import EasyDict as edict
import yaml
import pdb

"""
default config
"""
cfg = edict()
cfg.BATCH_SIZE = 1 # default 16设置默认批量大小为 4
cfg.LAMBDA_1 = 5 # default: 5设置默认的 λ1 值为 5。
cfg.MASK_NUM = 1 # 5 表示完全监督，1 表示弱监督

###############################
# 1. `--session_name`: 指定训练会话的名称，默认为"MS3"。
#
# 2. `--visual_backbone`: 指定视觉模型的主干网络，可以是"resnet50"或"pvt-v2"，默认为"resnet"。
#
# 3. `--train_batch_size`: 训练时的批处理大小，默认为4。
#
# 4. `--val_batch_size`: 验证时的批处理大小，默认为1。
#
# 5. `--max_epoches`: 最大训练轮数，默认为15。
#
# 6. `--lr`: 学习率，默认为0.0001。
#
# 7. `--num_workers`: 数据加载时的工作线程数，默认为0。
#
# 8. `--wt_dec`: 权重衰减，默认为5e-4。
#
# 9. `--masked_av_flag`: 是否使用额外的自监督/掩码视觉损失，默认为True。
#
# 10. `--lambda_1`: 平衡L4损失的权重，默认为0。
#
# 11. `--masked_av_stages`: 在哪些阶段计算自监督/掩码视觉损失，默认为空列表。
#
# 12. `--threshold_flag`: 是否对生成的掩码进行阈值处理，默认为False。
#
# 13. `--mask_pooling_type`: 预测掩码的下采样方式，默认为'avg'（平均池化）。
#
# 14. `--norm_fea_flag`: 是否归一化音频-视觉特征，默认为False。
#
# 15. `--closer_flag`: 是否使用更接近的损失函数，默认为False。
#
# 16. `--euclidean_flag`: 是否使用欧几里得距离计算损失，默认为False。
#
# 17. `--kl_flag`: 是否使用KL散度损失，默认为True。
#
# 18. `--load_s4_params`: 是否使用S4参数进行初始化，默认为False。
#
# 19. `--trained_s4_model_path`: 预训练的S4模型路径，默认为空。
#
# 20. `--tpavi_stages`: 在哪些阶段添加TPAVI块，默认为空列表。
#
# 21. `--tpavi_vv_flag`: 是否使用视觉-视觉自注意力，默认为False。
#
# 22. `--tpavi_va_flag`: 是否使用视觉-音频交叉注意力，默认为True。
#
# 23. `--weights`: 训练模型的路径，默认为空。
#
# 24. `--log_dir`: 日志文件的保存目录，默认为'./train_logs'。


# TRAIN
cfg.TRAIN = edict()
# cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR = True
# cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = "../../pretrained_backbones/vggish-10086976.pth"
# cfg.TRAIN.PREPROCESS_AUDIO_TO_LOG_MEL = False
# cfg.TRAIN.POSTPROCESS_LOG_MEL_WITH_PCA = False
# cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH = "../../pretrained_backbones/vggish_pca_params-970ea276.pth"
# cfg.TRAIN.FREEZE_VISUAL_EXTRACTOR = True
# cfg.TRAIN.PRETRAINED_RESNET50_PATH = "../../pretrained_backbones/resnet50-19c8e357.pth"
# cfg.TRAIN.PRETRAINED_PVTV2_PATH = "../../pretrained_backbones/pvt_v2_b5.pth"

cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = "F:/AVSBench-main/pretrained_backbones/vggish-10086976.pth"
cfg.TRAIN.PREPROCESS_AUDIO_TO_LOG_MEL = False
cfg.TRAIN.POSTPROCESS_LOG_MEL_WITH_PCA = False
cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH = r"F:\AVSBench-main\pretrained_backbones\vggish_pca_params-970ea276.pth"
cfg.TRAIN.FREEZE_VISUAL_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_RESNET50_PATH = "F:/AVSBench-main/pretrained_backbones/resnet50-19c8e357.pth"
cfg.TRAIN.PRETRAINED_PVTV2_PATH = "F:/AVSBench-main/pretrained_backbones/pvt_v2_b5.pth"


cfg.TRAIN.FINE_TUNE_SSSS = False
cfg.TRAIN.PRETRAINED_S4_aAVS_WO_TPAVI_PATH = "../avs_s4/train_logs/checkpoints/checkpoint_xxx.pth.tar"
cfg.TRAIN.PRETRAINED_S4_AVS_WITH_TPAVI_PATH = "../avs_s4/train_logs/checkpoints/checkpoint_xxx.pth.tar"

###############################
# DATA
cfg.DATA = edict()
# cfg.DATA.ANNO_CSV = "../../avsbench_data/det/det/det.csv"
# cfg.DATA.DIR_IMG = "../../avsbench_data//Multi-sources/ms3_data/visual_frames"
# cfg.DATA.DIR_AUDIO_LOG_MEL = "../../avsbench_data//Multi-sources/ms3_data/audio_log_mel"
# cfg.DATA.DIR_MASK = "../../avsbench_data//Multi-sources/ms3_data/gt_masks"
# cfg.DATA.IMG_SIZE = (224, 224)

cfg.DATA.ANNO_CSV = "F:/AVSBench-main/avsbench_data/det/det.csv"
cfg.DATA.DIR_IMG = "F:/AVSBench-main/avsbench_data/det/visual_frames"
cfg.DATA.DIR_AUDIO_LOG_MEL = "F:/AVSBench-main/avsbench_data/det/audio_log_mel"
cfg.DATA.DIR_MASK = "F:/AVSBench-main/avsbench_data/det/gt_masks"
cfg.DATA.IMG_SIZE = (224, 224)
###############################
# cfg.DATA.RESIZE_PRED_MASK = True
# cfg.DATA.SAVE_PRED_MASK_IMG_SIZE = (224, 224) # (width, height)
# def _edict2dict(dest_dict, src_edict):
#     if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
#         for k, v in src_edict.items():
#             if not isinstance(v, edict):
#                 dest_dict[k] = v
#             else:
#                 dest_dict[k] = {}
#                 _edict2dict(dest_dict[k], v)
#     else:
#         return
#
# def gen_config(config_file):
#     cfg_dict = {}
#     _edict2dict(cfg_dict, cfg)
#     with open(config_file, 'w') as f:
#         yaml.dump(cfg_dict, f, default_flow_style=False)
#
# def _update_config(base_cfg, exp_cfg):
#     if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
#         for k, v in exp_cfg.items():
#             if k in base_cfg:
#                 if not isinstance(v, dict):
#                     base_cfg[k] = v
#                 else:
#                     _update_config(base_cfg[k], v)
#             else:
#                 raise ValueError("{} not exist in config.py".format(k))
#     else:
#         return
#
# def update_config_from_file(filename):
#     exp_config = None
#     with open(filename) as f:
#         exp_config = edict(yaml.safe_load(f))
#         _update_config(cfg, exp_config)

if __name__ == "__main__":
    print(cfg)
    pdb.set_trace()