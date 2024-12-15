from easydict import EasyDict as edict
import yaml
import pdb

"""
default config
"""
cfg = edict()
cfg.BATCH_SIZE = 1 # default 16设置默认批量大小为 4
cfg.LAMBDA_1 = 5 # default: 5设置默认的 λ1 值为 5。
cfg.MASK_NUM = 5 # 5 表示完全监督，1 表示弱监督

###############################
# TRAIN
cfg.TRAIN = edict()



###############################
# DATA


# cfg.DATA.IMG_SIZE = (224, 224)

###############################
# cfg.DATA.RESIZE_PRED_MASK = True
# cfg.DATA.SAVE_PRED_MASK_IMG_SIZE = (448, 448) # (width, height)

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
