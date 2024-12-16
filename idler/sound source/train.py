import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging

from config import cfg
from dataloader import MS3Dataset
from torchvggish import vggish
from loss import IouSemanticAwareLoss

from utils import pyutils
from utils.utility import logger, mask_iou
from utils.system import setup_logging
import pdb

#定义 audio_extractor 类，继承自 torch.nn.Module。这个类使用 VGGish 模型提取音频特征：
class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea

#使用 argparse 解析命令行参数：
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="MS3", type=str, help="the MS3 setting")
    parser.add_argument("--visual_backbone", default="resnet", type=str, help="use resnet50 or pvt-v2 as the visual backbone")

    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--max_epoches", default=15, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)

    parser.add_argument('--masked_av_flag', action='store_true', default=True, help='additional sa/masked_va loss for five frames')
    parser.add_argument("--lambda_1", default=0, type=float, help='weight for balancing l4 loss')
    parser.add_argument("--masked_av_stages", default=[], nargs='+', type=int, help='compute sa/masked_va loss in which stages: [0, 1, 2, 3]')
    parser.add_argument('--threshold_flag', action='store_true', default=False, help='whether thresholding the generated masks')
    parser.add_argument("--mask_pooling_type", default='avg', type=str, help='the manner to downsample predicted masks')
    parser.add_argument('--norm_fea_flag', action='store_true', default=False, help='normalize audio-visual features')
    parser.add_argument('--closer_flag', action='store_true', default=False, help='use closer loss for masked_va loss')
    parser.add_argument('--euclidean_flag', action='store_true', default=False, help='use euclidean distance for masked_va loss')
    parser.add_argument('--kl_flag', action='store_true', default=True, help='use kl loss for masked_va loss')

    parser.add_argument("--load_s4_params", action='store_true', default=False, help='use S4 parameters for initilization')
    parser.add_argument("--trained_s4_model_path", type=str, default='', help='pretrained S4 model')
# 设置随机种子
    parser.add_argument("--tpavi_stages", default=[], nargs='+', type=int, help='add tpavi block in which stages: [0, 1, 2, 3]')
    parser.add_argument("--tpavi_vv_flag", action='store_true', default=False, help='visual-visual self-attention')
    parser.add_argument("--tpavi_va_flag", action='store_true', default= True, help='visual-audio cross-attention')

    parser.add_argument("--weights", type=str, default='', help='path of trained model')
    parser.add_argument('--log_dir', default='./train_logs', type=str)

    args = parser.parse_args()
#接下来会根据设置你要的视觉特征提取backbone，语音的默认使用vggish特征提取
    if (args.visual_backbone).lower() == "resnet":
        from model import ResNet_AVSModel as AVSModel
        print('==> Use ResNet50 as the visual backbone...')
    elif (args.visual_backbone).lower() == "pvt":
        from model import PVT_AVSModel as AVSModel
        print('==> Use pvt-v2 as the visual backbone...')
    else:
        raise NotImplementedError("only support the resnet50 and pvt-v2")


    # 固定种子
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # 日志目录
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    # 原木
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    args.log_dir = log_dir

    # 保存脚本
    script_path = os.path.join(log_dir, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path, exist_ok=True)

    scripts_to_save = ['train.sh', 'train.py', 'test.sh', 'test.py', 'config.py', 'dataloader.py', './model/ResNet_AVSModel.py', './model/PVT_AVSModel.py', 'loss.py']
    for script in scripts_to_save:
        dst_path = os.path.join(script_path, script)
        try:
            shutil.copy(script, dst_path)
        except IOError:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(script, dst_path)

    # 检查点目录
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # 设置记录器
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.session_name))

    # 模型
    model = AVSModel.Pred_endecoder(channel=256, \
                                        config=cfg, \
                                        tpavi_stages=args.tpavi_stages, \
                                        tpavi_vv_flag=args.tpavi_vv_flag, \
                                        tpavi_va_flag=args.tpavi_va_flag)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    logger.info("==> Total params: %.2fM" % ( sum(p.numel() for p in model.parameters()) / 1e6))

    # 加载预训练的 S4 模型
    if args.load_s4_params: # fine-tune single sound source segmentation model
        model_dict = model.state_dict()
        s4_state_dicts = torch.load(args.trained_s4_model_path)
        state_dict = {'module.' + k : v for k, v in s4_state_dicts.items() if 'module.' + k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        logger.info("==> Reload pretrained S4 model from %s"%(args.trained_s4_model_path))

    # 视频骨干网
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device)
    audio_backbone.cuda()
    audio_backbone.eval()

    # 数据
    train_dataset = MS3Dataset('train')

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.train_batch_size,
                                                        shuffle=True,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)
    max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

    val_dataset = MS3Dataset('val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                        batch_size=args.val_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)

    # 优化
    model_params = model.parameters()
    optimizer = torch.optim.Adam(model_params, args.lr)
    avg_meter_total_loss = pyutils.AverageMeter('total_loss')
    avg_meter_sa_loss = pyutils.AverageMeter('sa_loss')
    avg_meter_iou_loss = pyutils.AverageMeter('iou_loss')

    avg_meter_miou = pyutils.AverageMeter('miou')

    # Train先load图像5帧view合并在一起，再获取语音特征，送入模型。然后计算损失和Iou得分。
    # 输入模型的数据分为两部分，图像帧【bs*5, 3, 224, 224】，乘以5意思是每个视频有5帧，第二部分是语音帧，维度相似
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    for epoch in range(args.max_epoches):
        for n_iter, batch_data in enumerate(train_dataloader):
            imgs, audio, mask, _ = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5 or 1, 1, 224, 224]

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W)
            mask_num = 5
            mask = mask.view(B*mask_num, 1, H, W)
            audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4]) # [B*T, 1, 96, 64]
            with torch.no_grad():
                audio_feature = audio_backbone(audio) # [B*T, 128]

            output, v_map_list, a_fea_list = model(imgs, audio_feature) # [bs*5, 1, 224, 224]
            loss, loss_dict = IouSemanticAwareLoss(output, mask, a_fea_list, v_map_list, \
                                        sa_loss_flag=args.masked_av_flag, lambda_1=args.lambda_1, count_stages=args.masked_av_stages, \
                                        mask_pooling_type=args.mask_pooling_type, threshold=args.threshold_flag, norm_fea=args.norm_fea_flag, \
                                        closer_flag=args.closer_flag, euclidean_flag=args.euclidean_flag, kl_flag=args.kl_flag)

            avg_meter_total_loss.add({'total_loss': loss.item()})
            avg_meter_iou_loss.add({'iou_loss': loss_dict['iou_loss']})
            avg_meter_sa_loss.add({'sa_loss': loss_dict['sa_loss']})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if (global_step-1) % 20 == 0:
                train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, sa_loss:%.4f, lr: %.4f'%(
                            global_step-1, max_step, avg_meter_total_loss.pop('total_loss'), avg_meter_iou_loss.pop('iou_loss'), avg_meter_sa_loss.pop('sa_loss'), optimizer.param_groups[0]['lr'])
                # train_log = ['Iter:%5d/%5d' % (global_step - 1, max_step),
                #         'Total_Loss:%.4f' % (avg_meter_total_loss.pop('total_loss')),
                #         'iou_loss:%.4f' % (avg_meter_L1.pop('iou_loss')),
                #         'sa_loss:%.4f' % (avg_meter_L4.pop('sa_loss')),
                #         'lambda_1:%.4f' % (args.lambda_1),
                #         'lr: %.4f' % (optimizer.param_groups[0]['lr'])]
                # print(train_log, flush=True)
                logger.info(train_log)

        # Validation:
        model.eval()
        with torch.no_grad():
            for n_iter, batch_data in enumerate(val_dataloader):
                imgs, audio, mask, _ = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]

                imgs = imgs.cuda()
                audio = audio.cuda()
                mask = mask.cuda()
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B*frame, C, H, W)
                mask = mask.view(B*frame, H, W)
                audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
                with torch.no_grad():
                    audio_feature = audio_backbone(audio)

                output, _, _ = model(imgs, audio_feature) # [bs*5, 1, 224, 224]

                miou = mask_iou(output.squeeze(1), mask)
                avg_meter_miou.add({'miou': miou})

            miou = (avg_meter_miou.pop('miou'))
            if miou > max_miou:
                model_save_path = os.path.join(checkpoint_dir, '%s_best.pth'%(args.session_name))
                torch.save(model.module.state_dict(), model_save_path)
                best_epoch = epoch
                logger.info('save best model to %s'%model_save_path)

            miou_list.append(miou)
            max_miou = max(miou_list)

            val_log = 'Epoch: {}, Miou: {}, maxMiou: {}'.format(epoch, miou, max_miou)
            # print(val_log)
            logger.info(val_log)

        model.train()
    logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))









