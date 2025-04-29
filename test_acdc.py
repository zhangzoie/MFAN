#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import test_single_volume
from model.model_mfan import MFAN
import importlib
from torchvision import transforms
from datasets.dataset_acdc import ACDCdataset, RandomGenerator

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=24, help="batch size")
parser.add_argument("--lr", default=0.0001, help="learning rate")
parser.add_argument("--max_epochs", default=100)
parser.add_argument("--img_size", default=224)
parser.add_argument('--config_file', type=str,
                    default='swin_224_7_3level', help='config file name w/o suffix')
parser.add_argument("--save_path", default="/home/JianjianYin/transdeeplab/ACDC/pth")
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default="/home/user/MFAN/ACDC/ACDC/lists_ACDC")
parser.add_argument("--root_dir", default="/home/user/MFAN/ACDC/ACDC")
parser.add_argument("--volume_path", default="/home/user/MFAN/ACDC/ACDC/test")
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=4)
parser.add_argument('--test_save_dir', default=None, help='saving prediction as nii!')
parser.add_argument("--patches_size", default=16)
parser.add_argument("--n_skip", default=1)
args = parser.parse_args()

model_config = importlib.import_module(f'model.configs.{args.config_file}')
model_config.DecoderConfig.num_classes = 4
model = MFAN(
    model_config.EncoderConfig, 
    model_config.ASPPConfig, 
    model_config.DecoderConfig
).cuda()

model.load_state_dict(torch.load("/home/user/MFAN/Results/ACDC/best_4.pth"))

def inference(args, model, testloader, test_save_path=None):
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        number_zero = [] # 计算总体的DSC方差
        number_one = []
        number_two = []
        number_three = []
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            metric_list += np.array(metric_i)
            print('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
            number_zero.append(np.mean(metric_i, axis=0)[0]) # 保存每个样例均值的DSC系数
            
            number_one.append(metric_i[0][0])
            number_two.append(metric_i[1][0])
            number_three.append(metric_i[2][0])
        metric_list = metric_list / len(testloader)
        for i in range(1, args.num_classes):
            print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
            logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]

        over_all_std = np.std(number_zero)
        class_1_std = np.std(number_one)
        class_2_std = np.std(number_two)
        class_3_std = np.std(number_three)
        print('Overall DSC Std: %f, Class 1 Std: %f, Class 2 Std: %f, Class 3 Std: %f' % (over_all_std, class_1_std, class_2_std, class_3_std))
        print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))

        logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
        logging.info("Testing Finished!")
        return performance, mean_hd95

train_dataset = ACDCdataset(args.root_dir, args.list_dir, split="train", transform=
                                   transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
Train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
db_val = ACDCdataset(base_dir=args.root_dir, list_dir=args.list_dir, split="valid")
valloader = DataLoader(db_val, batch_size=1, shuffle=False)
db_test = ACDCdataset(base_dir=args.volume_path,list_dir=args.list_dir, split="test")
testloader = DataLoader(db_test, batch_size=1, shuffle=False)
model = model.cuda()
model.eval()

avg_dcs, avg_hd = inference(args, model, testloader, args.test_save_dir)
