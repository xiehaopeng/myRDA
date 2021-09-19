"""
噪声环境下的阶段性循环域适应
"""

import tqdm
import argparse
import numpy as np
from torch.autograd import Variable
import torch
import sys
sys.path.insert(0, "/home/ubuntu/nas/projects/RDA")
from utils.config import Config
import warnings
warnings.filterwarnings("ignore")


def train(model_instance, source_file, target_file, max_cyc_iter, args):
    cfg = Config(args.config)
    print("start Cycle train...")
    iter_num = 0
    total_progress_bar = tqdm.tqdm(desc='Cycle iter', total=max_cyc_iter)

    # 本阶段学习样本  待学习样本
    cur_stage_source_file, rest_stage_source_file = "", source_file
    cur_stage_target_file, rest_stage_target_file = "", target_file
    cur_rest_source_noisy_rate = args.noisy_rate
    noisy_type = args.noisy_type

    while True:
        ## (a) 源域样本预习：筛选本阶段学习样本
        model_instance.set_stage("stage_a")
        cur_stage_source_file, rest_stage_source_file, cur_rest_source_noisy_rate = model_instance.source_sample_selection(cur_stage_source_file, rest_stage_source_file, max_epoch=30, noisy_rate=cur_rest_source_noisy_rate, noisy_type=noisy_type, cur_cyc_iter_num=iter_num, cfg=cfg)
        ## (b) 源域样本复习：训练特征提取器与分类器
        model_instance.source_train(cur_stage_source_file, target_file, max_epoch=30, cur_cyc_iter_num=iter_num, cfg=cfg)
        ## (c) 目标域伪标签生成
        ## (d) 目标域样本预习
        ## (e) 目标域样本复习
        ## (f) 域迁移：用目标域分类器给源域样本打标签，与真实标签比较，更新特征提取器，拉近源域与目标域距离
        ## (g) 噪声样本修正：用更新完的特征提取器给源域样本做聚类，修正错误标签
        # TODO 计算修正之后源域剩余样本的cur_rest_source_noisy_rate

        iter_num += 1
        total_progress_bar.update(1)

        # 退出cyc条件
        if iter_num >= max_cyc_iter:
            break

    print('finish Cycle train')
    return 


if __name__ == '__main__':
    from model.Periodic_cycle import PCycleModel
    from preprocess.data_provider import load_images
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='all sets of configuration parameters',
                        default='/home/liujintao/app/transfer-lib/config/dann.yml')
    parser.add_argument('--dataset', default='Office-31', type=str,
                        help='which dataset')
    parser.add_argument('--src_address', default=None, type=str,
                        help='address of image list of source dataset')
    parser.add_argument('--tgt_address', default=None, type=str,
                        help='address of image list of target dataset')
    parser.add_argument('--stats_file', default=None, type=str,
                        help='store the training loss and and validation acc')
    parser.add_argument('--noisy_rate', default=0.4, type=float,
                        help='noisy rate')
    parser.add_argument('--noisy_type', default='feature_uniform', type=str,
                        help='noisy rate')

    args = parser.parse_args()
    print(args)
    
    source_file = args.src_address
    target_file = args.tgt_address

    if args.dataset == 'Office-31':
        class_num = 31
        width = 1024
        srcweight = 4
        is_cen = False
    elif args.dataset == 'Office-home':
        class_num = 65
        width = 2048
        srcweight = 2
        is_cen = False
    elif args.dataset == 'Bing-Caltech':
        class_num = 257
        width = 2048
        srcweight = 2
        is_cen = False
    elif args.dataset == 'COVID-19':
        class_num = 3
        width = 256
        srcweight = 4
        is_cen = False
        # Another choice for Office-home:
        # width = 1024
        # srcweight = 3
        # is_cen = True
    else:
        width = -1

    model_instance = PCycleModel(base_net='ResNet50', width=width, use_gpu=True, class_num=class_num, srcweight=srcweight)

    # 训练模型，其中max_cyc_iter表示cyc迭代次数
    train(model_instance, source_file, target_file, max_cyc_iter=10, args=args)

    # to_dump = train(model_instance, source_file, target_file, max_cyc_iter=20, args=args)

    # # 把训练阶段的损失loss和阶段性验证结果result保存到args.stats_file中
    # pickle.dump(to_dump, open(args.stats_file, 'wb'))
