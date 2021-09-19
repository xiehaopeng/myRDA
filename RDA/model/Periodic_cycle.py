import torch.nn as nn
import model.backbone as backbone
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
import tqdm
import shutil
import math
from preprocess.data_provider_cyc import load_images


#==============eval
def evaluate(model_instance, input_loader, loss_matrix, epoch):
    ori_stage = model_instance.c_net.stage
    model_instance.c_net.train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        _, _, probabilities = model_instance.predict(inputs)

        probabilities = probabilities.data.float()
        labels = labels.data.float()
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)

    index = torch.LongTensor(np.array(range(all_labels.size(0)))).cuda()
    a_labels = all_labels
    pred = all_probs[index, a_labels.long()]
    loss = - torch.log(pred)
    loss = loss.data.cpu().numpy()
    loss_matrix[:,epoch]=loss       # 把本轮的loss加入到loss矩阵中
    _, predict = torch.max(all_probs, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels).float() / float(all_labels.size()[0])
    model_instance.set_stage(ori_stage)
    return accuracy, loss_matrix


# 学习率变化器
class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, group_ratios, optimizer, num_iter):
        lr = self.init_lr * (1 + self.gamma * num_iter) ** (-self.decay_rate)
        i=0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * group_ratios[i]
            i+=1
        return optimizer


class PCycleNet(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=31):
        super(PCycleNet, self).__init__()
        # PCycle模型当前所属阶段
        self.stage = "stage_a"
        # 特征提取器 + 瓶颈层
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        # source分类器
        self.source_classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.source_classifier_layer = nn.Sequential(*self.source_classifier_layer_list)
        # target分类器
        self.target_classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.target_classifier_layer = nn.Sequential(*self.target_classifier_layer_list)
        self.softmax = nn.Softmax(dim=1)

        # 初始化网络参数
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.source_classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.source_classifier_layer[dep * 3].bias.data.fill_(0.0)
            self.target_classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.target_classifier_layer[dep * 3].bias.data.fill_(0.0)


        # params_lr_map
        self.params_lr_map = {"base_network": 0.1,
                                "bottleneck_layer": 1,
                                "source_classifier_layer": 1,
                                "target_classifier_layer": 1}
    
    # 数据向前传播过程
    def forward(self, inputs):
        # 特征提取器
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        # 分类器
        if self.stage in ["stage_a","stage_b","stage_c"]:
            outputs = self.source_classifier_layer(features)
        elif self.stage in ["stage_d","stage_e","stage_f"]:
            outputs = self.target_classifier_layer(features)
        elif self.stage == "stage_g":
            outputs = features
        softmax_outputs = self.softmax(outputs)
        return features, outputs, softmax_outputs

    # 设置当前阶段
    def set_stage(self, stage):
        self.stage = stage

    # 初始化网络参数
    def init_params(self):
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.source_classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.source_classifier_layer[dep * 3].bias.data.fill_(0.0)
            self.target_classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.target_classifier_layer[dep * 3].bias.data.fill_(0.0)

class PCycleModel(object):
    def __init__(self, base_net='ResNet50', width=1024, class_num=31, use_bottleneck=True, use_gpu=True, srcweight=3):
        self.c_net = PCycleNet(base_net, use_bottleneck, width, width, class_num)
        self.use_gpu = use_gpu
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight

    def predict(self, inputs):
        features, logits, softmax_outputs = self.c_net(inputs)
        return features, logits, softmax_outputs        # 特征,分类器结果,softmax结果

    # 按照不同阶段返回不同参数
    def get_parameter_list(self):
        parameter_name_list = []
        cur_stage_parameter_list = []
        for name, p in self.c_net.named_parameters():
            if self.c_net.stage in ["stage_a", "stage_b"]:                   # a,b阶段目标域分类器不参与
                if name.startswith('target_classifier_layer'):
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                    parameter_name_list.append(name)
                    cur_stage_parameter_list.append({"params": p,"lr": self.c_net.params_lr_map[name.split('.')[0]]})
            elif self.c_net.stage in ["stage_d", "stage_e"]:                # d,e阶段只更新目标域分类器
                if name.startswith('target_classifier_layer'):              
                    p.requires_grad = True
                    parameter_name_list.append(name)
                    cur_stage_parameter_list.append({"params": p,"lr": self.c_net.params_lr_map[name.split('.')[0]]})
                else:
                    p.requires_grad = False
            elif self.c_net.stage == "stage_f":                             # f阶段只更新特征提取器
                if name.startswith('base_network') or name.startswith('bottleneck_layer'):      
                    p.requires_grad = True
                    parameter_name_list.append(name)
                    cur_stage_parameter_list.append({"params": p,"lr": self.c_net.params_lr_map[name.split('.')[0]]})
                else:
                    p.requires_grad = False

        return parameter_name_list, cur_stage_parameter_list     ## TODO ## lr送出去之后，本阶段训练结束，需要更新c_net中的学习率map

    # 设置阶段，按照阶段设置 是否开启 训练模式
    def set_stage(self, stage):
        self.c_net.set_stage(stage)
        if stage in ["stage_a", "stage_b", "stage_d", "stage_e", "stage_f"]:
            self.c_net.train(True)
        elif stage in ["stage_c", "stage_g"]:
            self.c_net.train(False)


    ## (a) 源域样本预习：筛选本阶段学习样本
    def source_sample_selection(self,cur_stage_source_file, rest_stage_source_file, max_epoch, noisy_rate, noisy_type, cur_cyc_iter_num, cfg):
        ## 判断是否需要进行样本筛选
        # 计算样本噪声率
        nr = noisy_rate
        if cur_cyc_iter_num == 0:
            save_clean_file = rest_stage_source_file.split('.t')[0] + '_source_true_pred_' + str(cur_cyc_iter_num+1) + '.txt'
            save_noisy_file = rest_stage_source_file.split('.t')[0] + '_source_false_pred_' + str(cur_cyc_iter_num+1) + '.txt'
            if noisy_type == 'ood_uniform':
                nr = 1 - (1-nr/2)*(1-nr/2)              # 真实噪声比
            elif noisy_type == 'feature_uniform':
                nr = nr/2
            elif noisy_type == 'ood_feature':
                nr = nr/2
            elif noisy_type == 'ood_feature_uniform':
                nr = nr/3*2
                nr = 1 - (1-nr/2)*(1-nr/2)
        else:
            save_clean_file = rest_stage_source_file.split('_false')[0] + '_true_pred_' + str(cur_cyc_iter_num+1) + '.txt'
            save_noisy_file = rest_stage_source_file.split('_false')[0] + '_false_pred_' + str(cur_cyc_iter_num+1) + '.txt'
        # 计算样本筛选率
        # nr = 0.2, cr = 0.72
        # nr = 0.4, cr = 0.52
        # nr = 0.6, cr = 0.28
        # nr = 0.8, cr = 0.04  ；nr最高只能设置0.8，不然会把所有样本都淘汰
        cr = min(1-1.2*nr, 0.9*(1-nr))      # 每个类挑选的干净样本比例clean rate
        assert cr >= 0, '样本挑选率为0!'
        # 计算样本数量
        # 检查待筛选样本 是否为干净样本(is_clean) 干净标签(clean_labels) 带噪标签(noise_labels) 图片地址(imgs)
        is_clean, clean_labels, noise_labels, imgs = [], [], [], []
        with open(rest_stage_source_file, 'r') as f:
            images = f.readlines()
            for index, i in enumerate(images):
                i =  i.split()  # 空格分隔
                img = i[0]
                imgs.append(img)
                noisy_label = i[1]
                clean_label = i[2]
                noise_labels.append(int(noisy_label))
                clean_labels.append(int(clean_label))
                if noisy_label == clean_label:
                    is_clean.append(1)
                else:
                    is_clean.append(0)
        # 样本噪声过多，当前待筛选样本的noisy_rate大于0.8，则不筛选；
        # 样本数量不够，当前样本数量小于100张，则不筛选
        if nr > 0.8 or len(is_clean) < 100 or int(len(is_clean)/self.class_num*cr) == 0:
            print('')
            print('============================================================(a)阶段训练结果============================================================')
            print('当前Cyc迭代次数 cur_cyc_iter_num {}'.format(cur_cyc_iter_num))
            print('当前待筛选样本数量太少噪声比太高，无法筛选样本! vs nr {} vs sample_num {} '.format(nr,len(is_clean)))
            print('============================================================(a)阶段训练结果============================================================')
            shutil.copy(cur_stage_source_file, save_clean_file)
            shutil.copy(rest_stage_source_file, save_noisy_file)
            return save_clean_file, save_noisy_file, nr

        ## TODO ## 保存网络参数，本阶段结束之后复原网络参数

        ## 设置优化器和变化学习率
        parameter_name_list, param_groups = self.get_parameter_list()
        group_ratios = [group['lr'] for group in param_groups]
        assert cfg.optim.type == 'sgd', 'Optimizer type not supported!'
        optimizer = torch.optim.SGD(param_groups, **cfg.optim.params)
        assert cfg.lr_scheduler.type == 'inv', 'Scheduler type not supported!'
        lr_scheduler = INVScheduler(gamma=cfg.lr_scheduler.gamma,
                                    decay_rate=cfg.lr_scheduler.decay_rate,
                                    init_lr=cfg.init_lr)
        ## 数据集加载器
        train_source_loader = load_images(rest_stage_source_file, batch_size=32)
        test_source_loader = load_images(rest_stage_source_file, batch_size=32, is_train=False, is_eval=False)  # 测试(不属于严格意义的验证，因为用的还是含噪标签)

        ## 预训练
        epoch, iter_num = 0,0
        sample_num = len(test_source_loader.dataset)
        loss_matrix = np.zeros((sample_num, max_epoch))
        while True:
            for datas in tqdm.tqdm(train_source_loader, total=len(train_source_loader), desc='Train epoch = {}'.format(epoch), ncols=0, leave=False):
                inputs_source, labels_source, _ = datas

                optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, iter_num/5)
                optimizer.zero_grad()

                if self.use_gpu:
                    inputs_source, labels_source = Variable(inputs_source).cuda(), Variable(labels_source).cuda()
                else:
                    inputs_source, labels_source = Variable(inputs_source), Variable(labels_source)

                # 求损失并更新网络
                inputs = inputs_source
                class_criterion = nn.CrossEntropyLoss()     # 用nn.CrossEntropyLoss就默认加上了softmax
                _, outputs, _, = self.c_net(inputs)
                total_loss = class_criterion(outputs, labels_source)
                total_loss.backward()
                optimizer.step()

                iter_num += 1

            # 测试预训练模型在含噪源域样本的准确率，并计算loss_matrix，用于样本筛选(不属于严格意义的验证，因为用的还是含噪标签)
            eval_result, loss_matrix = evaluate(self, test_source_loader, loss_matrix, epoch)
            if epoch % 10 == 0:
                print('source domain accuracy:', eval_result)       # 含噪的训练样本分类结果准确率
            epoch += 1

            if epoch >= max_epoch:
                break
        # np.save(args.stats_file, loss_matrix)   ## TODO ## 保存loss_matrix

        ## 筛选样本
        loss_sele = loss_matrix[:, :epoch]
        loss_mean = loss_sele.mean(axis=1)
        sort_index = np.argsort(loss_mean)

        #sort samples per class
        clean_index, noisy_index = [], []            # 存放所有预测干净/噪声样本的位置下标
        for i in range(int(self.class_num)):
            c = []
            for idx in sort_index:
                if noise_labels[idx] == i:
                    c.append(idx)
            clean_num = int(len(c)*cr)
            clean_idx = c[:clean_num]
            clean_index.extend(clean_idx)
            noisy_idx = c[clean_num:]
            noisy_index.extend(noisy_idx)

        acc_clean_num, acc_noisy_num = 0, 0
        for i in clean_index:
            if is_clean[i] == 1:
                acc_clean_num += 1
        for i in noisy_index:
            if is_clean[i] == 0:
                acc_noisy_num += 1
        acc_clean = acc_clean_num/len(clean_index)                      # 挑干净样本的正确率
        acc_noisy = acc_noisy_num/len(noisy_index)                      # 挑噪声样本的正确率
        # amazon: 2817; dslr: 498; webcam: 795; 
        print('')
        print('============================================================(a)阶段训练结果============================================================')
        print('当前Cyc迭代次数 cur_cyc_iter_num {}'.format(cur_cyc_iter_num))
        print("训练总轮数 Epoch {} vs 挑干净样本的正确率 Acc_clean {} vs 挑噪声样本的正确率 Acc_noisy {} vs 样本筛选率 cr {} vs 样本噪声比 nr {}".format(epoch,round(acc_clean,4),round(acc_noisy,4),round(cr,4),round(nr,4)))
        print("样本总数   {} vs 真实干净样本数    {} vs 真实噪声样本数    {}".format( len(is_clean), acc_clean_num+len(noisy_index)-acc_noisy_num, len(clean_index)-acc_clean_num+acc_noisy_num ))
        print("预测干净样本 {}: {}/{}".format(len(clean_index), acc_clean_num, len(clean_index)-acc_clean_num ))
        print("预测噪声样本 {}: {}/{}".format(len(noisy_index), len(noisy_index)-acc_noisy_num, acc_noisy_num ))
        print("数据利用率   {}".format(round(acc_clean_num/(acc_clean_num+len(noisy_index)-acc_noisy_num),4)))
        print('============================================================(a)阶段训练结果============================================================')
        
        ## 生成新的样本txt文件
        # 其中干净数据需要在之前数据的基础上进行追加
        if cur_cyc_iter_num != 0:
            shutil.copy(cur_stage_source_file, save_clean_file)
        with open(save_clean_file,'a') as f:
            with open(save_noisy_file, 'w') as ff:
                for idx, img in enumerate(imgs):
                    if idx in clean_index:
                        f.write('{} {} {}\n'.format(img, noise_labels[idx], clean_labels[idx]))
                    else:
                        ff.write('{} {} {}\n'.format(img, noise_labels[idx], clean_labels[idx]))

        cur_rest_source_noisy_rate = round(math.ceil(acc_noisy * 10)/10,1)          # 计算筛选后脏样本中的噪声比
        return save_clean_file, save_noisy_file, cur_rest_source_noisy_rate         # 返回预测的干净样本作为本阶段学习样本，脏样本作为之后待学习样本

    ## (b) 源域样本复习：训练特征提取器与分类器
    def source_train(self, cur_stage_source_file, target_file, max_epoch, cur_cyc_iter_num, cfg):
        ## 设置优化器和变化学习率
        parameter_name_list, param_groups = self.get_parameter_list()
        group_ratios = [group['lr'] for group in param_groups]
        assert cfg.optim.type == 'sgd', 'Optimizer type not supported!'
        optimizer = torch.optim.SGD(param_groups, **cfg.optim.params)
        assert cfg.lr_scheduler.type == 'inv', 'Scheduler type not supported!'
        lr_scheduler = INVScheduler(gamma=cfg.lr_scheduler.gamma,
                                    decay_rate=cfg.lr_scheduler.decay_rate,
                                    init_lr=cfg.init_lr)
        ## 数据集加载器
        train_source_loader = load_images(cur_stage_source_file, batch_size=32)
        test_source_loader = load_images(target_file, batch_size=32, is_train=False, is_eval=True)          # 验证

        ## 训练
        epoch, iter_num = 0,0
        sample_num = len(test_source_loader.dataset)
        loss_matrix = np.zeros((sample_num, max_epoch))
        while True:
            for datas in tqdm.tqdm(train_source_loader, total=len(train_source_loader), desc='Train epoch = {}'.format(epoch), ncols=0, leave=False):
                inputs_source, labels_source, _ = datas

                optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, iter_num/5)
                optimizer.zero_grad()

                if self.use_gpu:
                    inputs_source, labels_source = Variable(inputs_source).cuda(), Variable(labels_source).cuda()
                else:
                    inputs_source, labels_source = Variable(inputs_source), Variable(labels_source)

                # 求损失并更新网络
                inputs = inputs_source
                class_criterion = nn.CrossEntropyLoss()     # 用nn.CrossEntropyLoss就默认加上了softmax
                _, outputs, _, = self.c_net(inputs)
                total_loss = class_criterion(outputs, labels_source)
                total_loss.backward()
                optimizer.step()

                iter_num += 1

            # 验证
            eval_result, loss_matrix = evaluate(self, test_source_loader, loss_matrix, epoch)
            if epoch % 10 == 0:
                print('source domain accuracy:', eval_result)       # 含噪的训练样本分类结果准确率
            epoch += 1

            if epoch >= max_epoch:
                break
        # np.save(args.stats_file, loss_matrix)   ## TODO ## 保存loss_matrix

        ## 筛选样本
        loss_sele = loss_matrix[:, :epoch]
        loss_mean = loss_sele.mean(axis=1)
        sort_index = np.argsort(loss_mean)

        #sort samples per class
        clean_index, noisy_index = [], []            # 存放所有预测干净/噪声样本的位置下标
        for i in range(int(self.class_num)):
            c = []
            for idx in sort_index:
                if noise_labels[idx] == i:
                    c.append(idx)
            clean_num = int(len(c)*cr)
            clean_idx = c[:clean_num]
            clean_index.extend(clean_idx)
            noisy_idx = c[clean_num:]
            noisy_index.extend(noisy_idx)

        acc_clean_num, acc_noisy_num = 0, 0
        for i in clean_index:
            if is_clean[i] == 1:
                acc_clean_num += 1
        for i in noisy_index:
            if is_clean[i] == 0:
                acc_noisy_num += 1
        acc_clean = acc_clean_num/len(clean_index)                      # 挑干净样本的正确率
        acc_noisy = acc_noisy_num/len(noisy_index)                      # 挑噪声样本的正确率
        # amazon: 2817; dslr: 498; webcam: 795; 
        print('')
        print('============================================================(a)阶段训练结果============================================================')
        print('当前Cyc迭代次数 cur_cyc_iter_num {}'.format(cur_cyc_iter_num))
        print("训练总轮数 Epoch {} vs 挑干净样本的正确率 Acc_clean {} vs 挑噪声样本的正确率 Acc_noisy {} vs 样本筛选率 cr {} vs 样本噪声比 nr {}".format(epoch,round(acc_clean,4),round(acc_noisy,4),round(cr,4),round(nr,4)))
        print("样本总数   {} vs 真实干净样本数    {} vs 真实噪声样本数    {}".format( len(is_clean), acc_clean_num+len(noisy_index)-acc_noisy_num, len(clean_index)-acc_clean_num+acc_noisy_num ))
        print("预测干净样本 {}: {}/{}".format(len(clean_index), acc_clean_num, len(clean_index)-acc_clean_num ))
        print("预测噪声样本 {}: {}/{}".format(len(noisy_index), len(noisy_index)-acc_noisy_num, acc_noisy_num ))
        print("数据利用率   {}".format(round(acc_clean_num/(acc_clean_num+len(noisy_index)-acc_noisy_num),4)))
        print('============================================================(a)阶段训练结果============================================================')
        
        ## 生成新的样本txt文件
        # 其中干净数据需要在之前数据的基础上进行追加
        if cur_cyc_iter_num != 0:
            shutil.copy(cur_stage_source_file, save_clean_file)
        with open(save_clean_file,'a') as f:
            with open(save_noisy_file, 'w') as ff:
                for idx, img in enumerate(imgs):
                    if idx in clean_index:
                        f.write('{} {} {}\n'.format(img, noise_labels[idx], clean_labels[idx]))
                    else:
                        ff.write('{} {} {}\n'.format(img, noise_labels[idx], clean_labels[idx]))

        cur_rest_source_noisy_rate = round(math.ceil(acc_noisy * 10)/10,1)          # 计算筛选后脏样本中的噪声比
        return save_clean_file, save_noisy_file, cur_rest_source_noisy_rate         # 返回预测的干净样本作为本阶段学习样本，脏样本作为之后待学习样本
