import torch.nn as nn
import model.backbone as backbone
from model.modules.grl import WarmStartGradientReverseLayer
import torch.nn.functional as F
import torch
import numpy as np


class RDANet(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=31):
        super(RDANet, self).__init__()
        ## set base network
        # 特征提取器
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        # 梯度反转层
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000., auto_step=True)
        # 瓶颈层
        self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        # 分类器1
        self.classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        # 分类器2
        self.classifier_layer_2_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer_2 = nn.Sequential(*self.classifier_layer_2_list)
        self.softmax = nn.Softmax(dim=1)

        ## initialization 初始化权重
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.classifier_layer_2[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer_2[dep * 3].bias.data.fill_(0.0)
            self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[dep * 3].bias.data.fill_(0.0)

        ## collect parameters
        self.parameter_list = [{"params":self.base_network.parameters(), "lr":0.1},
                            {"params":self.bottleneck_layer.parameters(), "lr":1},
                        {"params":self.classifier_layer.parameters(), "lr":1},
                               {"params":self.classifier_layer_2.parameters(), "lr":1}]
    def forward(self, inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        features_adv = self.grl_layer(features)
        outputs_adv = self.classifier_layer_2(features_adv)
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        return features, outputs, softmax_outputs, outputs_adv      # [特征,分类器1输出,softmax分类器1输出,反转层分类器2输出]

# Proxy Margin Discrepancy 代理边缘差异
class PMD(object):
    def __init__(self, base_net='ResNet50', width=1024, class_num=31, use_bottleneck=True, use_gpu=True, srcweight=3):
        self.c_net = RDANet(base_net, use_bottleneck, width, width, class_num)
        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight

    def get_loss(self, inputs, labels_source, max_iter, del_rate=0.4, noisy_source_num=100):
        class_criterion = nn.CrossEntropyLoss()
        #introduce noisy source instances to improve the discrepancy
        # inputs = torch.cat((inputs_source, inputs_target, labels_source_noisy), dim=0)
        source_size, source_noisy_size, target_size = labels_source.size(0), noisy_source_num, \
            inputs.size(0) - labels_source.size(0) - noisy_source_num

        #gradual transition
        lr = linear_rampup(self.iter_num, total_iter=max_iter)

        _, outputs, _, outputs_adv = self.c_net(inputs)         # [特征,分类器1输出,softmax分类器1输出,反转层分类器2输出]

        #compute cross entropy loss on source domain
        #classifier_loss = class_criterion(outputs.narrow(0, 0, labels_source.size(0)), labels_source)
        #get large loss samples index
        outputs_src = outputs.narrow(0, 0, source_size)         # 干净源域输出结果
        classifier_loss, index_src = class_rank_criterion(outputs_src, labels_source, lr, del_rate)

        #compute discrepancy
        target_adv = outputs.max(1)[1]
        target_adv_src = target_adv.narrow(0, 0, source_size)                               # 干净源域分类结果
        target_adv_tgt = target_adv.narrow(0, source_size, target_size)                     # 目标域分类结果
        target_adv_noisy = target_adv.narrow(0, source_size+target_size, source_noisy_size) # 带噪源域分类结果

        outputs_adv_src = outputs_adv.narrow(0, 0, source_size)                             # 反转层干净源域输出结果
        outputs_adv_tgt = outputs_adv.narrow(0, source_size, target_size)                   # 反转层目标域输出结果
        outputs_adv_noisy = outputs_adv.narrow(0, source_size+target_size, source_noisy_size)   # 反转层带噪源域输出结果

        outputs_adv_src = outputs_adv_src[index_src]
        target_adv_src = target_adv_src[index_src] 
        #classifier_loss_adv_src = class_criterion(torch.cat((outputs_adv_src, outputs_adv_noisy),dim=0), \
        #    torch.cat((target_adv_src, target_adv_noisy), dim=0))
        classifier_loss_adv_src = class_criterion(outputs_adv_src, target_adv_src)

        logloss_tgt = torch.log(torch.clamp(1 - F.softmax(outputs_adv_tgt, dim = 1), min=1e-15))
        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

        en_loss = entropy(outputs_adv_tgt) + entropy(outputs_adv_noisy) #+ entropy(outputs_adv_src)
        transfer_loss = self.srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt

        self.iter_num += 1
        #total_loss = classifier_loss + transfer_loss + 0.1*en_loss
        total_loss = classifier_loss + transfer_loss #+ 0.1*en_loss
        #print(classifier_loss.data, transfer_loss.data, en_loss.data)
        return [total_loss, classifier_loss, transfer_loss, classifier_loss_adv_src, classifier_loss_adv_tgt]

    def predict(self, inputs):
        feature, _, softmax_outputs,_= self.c_net(inputs)
        return softmax_outputs, feature

    def get_parameter_list(self):
        return self.c_net.parameter_list

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode

    def get_loss_without_unlabeled_data(self, inputs, labels_source, max_iter, del_rate=0.4):   # input包含source和target
        class_criterion = nn.CrossEntropyLoss()

        #mixup inputs between source and target data
        #TODO Random concat samples into new distribution.
        #source_input = inputs.narrow(0, 0, labels_source.size(0))
        #target_input = inputs.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))

        #gradual transition
        lr = linear_rampup(self.iter_num, total_iter=max_iter)

        _, outputs, _, outputs_adv = self.c_net(inputs)   # [特征,分类器1输出,softmax分类器1输出,反转层分类器2输出]

        #compute cross entropy loss on source domain
        #classifier_loss = class_criterion(outputs.narrow(0, 0, labels_source.size(0)), labels_source)
        #get large loss samples index
        # narrow为切片操作
        outputs_src = outputs.narrow(0, 0, labels_source.size(0))           # 源域输出结果
        classifier_loss, index_src = class_rank_criterion(outputs_src, labels_source, lr, del_rate)     # 源域样本 求 分类loss 和 loss较小的样本下标

        #compute discrepancy
        target_adv = outputs.max(1)[1]                                      # 求每行最大值对应的索引，即判别样本被分类器分为哪个类别
        target_adv_src = target_adv.narrow(0, 0, labels_source.size(0))     # 源域分类结果
        target_adv_tgt = target_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))    # 目标域分类结果
        #target_adv_mix = torch.cat((target_adv_src[index_src], target_adv_tgt[index_tgt]), dim=0)

        outputs_adv_src = outputs_adv.narrow(0, 0, labels_source.size(0))   # 源域反转层输出结果
        outputs_adv_tgt = outputs_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))  # 目标域反转层输出结果
        #outputs_adv_mix = torch.cat((outputs_adv_src[index_src], outputs_adv_tgt[index_tgt]), dim=0)

        #print(target_adv_mix, outputs_adv_mix)

        # 选前loss较小的样本
        outputs_adv_src = outputs_adv_src[index_src]
        target_adv_src = target_adv_src[index_src]
        classifier_loss_adv_src = class_criterion(outputs_adv_src, target_adv_src)  # 源域分类结果 和 源域反转层输出结果 求loss？

        logloss_tgt = torch.log(torch.clamp(1 - F.softmax(outputs_adv_tgt, dim = 1), min=1e-15))
        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)           # 目标域分类结果 和 目标域反转层输出结果 求loss？

        #loss_adv_mix_2 = class_criterion(outputs_adv_mix, target_adv_mix)
        #logloss_mix = torch.log(torch.clamp(1 - F.softmax(outputs_adv_mix, dim = 1), min=1e-15))
        #loss_adv_mix_1 = F.nll_loss(logloss_mix, target_adv_mix)
        #transfer_loss = self.srcweight * classifier_loss_adv_src - loss_adv_mix_2 + self.srcweight * loss_adv_mix_2 + classifier_loss_adv_tgt
        transfer_loss = self.srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt      # 域迁移loss

        self.iter_num += 1
        total_loss = classifier_loss + transfer_loss #+ 0.1*en_loss
        #print(classifier_loss.data, transfer_loss.data, en_loss.data)
        return [total_loss, classifier_loss, transfer_loss, classifier_loss_adv_src, classifier_loss_adv_tgt]

def class_rank_criterion(outputs_source, labels_source, lr, del_rate):      # del_rate为样本筛选概率
    if lr > del_rate:
        lr = del_rate
    remove_num = torch.ceil(torch.tensor(labels_source.size(0)*lr))
    softmax = nn.Softmax(dim=1)
    index = torch.LongTensor(np.array(range(labels_source.size(0)))).cuda()
    outputs_source = softmax(outputs_source)
    pred = outputs_source[index, labels_source]
    loss = - torch.log(pred)
    _, indices = torch.sort(loss, 0)    # 按照列进行排序，indices为排序的索引结果矩阵
    topk = labels_source.size(0)-remove_num.long()
    index_src = indices[:topk]
    loss = torch.mean(loss)
    return loss, index_src      # 返回平均的源域样本分类loss和排名前几位的index

def entropy(output_target):
    """
    entropy minimization loss on target domain data
    """
    softmax = nn.Softmax(dim=1)
    output = output_target
    output = softmax(output)
    en = -torch.sum((output*torch.log(output + 1e-8)), 1)
    return torch.mean(en)

def linear_rampup(now_iter, total_iter=20000):
    if total_iter == 0:
        return 1.0
    else:
        current = np.clip(now_iter / total_iter, 0., 1.0)
        return float(current)
