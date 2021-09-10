# 基于RDA的 pp 域适应项目代码
首先需要下载数据集，放在prepare_work文件夹下

[Office-31](https://drive.google.com/file/d/1i9e23b5T5yTZ-FFuKd37_dTl-Dg64w4I/view?usp=sharing) ; [tiny-imagenet-200](https://drive.google.com/file/d/1LMAJwvQ1Ojn1NP6ymFJphZnHnYSWj-fN/view?usp=sharing) ; 

然后cd到prepare_work文件夹下，运行prepare_dataset.sh文件，完成数据准备工作。

## 新标题


## RDA项目代码结构
这个部分介绍了项目中各个文件和文件夹的作用

- myRDA
- RDA
  - config: yml文件里设置了统一的学习率和优化器的参数
  - data: 存放了各类数据集的图片地址和对应标签
    - Office-31: Office-31数据集下混合了各种类型噪声，每个txt文件表示一种噪声添加方法
        - amazon_ood_feature_noisy_0.4.txt
        - ...
    - ...
    - modify_directory.py
    - noisify.py: 生成噪声数据集
    - tinyimagenet.txt: tinyimagenet数据集的图片地址
  - model: 存放各种网络模型
    - backbone.py: 定义了各种基本的骨干网络，比如AlexNet,ResNet50等
    - Resnet.py: 默认用ResNet50做的一个Resnet网络
    - ...
  - preprocess
  - scripts
  - statistic
    - xxx.pkl: 保存各种训练方式下的训练loss和验证准确率result
    - xxx.pkl.pth: 保存各种训练方式下的网络参数
  - log: 存放训练的日志
  - trainer
  - utils
- prepare_word