#!/bin/bash
# @author: xhp
# @description: 创建home下的各种目录，把office31和tiny-imagenet-200数据集从压缩包中解压出来，放到对应的位置；注意请在myRDA/prepare_word文件夹下启动这个sh文件


# 创建home下的目录
echo -------------make home dir START-------------
dir_arrs=(/home /home/ubuntu /home/ubuntu/nas \
    /home/ubuntu/nas/datasets /home/ubuntu/nas/projects \
    /home/ubuntu/nas/datasets/imagenet /home/ubuntu/nas/datasets/office)

for dir in ${dir_arrs[@]}
    do
        if [ ! -d $dir  ];then
            echo mkdir $dir
            mkdir $dir
        else
            echo $dir exist    
        fi
    done
echo -------------make home dir END-------------


# 准备office31数据集
echo -------------prepare office31 dataset START-------------
office31_dir=/home/ubuntu/nas/datasets/office/office-31
if [ ! -d $office31_dir  ];then
    echo copy office31 dataset
    cp Original_images.zip /home/ubuntu/nas/datasets/office/Original_images.zip
    unzip /home/ubuntu/nas/datasets/office/Original_images.zip -d /home/ubuntu/nas/datasets/office/
    mv /home/ubuntu/nas/datasets/office/Original_images /home/ubuntu/nas/datasets/office/office-31
else
    echo office31 dataset exist    
fi
echo -------------prepare office31 dataset END-------------


# 准备tiny-imagenet-200数据集
echo -------------prepare tiny-imagenet-200 dataset START-------------
tiny_imagenet_200_dir=/home/ubuntu/nas/datasets/imagenet/tiny-imagenet-200
if [ ! -d $tiny_imagenet_200_dir  ];then
    echo copy tiny-imagenet-200 dataset
    cp tiny-imagenet-200.zip /home/ubuntu/nas/datasets/imagenet/tiny-imagenet-200.zip
    unzip /home/ubuntu/nas/datasets/imagenet/tiny-imagenet-200.zip -d /home/ubuntu/nas/datasets/imagenet/
else
    echo tiny-imagenet-200 dataset exist    
fi
echo -------------prepare tiny-imagenet-200 dataset END-------------


# 把RDA项目复制到/home/ubuntu/nas/projects/RDA
echo -------------prepare to copy RDA project-------------
cp -r ../RDA /home/ubuntu/nas/projects/RDA
echo -------------all preparation stages over-------------