import os
import numpy as np
from numpy.testing import assert_array_almost_equal
from imgaug import augmenters as iaa
from PIL import Image
import random
import tqdm
import glob


def listing_file(domains, data_dir, save_dir):
    ''' 创建数据集图片的地址和对应数字标签的txt文件
    '''
    for d in domains:
        file_dir = data_dir + d + '/*/*'
        save_file = save_dir + d + '.txt'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        img_files = glob.glob(file_dir, recursive=True)     # 根据地址找到对应的所有文件；recursive=True表示递归调用，与通配符*一起使用，获取目标文件夹下的所有文件
        num_labels = []                                     # 存放所有图片的数字标签，num_labels[i] 就是 img_files[i] 图片的标签
        str_num_label_map = {}                              # 表示字符串标签和数字标签的关系
        label_num = 0                                       # label_num是目前图片种类数量
        for img in img_files:
            str_label = img.split('/')[8]                   # str_label是当前的图片的字符串标签
            print(str_label)
            if not str_num_label_map.has_key(str_label): 
                str_num_label_map[str_label] = label_num
                label_num += 1
            cur_num_label = str_num_label_map[str_label]
            num_labels.append(cur_num_label)
        with open(save_file,'w') as f:
            for i, img in enumerate(img_files):
                f.write('{} {}\n'.format(img, num_labels[i]))   # 把每个图片地址对应的图片数字标签写入txt文件中，用空格隔开

# def listing_file(domains, data_dir, save_dir):
#     ''' 原始版本，对于不同域乱序的图片来说逻辑有错误
#     '''
#     for d in domains:
#         str_labels, num_labels = [], []         # 分别存储所有种类的字符串标签和数字标签
#         file_dir = data_dir + d + '/*/*'
#         save_file = save_dir + d + '.txt'
#         if not os.path.exists(save_dir):
#             os.mkdir(save_dir)
#         img_files = glob.glob(file_dir, recursive=True)     # 根据地址找到对应的文件，recursive=True表示递归调用，与*一起使用，获取目标文件夹下的所有文件
#         label, first = 0, True                  # label是当前图片种类的数字标签(对于office31来说就是0～30)
#         for img in img_files:
#             str_label = img.split('/')[8]       # str_label是当前的图片种类（字符串标签）
#             print(str_label)
#             if first:
#                 num_labels.append(label)
#                 str_labels.append(str_label)
#                 first = False
#             else:
#                 if str_label not in str_labels: # 如果当前的标签是新的标签，则数字标签+1
#                     label += 1
#                     str_labels.append(str_label)
#                     num_labels.append(label)
#                 else:
#                     num_labels.append(label)
#         with open(save_file,'w') as f:
#             for i, img in enumerate(img_files):
#                 f.write('{} {}\n'.format(img, num_labels[i]))


def prepare_imagenet():
    ''' 创建tinyimagenet所有训练图片的地址txt文件
    '''
    dataset_dir = '/home/ubuntu/nas/datasets/imagenet/tiny-imagenet-200/*/*/images/*'
    save_filename = 'tinyimagenet.txt'

    img_files = glob.glob(dataset_dir)
    with open(save_filename,'w') as f:
            for i, img in enumerate(img_files):
                f.write('{}\n'.format(img))


def multiclass_noisify(y, T, random_state=0):
    """
    Flip classes according to transition probability matrix T.
    根据变换可能性矩阵T，来改变原始类别标签y
    """
    y = np.asarray(y)
    #print(np.max(y), T.shape[0])
    assert T.shape[0] == T.shape[1]
    assert np.max(y) < T.shape[0]
    assert_array_almost_equal(T.sum(axis=1), np.ones(T.shape[1]))
    assert (T >= 0.0).all()
    m = y.shape[0]
    #print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)
    for idx in np.arange(m):
        i = y[idx]
        #print(i, T[i,:])
        flipped = flipper.multinomial(1, T[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]
    return new_y


def symmetric_noisy(y, noisy_rate, random_state=None, class_number=31):
    """
    flip y in the symmetric way
    对称 标签噪声：每个类别都有noisy_rate的可能性成为错误类别，并且各种错误类别的概率相同
    比如当前noisy_rate为0.2，当前样本类别为n的样本，有0.8的概率保持n标签，0.2/(n-1)概率为其他标签
    """
    r = noisy_rate
    T = np.ones((class_number, class_number))
    T = (r / (class_number - 1)) * T
    if r > 0.0:
        T[0, 0] = 1. - r
        for i in range(1, class_number-1):
            T[i, i] = 1. - r
        T[class_number-1, class_number-1] = 1. - r
        y_noisy = multiclass_noisify(y, T=T, random_state=random_state)
        actural_noise_rate = (y != y_noisy).mean()
        assert actural_noise_rate > 0.0
        print('Actural noisy rate is {}'.format(actural_noise_rate))
    #print(T)
    return y_noisy, actural_noise_rate


def asymmetric_noisy(y, noisy_rate, random_state=None, class_number=31):
    """
    flip in the pair
    不对称 标签噪声： 每个类别都有noisy_rate的可能性成为错误类别，错误类别固定为其数字标签的下一个数
    比如当前noisy_rate为0.2，当前样本类别为n的样本，有0.8的概率保持n标签，0.2概率为n+1标签
    """
    T = np.eye(class_number)
    r = noisy_rate
    if r > 0.0:
        T[0,0], T[0,1] = 1. - r, r
        for i in range(1, class_number-1):
            T[i,i], T[i,i+1] = 1. - r, r
        T[class_number-1, class_number-1], T[class_number-1,0] = 1. - r, r

        y_noisy = multiclass_noisify(y, T=T, random_state=random_state)
        actural_noise_rate = (y != y_noisy).mean()
        assert actural_noise_rate > 0.0
        print('Actural noisy rate is {}'.format(actural_noise_rate))
    #print(T)
    return y_noisy, actural_noise_rate


def pil_loader(path):
    ''' 由path拿到RGB通道图
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# 
def sp_blur_noise(image):
    '''
    Add salt and pepper noise to image and gaussian bluring image.
    为图片添加椒盐噪声和高斯模糊
    '''
    image = np.asarray(image)
    sp_blur = iaa.Sequential([iaa.GaussianBlur(sigma=8.00),
        iaa.CoarseSaltAndPepper(p=0.5, size_percent=0.04)])
    output = sp_blur.augment_image(image)
    output = Image.fromarray(output)
    return output


def corrupt_image(file_dir):
    for path in tqdm.tqdm(file_dir):
        save_path = path.split('.')[0] + '_corrupted.jpg'
        if not os.path.isfile(save_path):
            image = pil_loader(path)
            image = sp_blur_noise(image)
            image.save(save_path)
    print('complete corrupting images!')


if __name__ == '__main__':

    dataset = 'office-31' # or office-home

    if dataset is 'office-home':
        class_number = 65
        data_files = ['Office-home/Art.txt', 'Office-home/Clipart.txt', 'Office-home/Product.txt', 'Office-home/Real_world.txt']
        domains = ['Art', 'Clipart', 'Product', 'Real_world']
        data_dir = '/home/ubuntu/nas/datasets/office/office-home/'
        save_dir = 'Office-home/'
    elif dataset is 'office-31':
        class_number = 31
        #data_files = ['Office-31/webcam.txt', 'Office-31/dslr.txt', 'Office-31/amazon.txt']
        # 这里的data_files可以手动修改，想要noisy哪个域就填哪个
        data_files = ['Office-31/amazon.txt']           
        domains = ['webcam', 'dslr', 'amazon']
        data_dir = '/home/ubuntu/nas/datasets/office/office-31/'
        save_dir = 'Office-31/'
    elif dataset is 'COVID-19':
        class_number = 3
        #data_files = ['Office-31/webcam.txt', 'Office-31/dslr.txt', 'Office-31/amazon.txt']
        data_files = ['COVID-19/source.txt']#, 'COVID-19/target.txt']
        domains = ['source', 'target']
        data_dir = '/home/ubuntu/nas/datasets/COVID-19/DAXray/'
        save_dir = 'COVID-19/'
    else:
        raise Exception("Sorry, unsupported dataset")

    listing = False             # 是否创建数据集图片的地址txt文件
    corrupt_feature = False     # 是否需要生成添加椒盐噪声和高斯模糊的corrupted图片
    pre_imagenet = False        # 是否需要生成tinyimagenet所有训练图片的地址txt文件

    # listing source files of images into text if not exists.
    if listing:
        listing_file(domains, data_dir, save_dir)
    if pre_imagenet:
        prepare_imagenet()


    for data_file in data_files:    # Read text files
        file_dir, label = [], []    # 存放当前域的所有文件地址和数字标签
        with open(data_file, 'r') as f:
            for i in f.read().splitlines():
                file_dir.append(i.split(' ')[0])
                label.append(int(i.split(' ')[1]))
        if corrupt_feature:
            corrupt_image(file_dir)


        #############################################################
        ##                     Todo: label noisy                   ##
        #############################################################
        """
        noisy_rate = [0.2,0.4,0.6,0.8]
        noisy_type = ['uniform', 'pair']        # label noisy 有pair和uniform两种类型，分别对应了asymmetric_noisy和symmetric_noisy方法
        for tp in noisy_type:
            for rate in noisy_rate:
                if tp is 'pair':
                    label_noisy, acutal_noisy_rate = asymmetric_noisy(label, rate, class_number=class_number)
                elif tp is 'uniform':
                    label_noisy, acutal_noisy_rate = symmetric_noisy(label, rate, class_number=class_number)
                print('generate noisy rate:', acutal_noisy_rate)
                # save all the images in a txt
                save_file = data_file.split('.')[0] + '_{}_noisy_{}.txt'.format(tp, rate)
                with open(save_file,'w') as f:
                    for i, d in enumerate(file_dir):
                        f.write('{} {} {}\n'.format(d, label_noisy[i], label[i]))       # txt文件中分别存放图片地址对应的带噪声标签和原始clear标签

                '''
                split data into noisy and clean data for validating the idea
                把正确标签和错误标签样本分开放在 *_true.txt 和 *_false.txt 中
                '''
                #save_clean_file = data_file.split('.')[0] + '_{}_noisy_{}_true.txt'.format(tp, rate)
                #save_noisy_file = data_file.split('.')[0] + '_{}_noisy_{}_false.txt'.format(tp, rate)
                #with open(save_clean_file,'w') as f:
                #    with open(save_noisy_file, 'w') as ff:
                #        for i, d in enumerate(file_dir):
                #            if label[i] == label_noisy[i]:
                #                f.write('{} {}\n'.format(d, label_noisy[i]))
                #            else:
                #                ff.write('{} {}\n'.format(d, label_noisy[i]))

        print('complete corrupting labels!')
        """
        

        #############################################################
        ##                    Todo: feature noisy                  ##
        #############################################################
        '''
        # 生成feature noisy之前，需要保证所有图片都生成了_corrupted.jpg版本
        # 每张图片有noisy_feature_rate的概率变成含噪图片，噪声是固定参数的椒盐噪声和高斯模糊
        noisy_feature_rate = [0.2]      # [0.1,0.2,0.3,0.4,0.6,0.8]     
        for rate in noisy_feature_rate:
            save_file = data_file.split('.')[0] + '_feature_noisy_{}.txt'.format(rate)
            num = 0                     # 记录添加了feature noisy的噪声图片数量
            with open(save_file, 'w') as f:
                for i, d in enumerate(file_dir):
                    rdn = random.random()
                    if rdn < rate:
                        num+=1
                        d = d.split('.')[0] + '_corrupted.jpg'
                    f.write('{} {}\n'.format(d, label[i]))
            print('feature noise true/given {}/{}'.format(float(num)/len(file_dir), rate))
        print('complete corrupting features!')
        '''

     
        #############################################################
        ##      Todo: introduce out-of-distribution (ood) noise    ##
        #############################################################
        # 往原始数据集中，加入其他数据集图片，并且标签按照原始数据集的比例来设置
        # 比如原始数据集中有3个类别，分别有(10,20,30)个样本，假设当前ood_noisy_rate为0.5
        # 则表示原始数据集样本占总体数据集样本的50%，需要另外加入50%的其他样本
        # 加入ood样本之后，数据集分别为(20,40,60)，其中每个类别都有50%为ood样本
        """
        ood_dataset = "tinyimagenet.txt"    # Introduce other dataset
        with open(ood_dataset, 'r') as f:
            ood_file_dir, ood_label = [], []
            for i in f.read().splitlines():
                ood_file_dir.append(i)
        ood_noisy_rate = [0.1,0.2,0.3,0.4,0.6,0.8]
        for rate in ood_noisy_rate:
            save_file = data_file.split('.')[0] + '_ood_noisy_{}.txt'.format(rate)
            # num_per_class[i]表示原始类别i中需要加入的新域的样本数量
            num_per_class = np.zeros(class_number)
            for i in label:
                num_per_class[i] += 1
            num_per_class = np.ceil(num_per_class*(rate/(1-rate))) 
            random.shuffle(ood_file_dir)
            new_file_dir = ood_file_dir[:int(num_per_class.sum())]
            new_label = []
            for i, j in enumerate(num_per_class):   # i为当前类别，j为新增样本数量
                new_label.extend([i]*int(j))
            
            with open(save_file,'w') as f:
                    for i, d in enumerate(file_dir):
                        f.write('{} {} {}\n'.format(d, label[i], label[i]))             # 原始样本
                    for i, d in enumerate(new_file_dir):
                        f.write('{} {} {}\n'.format(d, new_label[i], class_number+1))   # ood样本，假标签，真标签(所有ood样本为一个类别)
        """


        #############################################################
        ##           Todo mix: feature noise + label noise         ##
        #############################################################
        # 混合feature noise 和 label noise
        # 前提 需要先生成featrue noisy的txt
        """
        noisy_rate = [0.4]
        for rate in noisy_rate:
            rate = rate #for fair comparison with TCL
            feature_noisy_file = data_file.split('.')[0] + '_feature_noisy_{}.txt'.format(rate/2)
            with open(feature_noisy_file, 'r') as f:
                file_dir, label = [], []
                for i in f.read().splitlines():
                    file_dir.append(i.split(' ')[0])
                    label.append(int(i.split(' ')[1]))
            #noisy label
            noisy_type = ['uniform']
            for tp in noisy_type:
                if tp is 'pair':
                    label_noisy, acutal_noisy_rate = asymmetric_noisy(label, rate/2, class_number=class_number)
                elif tp is 'uniform':
                    label_noisy, acutal_noisy_rate = symmetric_noisy(label, rate/2, class_number=class_number)
                print('generate noisy rate:', acutal_noisy_rate)
                #save all the images in a txt
                save_file = data_file.split('.')[0] + '_feature_{}_noisy_{}.txt'.format(tp, rate)
                with open(save_file,'w') as f:
                    for i, d in enumerate(file_dir):
                        f.write('{} {} {}\n'.format(d, label_noisy[i], label[i]))

                # save split noisy and clean data for validate the idea
                # 把正确标签和错误标签样本分开放在 *_true.txt 和 *_false.txt 中
                save_clean_file = data_file.split('.')[0] + '_feature_{}_noisy_{}_true.txt'.format(tp, rate)
                save_noisy_file = data_file.split('.')[0] + '_feature_{}_noisy_{}_false.txt'.format(tp, rate)
                with open(save_clean_file,'w') as f:
                    with open(save_noisy_file, 'w') as ff:
                        for i, d in enumerate(file_dir):
                            if label[i] == label_noisy[i]:
                                f.write('{} {}\n'.format(d, label_noisy[i]))
                            else:
                                ff.write('{} {}\n'.format(d, label_noisy[i]))
        """


        #############################################################
        ##             Todo mix: label noise + OOD noise           ##
        #############################################################
        # 需要先生成 label noise 的txt文件
        '''
        ood_dataset = "tinyimagenet.txt" #Introduce other dataset
        with open(ood_dataset, 'r') as f:
            ood_file_dir, ood_label = [], []
            for i in f.read().splitlines():
                ood_file_dir.append(i)
        ood_noisy_rate = [0.4]
        for rate in ood_noisy_rate:
            label_noise_file = data_file.split('.')[0] + '_uniform_noisy_{}.txt'.format(rate/2)
            with open(label_noise_file, 'r') as f:
                noisy_file_dir, noisy_label = [], []
                for i in f.read().splitlines():
                    noisy_file_dir.append(i.split(' ')[0])
                    noisy_label.append(int(i.split(' ')[1]))
            save_file = data_file.split('.')[0] + '_ood_uniform_noisy_{}.txt'.format(rate)
            #know every class number and add %rate
            num_per_class = np.zeros(class_number)
            for i in label:
                num_per_class[i] += 1
            rate = rate/2
            num_per_class = np.ceil(num_per_class*(rate/(1-rate)))
            random.shuffle(ood_file_dir)
            new_file_dir = ood_file_dir[:int(num_per_class.sum())]
            new_label = []
            for i, j in enumerate(num_per_class):
                new_label.extend([i]*int(j))
            
            with open(save_file,'w') as f:
                    for i, d in enumerate(noisy_file_dir):
                        f.write('{} {} {}\n'.format(d, noisy_label[i], noisy_label[i]))
                    for i, d in enumerate(new_file_dir):
                        f.write('{} {} {}\n'.format(d, new_label[i], class_number+1))
        '''


        #############################################################
        ##            Todo mix: feature noise + OOD noise          ##
        #############################################################
        # 需要先生成 feature noise 的txt文件
        '''
        ood_dataset = "tinyimagenet.txt" #Introduce other dataset
        with open(ood_dataset, 'r') as f:
            ood_file_dir, ood_label = [], []
            for i in f.read().splitlines():
                ood_file_dir.append(i)
        ood_noisy_rate = [0.4]
        for rate in ood_noisy_rate:
            feature_noise_file = data_file.split('.')[0] + '_feature_noisy_{}.txt'.format(rate/2)
            with open(feature_noise_file, 'r') as f:
                noisy_file_dir, noisy_label = [], []
                for i in f.read().splitlines():
                    noisy_file_dir.append(i.split(' ')[0])
                    noisy_label.append(int(i.split(' ')[1]))
            save_file = data_file.split('.')[0] + '_ood_feature_noisy_{}.txt'.format(rate)
            #know every class number and add %rate
            num_per_class = np.zeros(class_number)
            for i in label:
                num_per_class[i] += 1
            rate = rate/2
            num_per_class = np.ceil(num_per_class*(rate/(1-rate)))
            random.shuffle(ood_file_dir)
            new_file_dir = ood_file_dir[:int(num_per_class.sum())]
            new_label = []
            for i, j in enumerate(num_per_class):
                new_label.extend([i]*int(j))
            
            with open(save_file,'w') as f:
                    for i, d in enumerate(noisy_file_dir):
                        f.write('{} {} {}\n'.format(d, noisy_label[i], noisy_label[i]))
                    for i, d in enumerate(new_file_dir):
                        f.write('{} {} {}\n'.format(d, new_label[i], class_number+1))
        '''
    

        #############################################################
        ##     Todo mix: feature noise + label noise + ood noise   ##
        #############################################################
        # 需要先生成 feature noise + label noise 的txt文件
        '''
        ood_dataset = "tinyimagenet.txt" #Introduce other dataset
        with open(ood_dataset, 'r') as f:
            ood_file_dir, ood_label = [], []
            for i in f.read().splitlines():
                ood_file_dir.append(i)
        ood_noisy_rate = [0.6]
        for rate in ood_noisy_rate:
            mixed_noise_file = data_file.split('.')[0] + '_feature_uniform_noisy_{}.txt'.format(0.4)
            with open(mixed_noise_file, 'r') as f:
                noisy_file_dir, noisy_label = [], []
                for i in f.read().splitlines():
                    noisy_file_dir.append(i.split(' ')[0])
                    noisy_label.append(int(i.split(' ')[1]))
            save_file = data_file.split('.')[0] + '_ood_feature_uniform_noisy_{}.txt'.format(rate)
            #know every class number and add %rate
            num_per_class = np.zeros(class_number)
            for i in label:
                num_per_class[i] += 1
            rate = 0.2
            num_per_class = np.ceil(num_per_class*(rate/(1-rate)))
            random.shuffle(ood_file_dir)
            new_file_dir = ood_file_dir[:int(num_per_class.sum())]
            new_label = []
            for i, j in enumerate(num_per_class):
                new_label.extend([i]*int(j))
            
            with open(save_file,'w') as f:
                    for i, d in enumerate(noisy_file_dir):
                        f.write('{} {} {}\n'.format(d, noisy_label[i], noisy_label[i]))
                    for i, d in enumerate(new_file_dir):
                        f.write('{} {} {}\n'.format(d, new_label[i], class_number+1))
        '''