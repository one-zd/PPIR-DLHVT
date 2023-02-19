#!/usr/bin/env python
# coding: utf-8

# In[8]:


#数据准备
# # 解压数据集
# !tar -xf /home/aistudio/data/data105740/ILSVRC2012_val.tar -C /home/aistudio/work/
# 解压权重文件
#!unzip -q -o /home/aistudio/data/data105741/pretrained.zip -d /home/aistudio/work/

# cp /home/aistudio/data/data192313/g.pdparams  /home/aistudio/work/


# In[2]:


#图像分块嵌入
# coding=utf-8
# 导入环境
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import paddle
from paddle.io import Dataset
from paddle.nn import Conv2D, MaxPool2D, Linear, Dropout, BatchNorm, AdaptiveAvgPool2D, AvgPool2D
import paddle.nn.functional as F
import paddle.nn as nn

# 图像分块、Embedding
class PatchEmbed(nn.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # 原始大小为int，转为tuple，即：img_size原始输入224，变换后为[224,224]
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # 图像块的个数
        num_patches = (img_size[1] // patch_size[1]) *             (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # kernel_size=块大小，即每个块输出一个值，类似每个块展平后使用相同的全连接层进行处理
        # 输入维度为3，输出维度为块向量长度
        # 与原文中：分块、展平、全连接降维保持一致
        # 输出为[B, C, H, W]
        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1],             "Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # [B, C, H, W] -> [B, C, H*W] ->[B, H*W, C]
        x = self.proj(x).flatten(2).transpose((0, 2, 1))

        return x


# In[3]:


#VIT完整
class VisionTransformer(nn.Layer):
    def __init__(self,
                 img_size=384,
                 patch_size=16,
                 in_chans=3,
                 class_dim=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 **args):
        super().__init__()
        self.class_dim = class_dim

        self.num_features = self.embed_dim = embed_dim
        # 图片分块和降维，块大小为patch_size，最终块向量维度为768
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        # 分块数量
        num_patches = 16
        # 可学习的位置编码
        self.pos_embed = self.create_parameter(
            shape=(1, num_patches + 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("pos_embed", self.pos_embed)
        # 人为追加class token，并使用该向量进行分类预测
        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("cls_token", self.cls_token)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)
        # transformer
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                epsilon=epsilon) for i in range(depth)
        ])

        self.norm = eval(norm_layer)(embed_dim, epsilon=epsilon)

        # Classifier head
        self.head = nn.Linear(embed_dim,
                              class_dim) if class_dim > 0 else Identity()

        trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        self.apply(self._init_weights)
    # 参数初始化
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
    # 获取图像特征
    def forward_features(self, x):

        B = paddle.shape(x)[0]     #[2, 3, 384, 384]

        # 将图片分块，并调整每个块向量的维度
        # x = self.patch_embed(x)    #[2, 576, 768]
        # 将class token与前面的分块进行拼接
        cls_tokens = self.cls_token.expand((B, -1, -1))
       
        x = paddle.concat((cls_tokens, x), axis=1)
        # 将编码向量中加入位置编码
        # x = x + self.pos_embed
        x = self.pos_drop(x)
        # 堆叠 transformer 结构
        for blk in self.blocks:
            x = blk(x)
        # LayerNorm
        x = self.norm(x)
        # 提取分类 tokens 的输出
        return x[:, 0]

    def forward(self, x):
        # 获取图像特征
        x = self.forward_features(x)
        # 图像分类
        x = self.head(x)
        return x


# In[4]:


#定义ViT网络
# 参数初始化配置
trunc_normal_ = nn.initializer.TruncatedNormal(std=.02)
zeros_ = nn.initializer.Constant(value=0.)
ones_ = nn.initializer.Constant(value=1.)

# 将输入 x 由 int 类型转为 tuple 类型
def to_2tuple(x):
    return tuple([x] * 2)

# 定义一个什么操作都不进行的网络层
class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


# In[5]:


#基础模块
def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output

class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5):
        super().__init__()
        self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        # Multi-head Self-attention
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        # Multi-head Self-attention， Add， LayerNorm
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # Feed Forward， Add， LayerNorm
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x   


# In[6]:


#多层感知机（MLP
class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 输入层：线性变换
        x = self.fc1(x)
        # 应用激活函数
        x = self.act(x)
        # Dropout
        x = self.drop(x)
        # 输出层：线性变换
        x = self.fc2(x)
        # Dropout
        x = self.drop(x)
        return x


# In[7]:


# Multi-head Attention 多头自注意
class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        # 计算 q,k,v 的转移矩阵
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # 最终的线性层
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        N, C = x.shape[1:]
        # 线性变换
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C //
                                   self.num_heads)).transpose((2, 0, 3, 1, 4))
        # 分割 query key value
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Scaled Dot-Product Attention
        # Matmul + Scale
        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        # SoftMax
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        # Matmul
        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        # 线性变换
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# In[ ]:





# In[8]:


#############corel_10k数据读取
from __future__ import print_function
import tarfile
import numpy as np
import six
from PIL import Image
from six.moves import cPickle as pickle
from scipy.io import loadmat
import paddle
from paddle.io import Dataset
from paddle.dataset.common import _check_exists_and_download


__all__ = []


class corel_10k_train(Dataset):

    def __init__(self,
                 transform=None, ):
        self.transform = transform
        # read dataset into memory
        self._load_data()
        self.dtype = paddle.get_default_dtype()

    def _load_data(self):
        self.data = []
        output = open('data/data112459/train_64_data.pkl', 'rb')
        self.data = pickle.load(output)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = paddle.to_tensor(image)
        return image, np.array(label).astype('int64')
        return image.astype(self.dtype), np.array(label).astype('int64')

    def __len__(self):
        return len(self.data)


class corel_10k_test(Dataset):

    def __init__(self,
                 transform=None, ):
        self.transform = transform
        # read dataset into memory
        self._load_data()
        self.dtype = paddle.get_default_dtype()

    def _load_data(self):
        self.data = []
        output = open('data/data112459/test_64_data.pkl', 'rb')
        self.data = pickle.load(output)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        image = paddle.to_tensor(image)


        # if self.transform is not None:
        #     image = self.transform(image)
        return image, np.array(label).astype('int64')
        return image.astype(self.dtype), np.array(label).astype('int64')

    def __len__(self):
        return len(self.data)


# In[9]:


#数据准备
import paddle.vision.transforms as T
# from paddle.vision.datasets import Cifar10

transform = T.Compose([
    # T.Resize(size=(384,384)),
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],data_format='HWC'),
    # T.ToTensor()
])


# train_dataset = Cifar10(mode='train', transform=transform)
# val_dataset = Cifar10(mode='test',  transform=transform)

# train_dataset = hist_Cifar10(mode='train', transform=transform)
# val_dataset = hist_Cifar10(mode='test',  transform=transform)

train_dataset = corel_10k_train(transform=transform)
val_dataset = corel_10k_test( transform=transform)

# train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, drop_last=True,num_workers=8,shuffle=True)
# # for batch_id, data in enumerate(train_loader()):
# #     x_data = data[0]
# #     y_data = data[1]
# #     print(y_data)
# #     print(x_data.shape)
# #     print(y_data.shape)
# #     break
# val_loader = paddle.io.DataLoader(val_dataset, batch_size=64, drop_last=True,num_workers=8,shuffle=True)


# In[10]:


#模型加载
use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

print('start ')
# 实例化模型
model = VisionTransformer(
        patch_size=16,
        class_dim=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_rate=0.,
        attn_drop_rate=0.5,
        drop_path_rate=0.5,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6)
# 加载模型参数
# params_file_path="/home/aistudio/work/ViT_base_patch16_384_pretrained.pdparams"   ####预训练参数
params_file_path="data/data192313/g.pdparams"    ######测试参数

model_state_dict = paddle.load(params_file_path)
model.load_dict(model_state_dict)
# model = paddle.Model(model)


#####子块间置乱加密前后模型输出对比
# image = cv2.imread('work/img/913_enc_img_4.jpg')
# rand_img = cv2.imread('work/img/913_enc_img_rand4.jpg')
# model.eval()
# image = hist(image)
# rand_img = hist(rand_img)
# image = paddle.to_tensor(image)
# print(image.shape)
# logits = model(image)
# rand_logits = model(rand_img)
# x = ((logits == rand_logits).sum() / x.size)
# print(x)


# In[13]:


#######检索测试
from timeit import time

def test(model):
    
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=128, drop_last=True,num_workers=4,shuffle=True)
    test_loader = paddle.io.DataLoader(val_dataset, batch_size=128, drop_last=True,num_workers=4,shuffle=True)

    model.eval()

    total_time_start = time.time()
    start_train = True
    for batch_id, data in enumerate(train_loader()):
        x_train = data[0]
        label_train = data[1].numpy()
        trn_feature = model.forward_features(x_train).numpy()
        # trn_feature = model(x_train).numpy()

        
        if start_train:
            y_train = label_train
            trn_features = trn_feature
            start_train = False
        else:
            y_train = np.concatenate((y_train,label_train),axis=0)
            trn_features = np.concatenate((trn_features,trn_feature),axis=0)

    start_test = True    
    for batch_id, data in enumerate(test_loader()):
        x_test = data[0]
        label_test = data[1].numpy()
        tst_feature = model.forward_features(x_test).numpy()
        # tst_feature = model(x_test).numpy()

        if start_test:
            y_test = label_test
            tst_features = tst_feature
            start_test = False
        else:
            y_test = np.concatenate((y_test,label_test),axis=0)
            tst_features = np.concatenate((tst_features,tst_feature),axis=0)

    print('feature time =', time.time() - total_time_start)

    tst_retrieval_trn = 1
    tst_retrieval_in_all = 0

    if tst_retrieval_trn == 1:
        
        query_times = tst_features.shape[0]
        trainset_len = trn_features.shape[0]
        trn_label = y_train
        tst_label = y_test
        AP = np.zeros(query_times)
        Ns = np.arange(1, trainset_len + 1)

        TOP1=[]
        TOP5=[]
        TOP10=[]
        TOP20=[]
        TOP40=[]
        TOP60=[]
        TOP80=[]
        TOP100=[]
        total_time_start = time.time()
        for index in range(query_times):
        # randomimg = [random.randint(0, 2000) for _ in range(40)]
        # for index in n:
            # print('Query path =', test_label[index])
            query_label = tst_label[index]
            query_features = tst_features[index, :]
            dist = np.sqrt(np.sum(np.square(query_features - trn_features), axis=1))
            sort_indices = np.argsort(dist).astype(int)
            # print(sort_indices)
            # for index2 in sort_indices[0:10]:
            #     print('result path =', train_label[index2])
            buffer_yes = np.equal(query_label, np.array(trn_label)[sort_indices]).astype(int)

            Top1matrix = buffer_yes[0:1]
            Top5matrix = buffer_yes[0:5]
            Top10matrix = buffer_yes[0:10]
            Top20matrix = buffer_yes[0:20]
            Top40matrix = buffer_yes[0:40]
            Top60matrix = buffer_yes[0:60]
            Top80matrix = buffer_yes[0:80]
            Top100matrix = buffer_yes[0:100]

            Top1 = sum(Top1matrix == 1) / 1
            Top5 = sum(Top5matrix == 1) / 5
            Top10 = sum(Top10matrix == 1) / 10
            Top20 = sum(Top20matrix == 1) / 20
            # print('precision',Top20)
            # print('recall',sum(Top20matrix == 1) / 80)
            Top40 = sum(Top40matrix == 1) / 40
            Top60 = sum(Top60matrix == 1) / 60
            Top80 = sum(Top80matrix == 1) / 80
            Top100 = sum(Top100matrix == 1) / 100

            TOP1.append(Top1)
            TOP5.append(Top5)
            TOP10.append(Top10)
            TOP20.append(Top20)
            TOP40.append(Top40)
            TOP60.append(Top60)
            TOP80.append(Top80)
            TOP100.append(Top100)
        #     P = np.cumsum(buffer_yes) / Ns
        #     AP[index] = np.sum(P * buffer_yes) / sum(buffer_yes)
        # mAP = np.mean(AP)
        # print('The Precision when test set retrieves the training set')
        print('total query time =', time.time() - total_time_start)
        print('Top1 =', sum(TOP1)/len(TOP1))
        print('Top5 =', sum(TOP5)/len(TOP5))
        print('Top10 =', sum(TOP10)/len(TOP10))
        print('Top20 =', sum(TOP20)/len(TOP20))
        print('Top40 =', sum(TOP40)/len(TOP40))
        print('Top60 =', sum(TOP60)/len(TOP60))
        print('Top80 =', sum(TOP80)/len(TOP80))
        print('Top100 =', sum(TOP100)/len(TOP100))
        # print('mean Average Precision =' , mAP)


    if tst_retrieval_in_all == 1:
        query_times = tst_features.shape[0]
        trainset_len = trn_features.shape[0]
        label = np.concatenate((y_train,y_test),axis=0)
        features = np.concatenate((trn_features,tst_features),axis=0)
        trn_label = y_train
        tst_label = y_test
        AP = np.zeros(query_times)
        Ns = np.arange(1, features.shape[0] + 1)
        total_time_start = time.time()
        TOP10 = []
        TOP20 = []
        TOP40 = []
        TOP60 = []
        TOP80 = []
        TOP100 = []
        for index in range(query_times):
            # print('Query ', index + 1)
            query_label = tst_label[index]
            query_features = tst_features[index, :]
            dist = np.sqrt(np.sum(np.square(query_features - features), axis=1))
            sort_indices = np.argsort(dist).astype(int)
            buffer_yes = np.equal(query_label, label[sort_indices]).astype(int)
            Top10matrix = buffer_yes[0:10]
            Top20matrix = buffer_yes[0:20]
            Top40matrix = buffer_yes[0:40]
            Top60matrix = buffer_yes[0:60]
            Top80matrix = buffer_yes[0:80]
            Top100matrix = buffer_yes[0:100]

            Top10 = sum(Top10matrix == 1) / 10
            Top20 = sum(Top20matrix == 1) / 20
            Top40 = sum(Top40matrix == 1) / 40
            Top60 = sum(Top60matrix == 1) / 60
            Top80 = sum(Top80matrix == 1) / 80
            Top100 = sum(Top100matrix == 1) / 100

            TOP10.append(Top10)
            TOP20.append(Top20)
            TOP40.append(Top40)
            TOP60.append(Top60)
            TOP80.append(Top80)
            TOP100.append(Top100)
            P = np.cumsum(buffer_yes) / Ns
            AP[index] = np.sum(P * buffer_yes) / sum(buffer_yes)
        mAP = np.mean(AP)
        print('The Precision when test set retrieves the entire data set')
        print('Top10 =', sum(TOP10) / len(TOP10))
        print('Top20 =', sum(TOP20) / len(TOP20))
        print('Top40 =', sum(TOP40) / len(TOP40))
        print('Top60 =', sum(TOP60) / len(TOP60))
        print('Top80 =', sum(TOP80) / len(TOP80))
        print('Top100 =', sum(TOP100) / len(TOP100))
        print('mean Average Precision =' , mAP)
        print('total query time =', time.time() - total_time_start)
        print("================")

test(model)


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
