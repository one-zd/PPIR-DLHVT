import cv2 as cv
import numpy as np
import os
import joblib
import glob
import matplotlib.pyplot as plt
import random



filepaths = []  # 初始化列表用来
data = []
num_patche = 8  # 分块数量num_patche*num_patche

def all_files_path(rootDir,label):                              #######遍历文件下子文件的所有图像，一个子文件夹为1类，适合Corel 10k 82  ####################
    for root, dirs, files in os.walk(rootDir):     # 分别代表根目录、文件夹、文件
        for file in files:                         # 遍历文件
            file_path = os.path.join(root, file)   # 获取文件绝对路径
            print(file_path)
            feature = hist(file_path)
            data.append((feature, label,file_path))
            print(label)
            filepaths.append(file_path+' '+ str(label))            # 将文件路径添加进列表 + 标签
        for dir in dirs:                           # 遍历目录下的子目录
            dir_path = os.path.join(root, dir)     # 获取子目录路径
            label = label + 1
            all_files_path(dir_path,label)               # 递归调用
        return filepaths ,data


def one_files_path(file_path1):                  #################holiday数据集##################
    images_path = glob.glob(os.path.join(file_path1 + '*.jpg'))
    for image_path in images_path:
        print(image_path)
        feature = hist(image_path)
        name = os.path.split(image_path)[1]
        num = name.split('.jpg')[0]
        label = int((int(num) / 100) % 1000)
        print(label)
        data.append((feature, label))
    return data


def enc(img,dif_x,dif_y,num_patches):
    dis_h, dis_w, deep = img.shape
    l = dis_w * dis_h

    #2.  提取三通道，分别变化为1维
    lst_r = np.resize(img[:, :, 0],(l,))
    lst_g = np.resize(img[:, :, 1],(l,))
    lst_b = np.resize(img[:, :, 2],(l,))



    # 3. 构建密钥和加密后图像
    img_r_key = np.zeros([l, 1])
    img_g_key = np.zeros([l, 1])
    img_b_key = np.zeros([l, 1])
    img_enc = np.zeros([dis_h, dis_w, deep], dtype=np.uint8)

    #4. 置乱加密
    for i in reversed(range(l)):
        r = random.randint(0, i)
        g = random.randint(0, i)
        b = random.randint(0, i)
        img_r_key[i] = r
        img_g_key[i] = g
        img_b_key[i] = b
        lst_r[i], lst_r[r] = lst_r[r], lst_r[i]
        lst_g[i], lst_g[g] = lst_g[g], lst_g[i]
        lst_b[i], lst_b[b] = lst_b[b], lst_b[i]

    #值替换加密
        # space = 256/num_patches
        #
        # lst_r[i] = lst_r[i] - dif_x * space
        # lst_g[i] = lst_g[i] - dif_y * space
        # lst_b[i] = lst_b[i] - dif_x * dif_y * space


#########通道置换###########
    img_enc[:, :, 1] = lst_r.reshape(dis_h, dis_w)
    img_enc[:, :, 2] = lst_g.reshape(dis_h, dis_w)
    img_enc[:, :, 0] = lst_b.reshape(dis_h, dis_w)

    #加密前后hist可视化对比
    # rgb_hist(img, 'rgbhist')
    # rgb_hist(img_enc, 'rgbhist-enc')

    return img_enc



def hist(img):
    img = plt.imread(img)
    height, width , deep = img.shape
    enc_img = np.zeros([height, width, deep], dtype=np.uint8)  # 加密后的图像
    k = random.sample(range(0, num_patche * num_patche), num_patche * num_patche)

#####分块hist
    dis_h = int(height/num_patche)  # 子块的高度
    dis_w = int(width/num_patche)   # 子块的宽度
    start = True
    for i in range(num_patche):
        for j in range(num_patche):
            # print('i,j={}{}'.format(i, j))
            sub = img[dis_h * i:dis_h * (i + 1), dis_w * j:dis_w * (j + 1), :]       # 图像分块

            enc_sub = enc(sub,(k[num_patche * i + j] // num_patche),(k[num_patche * i + j] % num_patche),num_patche)                             # 子块加密

            enc_img[dis_h * (k[num_patche * i + j] // num_patche):dis_h * ((k[num_patche * i + j] // num_patche) + 1),
            dis_w * (k[num_patche * i + j] % num_patche):dis_w * ((k[num_patche * i + j] % num_patche) + 1), :] = enc_sub   ####子块间置乱


    for i in range(num_patche):
        for j in range(num_patche):

            sub_enc_img = enc_img[dis_h * i:dis_h * (i + 1), dis_w * j:dis_w * (j + 1), :]  # 加密图像分块

            # 子块特征提取
            bhist = cv.calcHist([sub_enc_img], [0],None,[256],[0,256])
            ghist = cv.calcHist([sub_enc_img], [1], None, [256], [0, 256])
            rhist = cv.calcHist([sub_enc_img], [2], None, [256], [0, 256])

            bghist = np.concatenate((bhist, ghist), axis=0)
            hist_sub = np.concatenate((bghist, rhist), axis=0)

            if start:
                hist_img = hist_sub
                start = False
            else:
                hist_img = np.concatenate((hist_img , hist_sub), axis=1)

    hist_img = hist_img.transpose(1,0)    #(16, 768)

    return hist_img



def main():
    train_data = r'E:\Postgraduate work\4polar code-watermark\code\our-220617\data\Corel10k\train'
    # test_data = r'E:\Postgraduate work\3trer\code\Corel10K82\test'

    # train_data = r'E:\Postgraduate work\3trer\code\data\shanghai\train'
    filepaths, data = all_files_path(train_data, -1)             ##################corel 10k and ox_paris

    # train_data = r'E:\Postgraduate work\3trer\code\data\holiday\train\*'  #####################################holiday
    # test_data = r'E:\Postgraduate work\3trer\code\data\holiday\test\*'
    # data = one_files_path(train_data)             ##########################holiday

    output = open('Corel10k_train_fire_path' + str(num_patche * num_patche) + '_data.pkl', 'wb')
    joblib.dump(data, output)
    output.close()

    # with open('test.txt', 'w+') as f:
    #     for filepath in filepaths:
    #         f.write(filepath + '\n')


if __name__ == '__main__':
    main()