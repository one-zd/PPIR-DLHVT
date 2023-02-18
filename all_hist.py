import cv2 as cv
import numpy as np
import os
import joblib
import glob
import matplotlib.pyplot as plt



filepaths = []  # 初始化列表用来
data = []


def hist(image):
    size = 16
    pad_size = 8

    image = plt.imread(image)
    hist = np.zeros([size+pad_size*2,size+pad_size*2,3],dtype=np.float32)

    bhist = cv.calcHist([image], [0], None, [256], [0, 255])
    ghist = cv.calcHist([image], [1], None, [256], [0, 255])
    rhist = cv.calcHist([image], [2], None, [256], [0, 255])

    hist[:, :, 0] = np.pad(bhist.reshape(size, size), pad_size, 'constant').astype(np.float32)
    hist[:, :, 1] = np.pad(ghist.reshape(size, size), pad_size, 'constant').astype(np.float32)
    hist[:, :, 2] = np.pad(rhist.reshape(size, size), pad_size, 'constant').astype(np.float32)

    return hist

def all_files_path(rootDir,label):                              #######遍历文件下子文件的所有图像，一个子文件夹为1类，适合Corel 10k 82  ####################
    for root, dirs, files in os.walk(rootDir):     # 分别代表根目录、文件夹、文件
        for file in files:                         # 遍历文件
            file_path = os.path.join(root, file)   # 获取文件绝对路径
            print(file_path)
            feature = hist(file_path)
            data.append((feature, label))
            print(label)
            filepaths.append(file_path+' '+ str(label))            # 将文件路径添加进列表 + 标签
        for dir in dirs:                           # 遍历目录下的子目录
            dir_path = os.path.join(root, dir)     # 获取子目录路径
            label = label + 1
            all_files_path(dir_path,label)               # 递归调用
        return filepaths ,data




def one_files_path(file_path1):
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

def main():
    train_data = r'E:\Postgraduate work\3trer\code\data\shanghai\test'
    # test_data = r'E:\Postgraduate work\3trer\code\Corel10K82\test'
    #
    filepaths ,data= all_files_path(train_data,-1)         ##################corel 10k
    # # filepaths, data = all_files_path(test_data, -1)

    #
    # test_data = r'E:\Postgraduate work\3trer\code\data\holiday\test\*'
    # data = one_files_path(test_data)



    output = open('shanghai_all_test_' + '_data.pkl', 'wb')
    joblib.dump(data, output)
    output.close()



if __name__ == '__main__':
    main()