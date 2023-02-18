import cv2
import numpy as np
import matplotlib.pyplot as plt
# from paddle.vision.datasets import Cifar10
from timeit import time

def img_hist(im,name):
    im = np.array(im)
    plt.hist(im.flatten(), bins = 256)
    plt.title(name)
    plt.show()


def hist(image):
    size = 16
    pad_size = 8

    # image = plt.imread(image)
    hist = np.zeros([size+pad_size*2,size+pad_size*2,3],dtype=np.float32)

    bhist = cv2.calcHist([image], [0], None, [256], [0, 256])
    ghist = cv2.calcHist([image], [1], None, [256], [0, 256])
    rhist = cv2.calcHist([image], [2], None, [256], [0, 256])

    hist[:, :, 0] = np.pad(bhist.reshape(size, size), pad_size, 'constant').astype(np.float32)
    hist[:, :, 1] = np.pad(ghist.reshape(size, size), pad_size, 'constant').astype(np.float32)
    hist[:, :, 2] = np.pad(rhist.reshape(size, size), pad_size, 'constant').astype(np.float32)

    return hist



def main():
    img = cv2.imread(r"E:\Postgraduate work\3trer\code\enc\913.jpg")
    height, width , deep = img.shape
    print('img.shape==', img.shape)
    num_patches = 16  # 分块数量4*4
    print('num_patches==', num_patches * num_patches)
    dis_h = int(height/num_patches)  # 子块的高度
    dis_w = int(width/num_patches)   # 子块的宽度

    start = True
    total_time_start = time.time()

    # for p in range(10):
    # #     hist1 = hist(img)

    for i in range(num_patches):
        for j in range(num_patches):
            # print('i,j={}{}'.format(i, j))
            sub = img[dis_h * i:dis_h * (i + 1), dis_w * j:dis_w * (j + 1), :]       # 图像分块

            ############子块特征提取

            bhist = cv2.calcHist([sub], [0], None, [256], [0, 255])
            ghist = cv2.calcHist([sub], [1], None, [256], [0, 255])
            rhist = cv2.calcHist([sub], [2], None, [256], [0, 255])


            bghist = np.concatenate((bhist, ghist), axis=0)
            hist_sub = np.concatenate((bghist, rhist), axis=0)
            # print(bghist.dtype)

            if start:
                hist_img = hist_sub
                start = False
            else:
                hist_img = np.concatenate((hist_img , hist_sub), axis=1)

    hist_img = hist_img.transpose(1,0)



    print('feature time =', time.time() - total_time_start)
    # print(hist_img.shape,hist_img.dtype)
    # print(hist_img)

if __name__ == '__main__':
    main()