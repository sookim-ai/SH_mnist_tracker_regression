import tensorflow as tf
from function import *
import numpy as np
import random


def read_input(path):
    image_inn=np.load("./moving_mnist_dataset/out.npz")
    label_inn=np.load("./moving_mnist_dataset/out_label.npz")
    size=4000 #4000 = 100k, 8000 = 200k
    image=np.reshape(image_inn['arr_0'],[size,25,10,64,64,1])
    label=np.reshape(label_inn['arr_0'],[size,25,10,64,64,1])
    itr=20
    image_out=np.zeros([size,25,itr+10,64,64,1])
    label_out=np.zeros([size,25,itr+10,64,64,1]) #train itr times
    #Swap first input
    for i in range(size):
        for k in range(25):
            one_char=label[i,k,0,:,:,:]
            image[i,k,0,:,:,:]=one_char
            for j in range(itr+10):
                if j<(itr+1):
                    label_out[i,k,j,:,:,:]=one_char
                    image_out[i,k,j,:,:,:]=one_char
                else:
                    label_out[i,k,j,:,:,:]=label[i,k,j-itr,:,:,:]
                    image_out[i,k,j,:,:,:]=image[i,k,j-itr,:,:,:]
    d,__,__,__,__,__ = np.shape(image_out)
    te_image=np.asarray(image_out[0:int(d*0.02)])
    te_label=np.asarray(label_out[0:int(d*0.02)]) #NOTE
    tr_image=np.asarray(image_out[int(d*0.020):d-10])
    tr_label=np.asarray(label_out[int(d*0.020):d-10])
    va_image=np.asarray(image_out[d-10:d])
    va_label=np.asarray(label_out[d-10:d])
    np.save("X_test.npy",te_image)
    np.save("Y_test.npy",te_label)
    print(np.shape(tr_image),np.shape(tr_label),np.shape(te_image),np.shape(te_label),np.shape(va_image),np.shape(va_label))
    return tr_image,tr_label,te_image,te_label,va_image,va_label


