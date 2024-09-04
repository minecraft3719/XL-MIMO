from keras.layers import Input, Dense, Dropout, Conv1D, Conv2D, MaxPool2D, BatchNormalization, Add, Activation, Subtract, Flatten
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from numpy import *
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   #allow growth
import scipy.io as sio

N=256 # BS antennas
snr_min=-10
snr_max=20
snr_increment=5

scale=1
Nx = 16
Ny = 16
nmseSummary = zeros((7,3),dtype=float)
dataType = ['SNR', 'decoded', 'estimate']
count = 0
for snr in range (snr_min,snr_max+snr_increment,snr_increment):
    P=10**(snr/10.0)
    ############## testing set ##################
    data_num_test=2000
    ## load channel
    H_noisy_in_test=zeros((data_num_test,Nx,Ny,2), dtype=float)
    H_true_out_test=zeros((data_num_test,Nx,Ny,2), dtype=float)
    data1 = sio.loadmat('Channel_f1n5_256ANTS_10by200')
    channel = data1['Channel_mat']
    for i in range(data_num_test):
        h = channel[i]
        H = np.reshape(h, (Nx,Ny))
        H_true_out_test[i, :, :, 0] = np.real(H)
        H_true_out_test[i, :, :, 1] = np.imag(H)
        noise = 1 / np.sqrt(2) * np.random.randn(Nx,Ny) + 1j * 1 / np.sqrt(2) * np.random.randn(Nx,Ny)
        H_noisy = H + 1 / np.sqrt(P) * noise
        H_noisy_in_test[i, :, :, 0] = np.real(H_noisy)
        H_noisy_in_test[i, :, :, 1] = np.imag(H_noisy)

    # load model
    ResCNN2d = load_model('ResCNN9_f1n5_256ANTS_1Kby100data_20dB_200ep.h5',compile=False)
    ResCNN2d.summary()

    decoded_channel = ResCNN2d.predict(H_noisy_in_test)
    print(((H_noisy_in_test)**2).mean(),((H_true_out_test)**2).mean(),((decoded_channel)**2).mean())
    nmse1=zeros((data_num_test,1), dtype=float)
    nmse2=zeros((data_num_test,1), dtype=float)
    for n in range(data_num_test):
        MSE1 = ((H_true_out_test[n,:,:,:] - H_noisy_in_test[n,:,:,:]) ** 2).sum()
        MSE2=((H_true_out_test[n,:,:,:]-decoded_channel[n,:,:,:])**2).sum()
        norm_real=((H_true_out_test[n,:,:,:])**2).sum()
        nmse1[n] = MSE1 / norm_real
        nmse2[n]=MSE2/norm_real
    print(nmse1.sum()/(data_num_test), nmse2.sum()/(data_num_test))
    nmseSummary[count,0] = snr
    nmseSummary[count,1] = nmse1.sum()/(data_num_test)
    nmseSummary[count,2] = nmse2.sum()/(data_num_test)
    count=count+1
print(dataType)
print(nmseSummary)

