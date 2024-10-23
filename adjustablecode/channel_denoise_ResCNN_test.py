from keras.layers import Input, Dense, Dropout, Conv1D, Conv2D, MaxPool2D, BatchNormalization, Add, Activation, Subtract, Flatten
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from numpy import *
import time
import numpy as np
import os
import sys
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
count = 0
snr_count = int((snr_max-snr_min)/snr_increment)

# load model
train_dir = r'../../(output)XL-MIMO'
model_dir = train_dir + r'/matlab_channel/model_input_python/'
model_file = r'Channel_f4n10_Total_Model100000_256ANTS_10by200.mat'

keras_model = train_dir + r'/mr_Son_training_result_single_SNR/ResCNN9_f10n10_256ANTS_1Kby100kdata_20dB_200ep.keras'
ResCNN2d = load_model(keras_model,compile="True")
ResCNN2d.summary()

data1 = sio.loadmat(model_file)
channel = data1['Channel_mat_total']
Lf = data1['Lf']
Ln = data1['Ln']
num_sta = int(data1['num_sta'][0][0])
num_ffading = int(data1['num_ffading'][0][0])
num_Channel = int(data1['num_Channel'][0][0])
data_num_test=num_Channel

nmseSummary = zeros((snr_count+1,8),dtype=float)
nmseSummary[:,0] = num_sta
nmseSummary[:,1] = num_ffading
nmseSummary[:,2] = data1['N']
nmseSummary[:,3] = Lf
nmseSummary[:,4] = Ln

for snr in range (snr_min,snr_max+snr_increment,snr_increment):
    P=10**(snr/10.0)
    ############## testing set ##################
    ## load channel
    H_noisy_in_test=zeros((data_num_test,Nx,Ny,2), dtype=float)
    H_true_out_test=zeros((data_num_test,Nx,Ny,2), dtype=float)

    for i in range(data_num_test):
        h = channel[i]
        H = np.reshape(h, (Nx,Ny))
        H_true_out_test[i, :, :, 0] = np.real(H)
        H_true_out_test[i, :, :, 1] = np.imag(H)
        noise = 1 / np.sqrt(2) * np.random.randn(Nx,Ny) + 1j * 1 / np.sqrt(2) * np.random.randn(Nx,Ny)
        H_noisy = H + 1 / np.sqrt(P) * noise
        H_noisy_in_test[i, :, :, 0] = np.real(H_noisy)
        H_noisy_in_test[i, :, :, 1] = np.imag(H_noisy)

    decoded_channel = ResCNN2d.predict(H_noisy_in_test)
    nmse1=zeros((data_num_test,1), dtype=float)
    nmse2=zeros((data_num_test,1), dtype=float)
    for n in range(data_num_test):
        MSE1 = ((H_true_out_test[n,:,:,:] - H_noisy_in_test[n,:,:,:]) ** 2).sum()
        MSE2=((H_true_out_test[n,:,:,:]-decoded_channel[n,:,:,:])**2).sum()
        norm_real=((H_true_out_test[n,:,:,:])**2).sum()
        nmse1[n] = MSE1 / norm_real
        nmse2[n]=MSE2/norm_real
    print(nmse1.sum()/(data_num_test), nmse2.sum()/(data_num_test))

    nmseSummary[count,5] = snr
    nmseSummary[count,6] = nmse1.sum()/(data_num_test)
    nmseSummary[count,7] = nmse2.sum()/(data_num_test)
    count=count+1
label = ['num_sta','num_ffading','matrix_size','far_field','near_field','SNR','H_noise','H_decoded']
print("size of matrix: ",sys.getsizeof(nmseSummary))
print("nmseSummary out put size ", nmseSummary.shape)
nmseSummary_as_str = np.vstack((label,nmseSummary))
print(nmseSummary_as_str)

np.savetxt(train_dir+"/nmse_output/"+time.strftime("%Y%m%d-%H%M%S")+'_nmseSummary_test.csv', nmseSummary_as_str, delimiter=',', fmt='%s')
print(data_num_test,num_sta,num_ffading)
print("nmse output location/",train_dir,"/nmse_output/")
