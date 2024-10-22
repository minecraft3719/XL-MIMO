from keras.layers import Input, Dense, Dropout, Conv1D, Conv2D, MaxPool2D, BatchNormalization, Add, Activation, Subtract, Flatten
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from numpy import *
import subprocess
import time
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   #allow growth
import scipy.io as sio

N=256 # BS antennas need to matching with matlab
scale=1
Nx = int(np.sqrt(N))
Ny = Nx
snr_min_train=10
snr_max_train=30
snr_increment_train=5
snr_count_train = int((snr_max_train-snr_min_train)/snr_increment_train)


############## training set ##################
train_dir = '../../(output)XL-MIMO'
if not os.path.isdir(train_dir+'/matlab_channel/model_input_python/'):
    os.mkdir(train_dir+'/matlab_channel/model_input_python/')

## load channel
data1 = sio.loadmat(train_dir+'/matlab_channel/model_input_python/Channel_f10n10_Total_Model100000_256ANTS_10by200.mat')
channel = data1['Channel_mat_total']
data_num_train=int(data1['num_Channel'][0][0])
print("shape of channel model ",channel.shape)

H_noisy_in = zeros((data_num_train*snr_count_train,Nx,Ny,2), dtype=float)
H_true_out = zeros((data_num_train*snr_count_train,Nx,Ny,2), dtype=float)
for snr in range(snr_min_train,snr_max_train+snr_increment_train,snr_increment_train):
    print('Generate noisy channel at snr = ', snr)
    P=10**(snr/10.0)
    count = 0
    for i in range(data_num_train):
        h = channel[i]
        H = np.reshape(h, (Nx,Ny))
        H_true_out[data_num_train*count+i,:,:,0] = np.real(H)
        H_true_out[data_num_train*count+i,:,:,1] = np.imag(H)
        noise = 1 / np.sqrt(2) * np.random.randn(Nx,Ny) + 1j * 1 / np.sqrt(2) * np.random.randn(Nx,Ny)
        H_with_noisy = H + 1 / np.sqrt(P) * noise
        H_noisy_in[data_num_train*count+i,:,:,0] = np.real(H_with_noisy)
        H_noisy_in[data_num_train*count+i,:,:,1]= np.imag(H_with_noisy)
    count = count + 1
print(((H_noisy_in)**2).mean(),((H_true_out)**2).mean())
print(H_noisy_in.shape,H_true_out.shape)

## Build DNN model
K=3
input_dim = (Nx,Ny,2)
output_dim = 2
inp = Input(shape=input_dim)
xn = Conv2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu')(inp)
xn = BatchNormalization()(xn)
xn = Conv2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu')(xn)
xn = BatchNormalization()(xn)
xn = Conv2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu')(xn)
xn = BatchNormalization()(xn)
xn = Conv2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu')(xn)
xn = BatchNormalization()(xn)
xn = Conv2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu')(xn)
xn = BatchNormalization()(xn)
xn = Conv2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu')(xn)
xn = BatchNormalization()(xn)
xn = Conv2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu')(xn)
xn = BatchNormalization()(xn)
xn = Conv2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu')(xn)
xn = BatchNormalization()(xn)
xn = Conv2D(filters=output_dim, kernel_size=(K,K), padding='Same', activation='linear')(xn)
x1 = Subtract()([inp, xn])

model = Model(inputs=inp, outputs=x1)

# checkpoint;
if not os.path.isdir(train_dir+'/keras_model/'):
    os.mkdir(train_dir+'/keras_model/')
filepath = train_dir+'/keras_model/ResCNN9_f10n10_256ANTS_1Kby100kdata_20dB_200ep.keras'


adam=Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam, loss='mse')
model.summary()

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history_callback = model.fit(x=H_noisy_in, y=H_true_out, epochs=1, batch_size=128, callbacks=callbacks_list
                             , verbose=2, shuffle=True, validation_split=0.1)
subprocess.Popen(train_dir+'/keras_model/')
loss_history = history_callback.history["loss"]
numpy_loss_history = np.array(loss_history)
np.savetxt(train_dir+"/keras_model/loss_history.txt", numpy_loss_history, delimiter=",")

model.save(filepath,save_format='keras',overwrite=True)

############## testing set ##################
data_num_test=2000
## load channel
test_dir = train_dir+'/matlab_channel/model_input_python/'

data1 = sio.loadmat(test_dir+'Channel_f10n10_Total_Model100000_256ANTS_10by200.mat')
channel = data1['Channel_mat_total']

# load model
ResCNN2d = load_model(train_dir+'/keras_model/ResCNN9_f10n10_256ANTS_1Kby100kdata_20dB_200ep.keras',compile="True")


snr_min=-10
snr_max=20
snr_increment=5
snr_count = int((snr_max-snr_min)/snr_increment)
nmseSummary = zeros((snr_count + 1,3),dtype=float)
count = 0
for snr in range(snr_min,snr_max+snr_increment,snr_increment):
    H_noisy_in_test=zeros((data_num_test,Nx,Ny,2), dtype=float)
    H_true_out_test=zeros((data_num_test,Nx,Ny,2), dtype=float)
    P=10**(snr/10.0)
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
    print(((H_noisy_in_test)**2).mean(),((H_true_out_test)**2).mean(),((decoded_channel)**2).mean())
    nmse1=zeros((data_num_test,1), dtype=float)
    nmse2=zeros((data_num_test,1), dtype=float)
    for n in range(data_num_test):
        MSE1 = ((H_true_out_test[n,:,:,:] - H_noisy_in_test[n,:,:,:]) ** 2).sum()
        MSE2 = ((H_true_out_test[n,:,:,:] - decoded_channel[n,:,:,:])**2).sum()
        norm_real=((H_true_out_test[n,:,:,:])**2).sum()
        nmse1[n] = MSE1 / norm_real
        nmse2[n]= MSE2 /norm_real
    nmseSummary[count,0] = snr
    nmseSummary[count,1] = nmse1.sum()/(data_num_test)
    nmseSummary[count,2] = nmse2.sum()/(data_num_test)
    count = count + 1
label = ['SNR','H_noise','H_decoded']
nmseSummary_as_str = np.vstack((label,nmseSummary))
print("nmseSummary out put size", nmseSummary.shape)
print(nmseSummary_as_str)

if not os.path.isdir(train_dir+"/nmse_output/"):
    os.mkdir(train_dir+"/nmse_output/")
np.savetxt(train_dir+"/nmse_output/"+time.strftime("%Y%m%d-%H%M%S")+'_nmseSummary_train.csv', nmseSummary_as_str, delimiter=',', fmt='%s')
subprocess.Popen(train_dir+'/nmse_output/')