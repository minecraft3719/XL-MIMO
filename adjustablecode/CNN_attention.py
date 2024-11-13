from keras.layers import Input, Dense, Dropout, Conv1D, Conv2D, Reshape, MaxPool2D, MaxPooling2D, Attention, UpSampling2D, BatchNormalization, Add, Activation, Subtract, Flatten, Conv2DTranspose, Concatenate
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

np.random.seed(2808)

N=256 # BS antennas need to matching with matlab
scale=1
Nx = int(np.sqrt(N))
Ny = Nx
snr_min_train=0
snr_max_train=20
snr_increment_train=5
snr_count_train = int((snr_max_train-snr_min_train)/snr_increment_train)


############## training set ##################
train_dir = r'adjustablecode\(output)XL-MIMO'
model_dir = train_dir + r'/matlab_channel/model_input_python/'
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
model_file = r'Channel_f10n10_Total_Model100000_256ANTS_10by200.mat'
print("Channel model location: ", model_dir + model_file)

## load channel
data1 = sio.loadmat(model_dir + model_file)
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
# Residual Block with Attention Mechanism
def residual_block_with_attention(x, filters, kernel_size=6):
    # First Convolutional Layer
    shortcut = x
    x = Conv2D(filters, kernel_size, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    
    # Second Convolutional Layer
    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    
    # Attention Layer
    attn = Attention()([x, x])
    x = Add()([x, attn])  # Add attention-enhanced feature map

    # Residual Connection
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    
    return x

# Model with Residual Blocks and Attention Mechanism
def build_residual_attention_cnn(input_shape, output_dim, num_residual_blocks=5):
    inp = Input(shape=input_shape)
    x = inp
    
    # Initial Convolution
    x = Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    
    # Residual Blocks with Attention
    for _ in range(num_residual_blocks):
        x = residual_block_with_attention(x, 64)
    
    # Output Layer
    x = Conv2D(output_dim, kernel_size=3, padding="same", activation="linear")(x)
    output = Subtract()([inp, x])  # Residual addition for denoising
    
    model = Model(inputs=inp, outputs=output)
    return model

# Define model
input_dim = (16, 16, 2)  # Nx, Ny, 2 (or customize based on your data)
output_dim = 2
model = build_residual_attention_cnn(input_dim, output_dim)
model.summary()

# checkpoint;
if not os.path.isdir(train_dir + r'/keras_model/'):
    os.mkdir(train_dir + r'/keras_model/')
filename = 'CNN_attention_subtract_3_f10n10_256ANTS_100kdata_100ep_mix_SNR_0_5_20'
filepath = train_dir + r'/keras_model/' + filename + '.keras'
print('model checkpoint location: ', filepath)


adam=Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam, loss='mse')
model.summary()

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history_callback = model.fit(x=H_noisy_in, y=H_true_out, epochs=200, batch_size=512, callbacks=callbacks_list
                             , verbose=1, shuffle=True, validation_split=0.1)

loss_history = history_callback.history["loss"]
numpy_loss_history = np.array(loss_history)
np.savetxt(train_dir + r"/keras_model/" + filename + ".txt", numpy_loss_history, delimiter=",")
print('model loss log location: ', train_dir , r"/keras_model/" + filename + ".keras")

model.save(filepath,save_format='keras',overwrite=True)

#subprocess.Popen(filepath) ### return access denied

############## testing set ##################
data_num_test=100000
## load channel
data1 = sio.loadmat(model_dir + model_file) ##use train data for testing

channel = data1['Channel_mat_total']

# load model
ResCNN2d = load_model(filepath,compile="True")


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
    
    nmse=zeros((data_num_test,1), dtype=float)
    for n in range(data_num_test):
        MSE = ((H_true_out_test[n,:,:,:] - decoded_channel[n,:,:,:])**2).sum()
        norm_real=((H_true_out_test[n,:,:,:])**2).sum()
        nmse[n]= MSE /norm_real
    nmseSummary[count,0] = snr
    nmseSummary[count,1] = nmse.sum()/(data_num_test)
    count = count + 1
label = ['SNR','H_noise','H_decoded']
nmseSummary_as_str = np.vstack((label,nmseSummary))
print("nmseSummary out put size", nmseSummary.shape)
print(nmseSummary_as_str)

if not os.path.isdir(train_dir+"/nmse_output/"):
    os.mkdir(train_dir+"/nmse_output/")
np.savetxt(train_dir+"/nmse_output/"+time.strftime("%Y%m%d-%H%M%S")+'_nmseSummary_train.csv', nmseSummary_as_str, delimiter=',', fmt='%s')

print("nmse output location/",train_dir,"/nmse_output/")
