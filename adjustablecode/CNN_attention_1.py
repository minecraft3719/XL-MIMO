from keras.layers import Input, Dense, Dropout, Conv2D, Reshape, BatchNormalization, Add, Activation, Subtract, Attention
from keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
from numpy import *
import scipy.io as sio
import os, io
import time

# GPU setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
np.random.seed(2808)

# Parameters
N = 256  # BS antennas need to match with MATLAB
epochs = 200 
batch_size = 512
batch_size_test = 1024
filter = 64
K = 3

Nx = int(np.sqrt(N))
Ny = Nx
snr_min_train =0
snr_max_train = 20
snr_increment_train = 5
SNR_range = [0, 5, 10, 15, 20, 0, 5, 10, 15, 20]
snr_count_train = len(SNR_range)

# Initialize TensorBoard writer
log_dir = os.path.join("adjustablecode", "tensorboard_logs", time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)
writer = tf.summary.create_file_writer(log_dir)  # Create a FileWriter
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    update_freq='epoch',  # Log every epoch (or set your preferred frequency)
    write_graph=False,  # Optional: disable graph visualization
    write_images=False  # Optional: disable image summaries
)

############## Training Set ##################
train_dir = r'adjustablecode\(output)XL-MIMO'
model_dir = os.path.join(train_dir, r'matlab_channel/model_input_python/')
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
data_file = r'Channel_f10n10_Total_Model100000_256ANTS_10by200.mat'

## Load channel
data1 = sio.loadmat(os.path.join(model_dir, data_file))
channel = data1['Channel_mat_total']
data_num_train = int(data1['num_Channel'][0][0])

H_noisy_in = zeros((data_num_train * snr_count_train, Nx, Ny, 2), dtype=float)
H_true_out = zeros((data_num_train * snr_count_train, Nx, Ny, 2), dtype=float)

for snr in SNR_range:
    print('Generate noisy channel at snr = ', snr)
    P = 10**(snr / 10.0)
    count = 0
    for i in range(data_num_train):
        h = channel[i]
        H = np.reshape(h, (Nx, Ny))
        H_true_out[data_num_train * count + i, :, :, 0] = np.real(H)
        H_true_out[data_num_train * count + i, :, :, 1] = np.imag(H)
        noise = 1 / np.sqrt(2) * np.random.randn(Nx, Ny) + 1j * 1 / np.sqrt(2) * np.random.randn(Nx, Ny)
        H_with_noisy = H + 1 / np.sqrt(P) * noise
        H_noisy_in[data_num_train * count + i, :, :, 0] = np.real(H_with_noisy)
        H_noisy_in[data_num_train * count + i, :, :, 1] = np.imag(H_with_noisy)
    count += 1
Mean_H_noisy_in = ((H_noisy_in)**2).mean()
Mean_H_true_out = ((H_true_out)**2).mean()

# Residual Block with Attention Mechanism
def residual_block_with_attention(x, filters, kernel_size=10):
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
num_residual_blocks = 5
model = build_residual_attention_cnn(input_dim, output_dim, num_residual_blocks)

model_summary_str = io.StringIO()
model.summary(print_fn=lambda x: model_summary_str.write(x + '\n'))
model_summary_str = model_summary_str.getvalue()

# Log the model summary to TensorBoard using the writer
with writer.as_default():  # Use the writer to log
    tf.summary.text("Model Summary", model_summary_str, step=0)

# Checkpoint
if not os.path.isdir(train_dir + r'/keras_model/'):
    os.mkdir(train_dir + r'/keras_model/')

model_name = f"CNN_attention_{num_residual_blocks}blocks_{N}ANTS_{data_num_train//1000}kdata_{snr_min_train}_{snr_increment_train}_{snr_max_train}_SNR_x2.keras"
filepath = os.path.join(train_dir, 'keras_model', model_name)
print('Model checkpoint location:', filepath)

adam = Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam, loss='mse')

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, tensorboard_callback]

history_callback = model.fit(
    x=H_noisy_in,
    y=H_true_out,
    epochs=200,
    batch_size=512,
    callbacks=callbacks_list,
    verbose=1,
    shuffle=True,
    validation_split=0.1
)


# Log model name and data info as text
with writer.as_default():
    tf.summary.text("Data Info", f"Data name: {data_file}", step=0)
    tf.summary.text("Model Info", f"Model Name: {model_name}", step=0)
    tf.summary.text("Data Info", f"Mean H_noisy_in: {Mean_H_noisy_in:.6f}, Mean H_true_out: {Mean_H_true_out:.6f}", step=0)

############## Testing Set ##################
data_num_test = 100000
channel = data1['Channel_mat_total']
best_model = load_model(filepath, compile="True")

snr_min = -10
snr_max = 20
snr_increment = 5
snr_count = int((snr_max - snr_min) / snr_increment)
nmseSummary = zeros((snr_count + 1, 3), dtype=float)
count = 0

for snr in range(snr_min, snr_max + snr_increment, snr_increment):
    H_noisy_in_test = zeros((data_num_test, Nx, Ny, 2), dtype=float)
    H_true_out_test = zeros((data_num_test, Nx, Ny, 2), dtype=float)
    P = 10**(snr / 10.0)
    for i in range(data_num_test):
        h = channel[i]
        H = np.reshape(h, (Nx, Ny))
        H_true_out_test[i, :, :, 0] = np.real(H)
        H_true_out_test[i, :, :, 1] = np.imag(H)
        noise = 1 / np.sqrt(2) * np.random.randn(Nx, Ny) + 1j * 1 / np.sqrt(2) * np.random.randn(Nx, Ny)
        H_noisy = H + 1 / np.sqrt(P) * noise
        H_noisy_in_test[i, :, :, 0] = np.real(H_noisy)
        H_noisy_in_test[i, :, :, 1] = np.imag(H_noisy)

    decoded_channel = best_model.predict(H_noisy_in_test, batch_size=batch_size_test)

    nmse = zeros((data_num_test, 1), dtype=float)
    for n in range(data_num_test):
        MSE = ((H_true_out_test[n, :, :, :] - decoded_channel[n, :, :, :])**2).sum()
        norm_real = ((H_true_out_test[n, :, :, :])**2).sum()
        nmse[n] = MSE / norm_real
    nmseSummary[count, 0] = snr
    nmseSummary[count, 1] = nmse.sum() / data_num_test
    count += 1

# Log NMSE summary to TensorBoard
for i in range(nmseSummary.shape[0]):
    with writer.as_default():
        tf.summary.text("NMSE_train", f"NMSE_SNR_{nmseSummary[i, 0]}: {nmseSummary[i, 1]:.6f}", step=0)

writer.close()
