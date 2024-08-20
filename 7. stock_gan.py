import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Reshape, BatchNormalization, Activation, Conv1D, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler

# 加载数据
messages = pd.read_csv('message_file.csv')
orderbook = pd.read_csv('orderbook_file.csv')

# 确保两个DataFrame的长度相同
assert len(messages) == len(orderbook)

# 数据预处理
historyLength = 50
orderLength = messages.shape[1] + orderbook.shape[1]
batch_size = 32  # 或者您选择的其他值
mini_batch_size = batch_size
noiseLength = 100
lstm_out_length = 64

print(f"orderLength: {orderLength}")

# 创建历史数据
history_data = np.hstack((messages.values, orderbook.values))
X = np.array([history_data[i:i+historyLength] for i in range(len(history_data)-historyLength)])

# 创建当前时间槽数据（假设第一列是时间）
current_time = messages.iloc[historyLength:, 0].values % 24  # 假设时间是24小时制

# 创建噪声数据
noise = np.random.normal(0, 1, (len(X), noiseLength))

# 创建真实数据（用于训练判别器）
y = history_data[historyLength:]

# 数据归一化

full_scaler = MinMaxScaler(feature_range=(-1, 1))
y_normalized = full_scaler.fit_transform(y)
X_normalized = np.array([full_scaler.transform(hist) for hist in X])

# 为最佳买价和最佳卖价创建一个单独的 scaler
bid_ask_scaler = MinMaxScaler(feature_range=(-1, 1))
bid_ask_scaler.fit(y[:, :2])

# 定义输入
history = Input(shape=(historyLength, orderLength), name='history_full')
history_input = Input(shape=(1,), name='history_time')
noise_input = Input(shape=(noiseLength,), name='noise_input_1')
truth_input = Input(shape=(2, 1, 1), name='truth_input')

# LSTM层
lstm_output = LSTM(lstm_out_length, return_sequences=False, name='lstm_generator')(history)
lstm_output_h = LSTM(lstm_out_length, return_sequences=False, name='lstm_critic')(history)

# 生成器输入
gen_input = Concatenate(axis=-1)([
    Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=1), rep=lstm_out_length, axis=1))(history_input),
    Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=1), rep=lstm_out_length, axis=1))(lstm_output),
    Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=1), rep=lstm_out_length, axis=1))(noise_input)
])

print("gen_input shape:", K.int_shape(gen_input))

# 生成器
G_1 = Sequential(name='generator_1')
G_1.add(Conv1D(64, 3, padding='same', input_shape=(lstm_out_length, 1 + lstm_out_length + noiseLength)))
G_1.add(BatchNormalization())
G_1.add(Activation('relu'))
G_1.add(Conv1D(32, 3, padding='same'))
G_1.add(BatchNormalization())
G_1.add(Activation('relu'))
G_1.add(Conv1D(16, 3, padding='same'))
G_1.add(BatchNormalization())
G_1.add(Activation('relu'))
G_1.add(Conv1D(1, 3, padding='same'))
G_1.add(Activation('tanh'))
G_1.add(Flatten())
G_1.add(Dense(orderLength * mini_batch_size))
G_1.add(Reshape((mini_batch_size, orderLength, 1)))
gen_output_1 = G_1(gen_input)

print("gen_output_1 shape:", K.int_shape(gen_output_1))
print("history_input shape:", K.int_shape(history_input))
print("lstm_output shape:", K.int_shape(lstm_output))
print("noise_input shape:", K.int_shape(noise_input))

# 提取orderbook历史
orderbook_history = Lambda(lambda x: x[:,-1,messages.shape[1]:], output_shape=(orderbook.shape[1],))(history)
print("orderbook_history shape:", K.int_shape(orderbook_history))

# 连接生成器输出和orderbook历史
gen_output_1_flat = Flatten()(gen_output_1)
cda_input = Concatenate(axis=1)([gen_output_1_flat, orderbook_history])
print("cda_input shape:", K.int_shape(cda_input))

# CDA网络
G_2 = Sequential(name='orderbook_gen')
G_2.add(Dense(256, input_shape=(K.int_shape(cda_input)[1],)))
G_2.add(BatchNormalization())
G_2.add(Activation('relu'))
G_2.add(Dense(128))
G_2.add(BatchNormalization())
G_2.add(Activation('relu'))
G_2.add(Dense(64))
G_2.add(BatchNormalization())
G_2.add(Activation('relu'))
G_2.add(Dense(2))  # 输出 2 个值，分别代表最佳买价和最佳卖价
G_2.add(Activation('tanh'))

gen_output_2 = G_2(cda_input)
print("gen_output_2 shape:", K.int_shape(gen_output_2))

# 最终生成器输出
gen_output = Reshape((2, 1, 1))(gen_output_2)
print("gen_output shape:", K.int_shape(gen_output))

# Critic
discriminator_input_fake = Concatenate(axis=-1)([
    Lambda(lambda x: K.tile(K.expand_dims(K.expand_dims(x, axis=1), axis=1), [1, 2, 1, 1]))(history_input),
    Lambda(lambda x: K.tile(K.reshape(x, (-1, 1, 1, lstm_out_length)), [1, 2, 1, 1]))(lstm_output_h),
    gen_output
])

discriminator_input_truth = Concatenate(axis=-1)([
    Lambda(lambda x: K.tile(K.expand_dims(K.expand_dims(x, axis=1), axis=1), [1, 2, 1, 1]))(history_input),
    Lambda(lambda x: K.tile(K.reshape(x, (-1, 1, 1, lstm_out_length)), [1, 2, 1, 1]))(lstm_output_h),
    truth_input
])

print("discriminator_input_fake shape:", K.int_shape(discriminator_input_fake))
print("discriminator_input_truth shape:", K.int_shape(discriminator_input_truth))

# Critic结构
D = Sequential(name='discriminator')
D.add(Conv2D(512, (3,3), padding='same', input_shape=(2, 1, 1 + lstm_out_length + 1)))
D.add(Activation('relu'))
D.add(Conv2D(256, (3,3), padding='same'))
D.add(Activation('relu'))
D.add(Conv2D(128, (3,3), padding='same'))
D.add(Activation('relu'))
D.add(Flatten())
D.add(Dense(1))

discriminator_output_fake = D(discriminator_input_fake)
discriminator_output_truth = D(discriminator_input_truth)

# 定义模型
generator = Model(inputs=[history_input, history, noise_input], outputs=gen_output, name='generator')

model_truth = Model(inputs=[history_input, history, noise_input, truth_input],
                    outputs=[discriminator_output_fake, discriminator_output_truth],
                    name='model_truth')
model_fake = Model(inputs=[history_input, history, noise_input],
                   outputs=discriminator_output_fake,
                   name='model_fake')

# 定义损失函数
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

# 编译模型
optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
generator.compile(optimizer=optimizer, loss='binary_crossentropy')

# 设置model_truth中生成器不可训练
for layer in model_truth.layers:
    layer.trainable = False
model_truth.get_layer(name='discriminator').trainable = True
model_truth.get_layer(name='lstm_critic').trainable = True
model_truth.compile(optimizer=optimizer, loss=[wasserstein_loss, wasserstein_loss])

# 设置model_fake中判别器不可训练
for layer in model_fake.layers:
    layer.trainable = True
model_fake.get_layer(name='discriminator').trainable = False
model_fake.get_layer(name='lstm_critic').trainable = False
model_fake.compile(optimizer=optimizer, loss=wasserstein_loss)

# 打印模型摘要
generator.summary()
model_fake.summary()
model_truth.summary()

# 训练循环
epochs = 1000
n_critic = 5

for epoch in range(epochs):
    for _ in range(n_critic):
        # 选择一个随机批次
        idx = np.random.randint(0, X_normalized.shape[0], batch_size)
        real_data = tf.convert_to_tensor(y_normalized[idx, :2].reshape(batch_size, 2, 1, 1), dtype=tf.float32)
        history_data = tf.convert_to_tensor(X_normalized[idx], dtype=tf.float32)
        current_time_data = tf.convert_to_tensor(current_time[idx].reshape(-1, 1), dtype=tf.float32)
        noise_data = tf.convert_to_tensor(np.random.normal(0, 1, (batch_size, noiseLength)), dtype=tf.float32)

        # 训练判别器
        fake_data = generator([current_time_data, history_data, noise_data])
        d_loss = model_truth.train_on_batch([current_time_data, history_data, noise_data, real_data],
                                            [np.ones((batch_size, 1)), -np.ones((batch_size, 1))])

    # 训练生成器
    g_loss = model_fake.train_on_batch([current_time_data, history_data, noise_data], np.ones((batch_size, 1)))

    if epoch % 1 == 0:
        print(f"Epoch {epoch}, D loss: {d_loss}, G loss: {g_loss}")

# 生成预测
start_time = datetime.strptime("2012-06-12 09:30:00", "%Y-%m-%d %H:%M:%S")
time_range = [start_time + timedelta(seconds=i) for i in range(len(X_normalized))]

# 生成预测
noise = np.random.normal(0, 1, (len(X_normalized), noiseLength))
generated_data = generator.predict([current_time[:len(X_normalized)].reshape(-1, 1), X_normalized, noise])
generated_data = generated_data.reshape(-1, 2)  # 调整形状为 (samples, 2)

print("Generated data shape:", generated_data.shape)

# 反归一化生成的数据
generated_data_original = bid_ask_scaler.inverse_transform(generated_data)

# 提取实际数据中的最佳买价和最佳卖价
real_best_bid = y[:, 0]  # 假设第一列是最佳买价
real_best_ask = y[:, 1]  # 假设第二列是最佳卖价

# 创建图表
plt.figure(figsize=(15, 10))

# 绘制实际数据
plt.plot(time_range, real_best_bid, label='Real Best Bid', color='blue', linewidth=1, alpha=0.7)
plt.plot(time_range, real_best_ask, label='Real Best Ask', color='red', linewidth=1, alpha=0.7)

# 绘制生成的数据
plt.plot(time_range, generated_data_original[:, 0], label='Generated Best Bid', color='cyan', linewidth=1, linestyle='--')
plt.plot(time_range, generated_data_original[:, 1], label='Generated Best Ask', color='magenta', linewidth=1, linestyle='--')

plt.title('Comparison of Real and Generated Best Bid/Ask Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

# 设置x轴为时间格式
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.gcf().autofmt_xdate()  # 自动旋转日期标签

plt.tight_layout()
plt.show()
