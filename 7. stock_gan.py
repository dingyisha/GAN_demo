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

# Load data
messages = pd.read_csv('message_file.csv')
orderbook = pd.read_csv('orderbook_file.csv')

# Ensure both DataFrames have the same length
assert len(messages) == len(orderbook)

# Data preprocessing
historyLength = 50
orderLength = messages.shape[1] + orderbook.shape[1]
batch_size = 32  # Or any other value you choose
mini_batch_size = batch_size
noiseLength = 100
lstm_out_length = 64

print(f"orderLength: {orderLength}")

# Create historical data
history_data = np.hstack((messages.values, orderbook.values))
X = np.array([history_data[i:i+historyLength] for i in range(len(history_data)-historyLength)])

# Create current timeslot data (assuming the first column is time)
current_time = messages.iloc[historyLength:, 0].values % 24  # Assuming time is in 24-hour format

# Create noise data
noise = np.random.normal(0, 1, (len(X), noiseLength))

# Create real data (for training the discriminator)
y = history_data[historyLength:]

# Normalize data
full_scaler = MinMaxScaler(feature_range=(-1, 1))
y_normalized = full_scaler.fit_transform(y)
X_normalized = np.array([full_scaler.transform(hist) for hist in X])

# Create a separate scaler for best bid and ask prices
bid_ask_scaler = MinMaxScaler(feature_range=(-1, 1))
bid_ask_scaler.fit(y[:, :2])

# Define inputs
history = Input(shape=(historyLength, orderLength), name='history_full')
history_input = Input(shape=(1,), name='history_time')
noise_input = Input(shape=(noiseLength,), name='noise_input_1')
truth_input = Input(shape=(2, 1, 1), name='truth_input')

# LSTM layers
lstm_output = LSTM(lstm_out_length, return_sequences=False, name='lstm_generator')(history)
lstm_output_h = LSTM(lstm_out_length, return_sequences=False, name='lstm_critic')(history)

# Generator input
gen_input = Concatenate(axis=-1)([
    Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=1), rep=lstm_out_length, axis=1))(history_input),
    Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=1), rep=lstm_out_length, axis=1))(lstm_output),
    Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=1), rep=lstm_out_length, axis=1))(noise_input)
])

print("gen_input shape:", K.int_shape(gen_input))

# Generator Model
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

# Extract orderbook history
orderbook_history = Lambda(lambda x: x[:,-1,messages.shape[1]:], output_shape=(orderbook.shape[1],))(history)
print("orderbook_history shape:", K.int_shape(orderbook_history))

# Combine generator output and orderbook history
gen_output_1_flat = Flatten()(gen_output_1)
cda_input = Concatenate(axis=1)([gen_output_1_flat, orderbook_history])
print("cda_input shape:", K.int_shape(cda_input))

# CDA network
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
G_2.add(Dense(2))  # Outputs two values representing the best bid and ask prices
G_2.add(Activation('tanh'))

gen_output_2 = G_2(cda_input)
print("gen_output_2 shape:", K.int_shape(gen_output_2))

# Final generator output
gen_output = Reshape((2, 1, 1))(gen_output_2)
print("gen_output shape:", K.int_shape(gen_output))

# Critic Model
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

# Critic structure
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

# Define models
generator = Model(inputs=[history_input, history, noise_input], outputs=gen_output, name='generator')

model_truth = Model(inputs=[history_input, history, noise_input, truth_input],
                    outputs=[discriminator_output_fake, discriminator_output_truth],
                    name='model_truth')
model_fake = Model(inputs=[history_input, history, noise_input],
                   outputs=discriminator_output_fake,
                   name='model_fake')

# Define loss function
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

# Compile models
optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
generator.compile(optimizer=optimizer, loss='binary_crossentropy')

# Set the generator as non-trainable in model_truth
for layer in model_truth.layers:
    layer.trainable = False
model_truth.get_layer(name='discriminator').trainable = True
model_truth.get_layer(name='lstm_critic').trainable = True
model_truth.compile(optimizer=optimizer, loss=[wasserstein_loss, wasserstein_loss])

# Set the discriminator as non-trainable in model_fake
for layer in model_fake.layers:
    layer.trainable = True
model_fake.get_layer(name='discriminator').trainable = False
model_fake.get_layer(name='lstm_critic').trainable = False
model_fake.compile(optimizer=optimizer, loss=wasserstein_loss)

# Print model summaries
generator.summary()
model_fake.summary()
model_truth.summary()

# Training loop
epochs = 1000
n_critic = 5

for epoch in range(epochs):
    for _ in range(n_critic):
        # Select a random batch
        idx = np.random.randint(0, X_normalized.shape[0], batch_size)
        real_data = tf.convert_to_tensor(y_normalized[idx, :2].reshape(batch_size, 2, 1, 1), dtype=tf.float32)
        history_data = tf.convert_to_tensor(X_normalized[idx], dtype=tf.float32)
        current_time_data = tf.convert_to_tensor(current_time[idx].reshape(-1, 1), dtype=tf.float32)
        noise_data = tf.convert_to_tensor(np.random.normal(0, 1, (batch_size, noiseLength)), dtype=tf.float32)

        # Train discriminator
        fake_data = generator([current_time_data, history_data, noise_data])
        d_loss = model_truth.train_on_batch([current_time_data, history_data, noise_data, real_data],
                                            [np.ones((batch_size, 1)), -np.ones((batch_size, 1))])

    # Train generator
    g_loss = model_fake.train_on_batch([current_time_data, history_data, noise_data], np.ones((batch_size, 1)))

    if epoch % 1 == 0:
        print(f"Epoch {epoch}, D loss: {d_loss}, G loss: {g_loss}")

# Generate predictions
start_time = datetime.strptime("2012-06-12 09:30:00", "%Y-%m-%d %H:%M:%S")
time_range = [start_time + timedelta(seconds=i) for i in range(len(X_normalized))]

# Generate predictions
noise = np.random.normal(0, 1, (len(X_normalized), noiseLength))
generated_data = generator.predict([current_time[:len(X_normalized)].reshape(-1, 1), X_normalized, noise])
generated_data = generated_data.reshape(-1, 2)  # Adjust shape to (samples, 2)

print("Generated data shape:", generated_data.shape)

# De-normalize generated data
generated_data_original = bid_ask_scaler.inverse_transform(generated_data)

# Extract actual data's best bid and ask prices
real_best_bid = y[:, 0]  # Assuming the first column is the best bid price
real_best_ask = y[:, 1]  # Assuming the second column is the best ask price

# Create plot
plt.figure(figsize=(15, 10))

# Plot actual data
plt.plot(time_range, real_best_bid, label='Real Best Bid', color='blue', linewidth=1, alpha=0.7)
plt.plot(time_range, real_best_ask, label='Real Best Ask', color='red', linewidth=1, alpha=0.7)

# Plot generated data
plt.plot(time_range, generated_data_original[:, 0], label='Generated Best Bid', color='cyan', linewidth=1, linestyle='--')
plt.plot(time_range, generated_data_original[:, 1], label='Generated Best Ask', color='magenta', linewidth=1, linestyle='--')

plt.title('Comparison of Real and Generated Best Bid/Ask Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

# Set x-axis to time format
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.gcf().autofmt_xdate()  # Automatically rotate date labels

plt.tight_layout()
plt.show()
