import os
import math
import numpy as np
import datetime as dt
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils2 import Timer  # Ensure the correct path is used here


class Model:
    """A class for building and inferring an LSTM model"""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print(f'[Model] Loading model from file {filepath}')
        self.model = load_model(filepath)

    from tensorflow.keras.layers import Input

    def build_model(self, configs):
        timer = Timer()
        timer.start()

        # Add Input layer explicitly
        input_timesteps = configs['model']['layers'][0].get('input_timesteps')
        input_dim = configs['model']['layers'][0].get('input_dim')
        self.model.add(Input(shape=(input_timesteps, input_dim)))

        for layer in configs['model']['layers']:
            neurons = layer.get('neurons')
            dropout_rate = layer.get('rate')
            activation = layer.get('activation')
            return_seq = layer.get('return_seq', False)

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            elif layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, return_sequences=return_seq))
            elif layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(
            loss=configs['model']['loss'],
            optimizer=configs['model']['optimizer']
        )

        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir):
        timer = Timer()
        timer.start()
        print(f'[Model] Training Started: {epochs} epochs, {batch_size} batch size')

        save_fname = os.path.join(save_dir, f"{dt.datetime.now():%d%m%Y-%H%M%S}-e{epochs}.h5")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        self.model.save(save_fname)

        print(f'[Model] Training Completed. Model saved as {save_fname}')
        timer.stop()

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        timer = Timer()
        timer.start()
        print(
            f'[Model] Training Started: {epochs} epochs, {batch_size} batch size, {steps_per_epoch} batches per epoch')

        # Use .keras extension for the filepath
        save_fname = os.path.join(save_dir, f"{dt.datetime.now():%d%m%Y-%H%M%S}-e{epochs}.keras")
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        self.model.fit(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks
        )

        print(f'[Model] Training Completed. Model saved as {save_fname}')
        timer.stop()

    def predict_point_by_point(self, data):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted


    def predict_sequences_multiple(self, data, window_size, prediction_len):
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[np.newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs


    def predict_sequence_full(self, data, window_size):
        # Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[np.newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        return predicted