__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import math
import matplotlib.pyplot as plt
from Run2.data_processor2 import DataLoader  # Ensure correct relative import
from Run2.model2 import Model  # Ensure correct relative import


def plot_results(predicted_data, true_data):
    """Plot true vs predicted data."""
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    """Plot multiple predicted sequences vs true data."""
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to its correct start
    for i, data in enumerate(predicted_data):
        padding = [None] * (i * prediction_len)
        plt.plot(padding + data, label=f'Prediction {i+1}')
        plt.legend()
    plt.show()


def main():
    # Load configuration file
    with open('../config.json', 'r') as config_file:
        configs = json.load(config_file)

    # Ensure the save directory exists
    os.makedirs(configs['model']['save_dir'], exist_ok=True)

    # Load data
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    # Initialize and build the model
    model = Model()
    model.build_model(configs)

    # Prepare training data
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # Out-of-memory generative training
    steps_per_epoch = math.ceil(
        (data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size']
    )
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir']
    )

    # Prepare test data
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # Generate predictions
    predictions = model.predict_sequences_multiple(
        x_test,
        configs['data']['sequence_length'],
        configs['data']['sequence_length']
    )

    # Plot results
    plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])


if __name__ == '__main__':
    main()
