import math
import numpy as np
import pandas as pd


class DataLoader:
    """A class for loading and transforming data for the LSTM model."""

    def __init__(self, filename, split, cols):
        """
        Initialize the DataLoader with a CSV file and configuration.
        Args:
            filename (str): Path to the CSV file.
            split (float): Fraction of data to use for training.
            cols (list): List of column names to extract.
        """
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)

        # Extract specified columns as numpy arrays
        self.data_train = dataframe[cols].values[:i_split]
        self.data_test = dataframe[cols].values[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise=True):
        """
        Create x, y test data windows.
        Args:
            seq_len (int): Sequence length for LSTM input.
            normalise (bool): Whether to normalize the data.
        Returns:
            tuple: (x, y) numpy arrays for test data.
        """
        data_windows = [
            self.data_test[i:i + seq_len] for i in range(self.len_test - seq_len)
        ]

        data_windows = np.array(data_windows, dtype=np.float64)
        if normalise:
            data_windows = self.normalise_windows(data_windows, single_window=False)

        x = data_windows[:, :-1]
        y = data_windows[:, -1, 0]  # Extract the first column for y
        return x, y

    def get_train_data(self, seq_len, normalise=True):
        """
        Create x, y train data windows.
        Args:
            seq_len (int): Sequence length for LSTM input.
            normalise (bool): Whether to normalize the data.
        Returns:
            tuple: (x, y) numpy arrays for training data.
        """
        data_x, data_y = [], []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)

        return np.array(data_x, dtype=np.float64), np.array(data_y, dtype=np.float64)

    def generate_train_batch(self, seq_len, batch_size, normalise=True):
        """
        Generator to yield training data batches.
        Args:
            seq_len (int): Sequence length for LSTM input.
            batch_size (int): Size of each training batch.
            normalise (bool): Whether to normalize the data.
        Yields:
            tuple: (x_batch, y_batch) numpy arrays for training.
        """
        i = 0
        while i < (self.len_train - seq_len):
            x_batch, y_batch = [], []
            for _ in range(batch_size):
                if i >= (self.len_train - seq_len):
                    yield np.array(x_batch, dtype=np.float64), np.array(y_batch, dtype=np.float64)
                    i = 0  # Reset to start for the next epoch

                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1

            yield np.array(x_batch, dtype=np.float64), np.array(y_batch, dtype=np.float64)

    def _next_window(self, i, seq_len, normalise):
        """
        Generate the next data window from index i.
        Args:
            i (int): Starting index for the window.
            seq_len (int): Sequence length for LSTM input.
            normalise (bool): Whether to normalize the data.
        Returns:
            tuple: (x, y) numpy arrays for the window.
        """
        window = self.data_train[i:i + seq_len]
        if normalise:
            window = self.normalise_windows(window, single_window=True)[0]
        x = window[:-1]
        y = window[-1, 0]  # Extract the first column for y
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        """
        Normalize windows with the first value as the base.
        Args:
            window_data (numpy.ndarray): Input data windows.
            single_window (bool): Whether to normalize a single window or a batch.
        Returns:
            numpy.ndarray: Normalized data.
        """
        normalised_data = []
        window_data = [window_data] if single_window else window_data

        for window in window_data:
            normalised_window = [
                [(float(p) / float(window[0, col_i])) - 1 for p in window[:, col_i]]
                for col_i in range(window.shape[1])
            ]
            normalised_data.append(np.array(normalised_window).T)

        return np.array(normalised_data, dtype=np.float64)
