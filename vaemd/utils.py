import numpy as np
import torch
from torch.utils.data import Dataset

def data_normalization(data_array):
    maximums = np.max(data_array, axis=0)
    minimums = np.min(data_array, axis=0)
    return (data_array - minimums) / (maximums - minimums), maximums, minimums

def data_denormalization(normalized_data, max_values, min_values):
    return normalized_data * (max_values - min_values) + min_values

class TimeLaggedDataset(Dataset):
    def __init__(self, data, time_lag):
        self.data = data
        self.time_lag = time_lag

    def __len__(self):
        return len(self.data) - self.time_lag

    def __getitem__(self, idx):
        x_input = self.data[idx]
        x_output = self.data[idx + self.time_lag]
        return torch.tensor(x_input, dtype=torch.float32), torch.tensor(x_output, dtype=torch.float32)