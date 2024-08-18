import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
import pandas as pd

def get_preqnt_datasets(data_file: str, response_column: str, train_ratio: float = 0.8):
    df = pd.read_csv(data_file)

    if response_column not in df.columns:
        raise ValueError(f"Response column '{response_column}' not found in data.")
    
    features = df.drop(columns=[response_column])
    response = df[response_column]

    features_np = features.values.astype(np.float32)
    response_np = response.values.astype(np.float32)

    features_tensor = torch.from_numpy(features_np)
    response_tensor = torch.from_numpy(response_np)

    full_dataset = TensorDataset(features_tensor, response_tensor)

    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset