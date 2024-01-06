from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

PROJECT_ROOT_DIR = rootutils.find_root(search_from=__file__, indicator=".project-root")
PROJECT_DATA_DIR = str(PROJECT_ROOT_DIR) + '/data/'


class ContactPoint_Dataset(Dataset):
    """cylinder prism fem dataset."""

    def __init__(self,
                 data_str: str = 'bc_3_trainingData.npy',
                 transform=None):
        """
        Args:
            data_path (string): Path to data samples.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        # self.save_hyperparameters(logger=False)
        # data = np.load(self.hparams.data_path)
        data_path = PROJECT_DATA_DIR + data_str
        data = np.load(data_path).astype('float32')
    
        self.input = data[:, 0 : -3]
        self.output = data[:, -3 : len(data[:,])]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input_tensor = torch.from_numpy(self.input[idx])
        output_tensor = torch.from_numpy(self.output[idx])
        
               
        return input_tensor,output_tensor
        
        
if __name__ == "__main__":
    
    data1 = ContactPoint_Dataset()
    
    train_dataloader = DataLoader(data1, batch_size=64, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    
    print(train_features.dtype)
    print(train_labels.dtype)
    print(f"Input Feature batch shape: {train_features.size()}")
    print(f"Output Feature batch shape: {train_labels.size()}")
    



