from typing import Any, Dict, Optional, Tuple
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np

class ContactPointDataModule(LightningDataModule):
    """
    LightningDataModule for ContactPoint dataset.
    ContactPoint dataset sample contains:
        6 * n - dim input for aruco markers' poses
        3 - dim output for contact location
        
    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(self,
                 dataset: Dataset,
                 train_val_test_split_ratio: Tuple[float, float] = (0.5, 0.2), 
                 batch_size: int = 128,
                 num_workers: int = 4,
                 pin_memory: bool = False,
    ) -> None:
        """Initialize a `ContactPointDataModule`.
        
        :param dataset: The dataSet to load. Defaults to `"ContactPoint_Dataset"`.
        :param train_val_test_split_ratio: The train, validation and test split. The third is automatically generated : 1 - 0.5 - 0.2 = 0.3 
        :param batch_size: The batch size. Defaults to `128`.
        :param num_workers: The number of workers. Defaults to `4`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = dataset

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self):
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # pass
        
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:            
            
            dataset = self.dataset
            
            train_length = int(self.hparams.train_val_test_split_ratio[0] * len(dataset))
            val_length = int(self.hparams.train_val_test_split_ratio[1] * len(dataset))
            test_length = int(len(dataset) - train_length - val_length)
            
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths= (train_length, val_length, test_length),
                generator=torch.Generator().manual_seed(42),
            )
            
            
            # train_features, train_labels = next(iter(self.data_train))
            # print(f"Input Feature batch shape: {train_features.size()}")
            # print(f"Output Feature batch shape: {train_labels.size()}")

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass
    
    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
    
    
if __name__ == "__main__":
    pass
    # a = ContactPointDataModule()
    # a.setup()
    
    # for batch in a.train_dataloader():
    #     x,y = batch
    #     print(x.shape, x.dtype)
    #     print(y.shape, y.dtype)
    
    
    