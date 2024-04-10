"""
  test data loaders
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

def get_data_loader(params, location, distributed, train=True):
    dataset = TestDataSet(params, location)
    # define a sampler for distributed training using DDP
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
    if train:
        batch_size = params.local_batch_size
    else:
        batch_size = params.local_valid_batch_size
    dataloader = DataLoader(dataset,
                            batch_size=int(batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=(sampler is None),
                            sampler=sampler,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available())
    return dataloader, sampler


class TestDataSet(Dataset):
    def __init__(self, params, location):
        self.params = params
        self.location = location
        self.n_samples = 128

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        ''' just return random tensors '''
        X = torch.rand((1,128,128))
        y = torch.rand((1,128,128))
        return X, y


