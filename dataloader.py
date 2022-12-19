from torch.utils.data import DataLoader
import torch


def collate_function(batch):
    inputs = torch.stack([x[0] for x in batch])
    labels = torch.tensor([x[1] for x in batch])

    return (inputs, labels)         # (batch_size, channels, h, w),  (batch_size)


def make_data_loader(dataset,
                     batch_size,
                     num_workers):

    return DataLoader(dataset,
                      batch_size=batch_size,
                      collate_fn=collate_function,
                      num_workers=num_workers,
                      persistent_workers=True)
