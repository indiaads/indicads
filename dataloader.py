from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
import random


def collate_function(batch):
    inputs = torch.stack([x[0] for x in batch])
    labels = torch.tensor([x[1] for x in batch])

    # (batch_size, channels, h, w),  (batch_size)
    return (inputs, labels)


def make_train_val_loaders(dataset, split, train_bs, valid_bs):
    """
        This function returns train_dataloader and validation_dataloader 
        based on the split specified.

        dataset: dataset object
        split: A tuple, or a list with float values summing to 1
        train_bs: batch size of train dataloader
        valid_bs: batch size of validation dataloader
    """
    assert isinstance(split, tuple) or isinstance(split, list)
    assert sum(split) == 1

    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)

    split_index = int(split[0]*len(dataset))
    train_subsampler = SubsetRandomSampler(indices[0:split_index])
    valid_subsampler = SubsetRandomSampler(indices[split_index:])

    return (make_data_loader(dataset=dataset,
                             batch_size=train_bs,
                             num_workers=4,
                             sampler=train_subsampler),
            make_data_loader(dataset=dataset,
                             batch_size=valid_bs,
                             num_workers=2,
                             sampler=valid_subsampler)
            )


def make_data_loader(dataset,
                     batch_size,
                     num_workers,
                     sampler=None):

    return DataLoader(dataset,
                      batch_size=batch_size,
                      collate_fn=collate_function,
                      num_workers=num_workers,
                      persistent_workers=True,
                      sampler=sampler)
