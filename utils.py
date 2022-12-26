import os
import torch
import yaml

def save_checkpoint(state, is_best, file_folder, experiment,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""

    file_folder = os.path.join(file_folder, experiment)
    os.makedirs(file_folder, exist_ok=True)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def load_config(config_file):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    return config
