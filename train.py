from transformers import ViTModel
from datasets import *
from dataloader import *
from tqdm import tqdm
import torch.nn as nn
from models import *
import torch.optim as optim
import numpy as np
import wandb
from utils import *


config = load_config('config.yaml')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wandb_log = config['wandb']['wandb_log']
if wandb_log:
    wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'],
               name=config['wandb']['exp_name'])

dataset = AdsNonAds(images_dir=config['data']['data_dir'],
                    img_height=config['data']['reshape_height'],
                    img_width=config['data']['reshape_width'],
                    seed=42,
                    of_num_imgs=config['of_test']['num_imgs_per_cls'],
                    overfit_test=config['of_test']['do_overfit_test'])

dataloader = make_data_loader(dataset=dataset,
                              batch_size=config['of_test']['batch_size'] if config['of_test']['do_overfit_test'] else config['model']['batch_size'] ,
                              num_workers=2)


pretrained_model = ViTModel.from_pretrained(config['model']['vit_pretrained'])

cls_head = SimpleClsHead(num_classes=2)

vit_model = VitWithCLShead(pretrained_model, cls_head)
vit_model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(vit_model.parameters(),
                      lr=config['model']['learning_rate'], momentum=0.9)


previous_epoch_loss = 1e10

epochs = config['of_test']['epochs'] if config['of_test']['do_overfit_test'] else config['model']['epochs']

# Putting the model in train mode.
vit_model.train()

for epoch in range(epochs):
    epoch_loss = []
    print("EPOCH:{0}".format(epoch))

    for batch in tqdm(dataloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # If we want only the features from the VitModel, without training it.
        # with torch.no_grad():
        #     hidden_state = vit_model(inputs).last_hidden_state

        optimizer.zero_grad()

        output = vit_model(inputs)
        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()
        epoch_loss.append(loss.item())

    epoch_loss = np.mean(epoch_loss)

    if wandb_log:
        wandb.log({'train_loss': epoch_loss})

    if not config['of_test']['do_overfit_test']:
        if epoch % config['ckpts']['ckpt_frequency'] == 0:
            save_states = {
                'epoch': epoch,
                'model_state_dict': cls_head.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_checkpoint(state=save_states, is_best=False,
                            file_folder=config['ckpt']['ckpt_folder'],
                            file_name='epoch_{:03d}.pth.tar'.format(epoch)
                            )
            if epoch_loss < previous_epoch_loss:
                previous_epoch_loss = epoch_loss
                save_checkpoint(state=save_states,
                                is_best=True,
                                file_folder=config['ckpt']['ckpt_folder'], 
                                file_name='epoch_{:03d}.pth.tar'.format(epoch)
                                )

if wandb_log:
    wandb.finish()