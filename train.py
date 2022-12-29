from transformers import ViTModel
from datasets import *
from dataloader import *
from tqdm import tqdm
import torch.nn as nn
from models import *
import numpy as np
import wandb
from utils import *
import argparse
from sklearn.metrics import f1_score, accuracy_score
from torchsummary import summary
from optim_lrsched import *
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str,
                    help="override the name of the experiment")
arguments = parser.parse_args()

config = load_config('config.yaml')

print("CONFIGURATION FOR THIS RUN \n")
pprint(config)

# Override the exp_name in the configuration file
if arguments.exp is not None:
    config['wandb']['exp_name'] = arguments.exp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wandb_log = config['wandb']['wandb_log']
if wandb_log:
    wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'],
               name=config['wandb']['exp_name'])

train_dataset = AdsNonAds(images_dir=config['data']['train_dir'],
                          img_height=config['data']['reshape_height'],
                          img_width=config['data']['reshape_width'],
                          seed=42,
                          of_num_imgs=None,
                          overfit_test=False,
                          augment_data=config['data']['augment'])

train_dataloader = make_data_loader(dataset=train_dataset,
                                    batch_size=config['model']['train_bs'],
                                    num_workers=4,
                                    sampler=None,
                                    data_augment=config['data']['augment'])


valid_dataset = AdsNonAds(images_dir=config['data']['val_dir'],
                          img_height=config['data']['reshape_height'],
                          img_width=config['data']['reshape_width'],
                          seed=42,
                          of_num_imgs=None,
                          overfit_test=False,
                          augment_data=False)

valid_dataloader = make_data_loader(dataset=valid_dataset,
                                    batch_size=config['model']['valid_bs'],
                                    num_workers=4,
                                    sampler=None,
                                    data_augment=False)


pretrained_model = ViTModel.from_pretrained(config['model']['vit_pretrained'])

cls_head = ClassificationHead(
    input_dim=config['model']['vit_feature_dim'], num_classes=2)

vit_model = VitWithCLShead(pretrained_model, cls_head,
                           is_vit_trainable=config['model']['is_vit_trainable'])

summary(vit_model, (3, 224, 224))

vit_model.to(device)

criterion = nn.CrossEntropyLoss()


if config['model']['is_vit_trainable']:
    vit_params = vit_model.vit_model.parameters()
    cls_params = vit_model.cls_head.parameters()

    param_groups = [vit_params, cls_params]
    param_grp_lr = [config["lr"]["vit_lr"], config["lr"]["cls_lr_start"]]
    has_sched = [False, True]

else:
    param_groups = [vit_model.cls_head.parameters()]
    param_grp_lr = [config["lr"]["cls_lr_start"]]
    has_sched = [True]

optimizers, schedulers = optims_and_scheds(
    param_groups=param_groups, param_lr=param_grp_lr,
    has_sched=has_sched, config=config)




previous_valid_loss = 1e10

epochs = config['model']['epochs']


for epoch in range(epochs):
    train_loss = []

    vit_model.train()

    gt_labels = []
    pred_labels = []

    print("EPOCH:{0}".format(epoch))
    for batch in tqdm(train_dataloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        for optimizer in optimizers:
            optimizer.zero_grad()

        output = vit_model(inputs)
        loss = criterion(output, labels)
        loss.backward()

        for optimizer in optimizers:
            optimizer.step()
        
        gt_labels.extend(labels.tolist())
        pred_labels.extend(output.argmax(dim=1).tolist())

        train_loss.append(loss.item())
    
    # Stepping learning rate after each epoch according to the given stepsize
    for scheduler in schedulers:
        scheduler.step()

    # Train_loss and train accuracy
    train_loss = np.mean(train_loss)
    train_acc = accuracy_score(y_pred=pred_labels, y_true=gt_labels)

    vit_model.eval()

    gt_labels = []
    pred_labels = []
    valid_loss = []

    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            output = vit_model(inputs)
            loss = criterion(output, labels)

            valid_loss.append(loss.item())

            gt_labels.extend(labels.tolist())
            pred_labels.extend(output.argmax(dim=1).tolist())

    # Validation loss, f1 score and accuracy
    valid_loss = np.mean(valid_loss)
    valid_f1_score = f1_score(y_true=gt_labels, y_pred=pred_labels)
    valid_accuracy = accuracy_score(y_true=gt_labels, y_pred=pred_labels)

    if wandb_log:
        wandb.log({'loss/train': train_loss, 'loss/val': valid_loss,
                  'f1/val': valid_f1_score, 'acc/train': train_acc,
                   'acc/val': valid_accuracy})

    # Preparing the state dictionary
    save_states = {
            'epoch': epoch,
            'model_state_dict': vit_model.state_dict(),
            'optimizers': [opt.state_dict() for opt in optimizers],
            'schedulers': [sch.state_dict() for sch in schedulers]
        }
    # Logging the checkpoints when the validation loss is better than the previous one
    if valid_loss < previous_valid_loss:
        previous_valid_loss = valid_loss
        save_checkpoint(state=save_states,
                        is_best=True,
                        file_folder=config['ckpt']['ckpt_folder'],
                        experiment=config['wandb']['exp_name'],
                        file_name='epoch_{:03d}.pth.tar'.format(epoch)
                        )

    # Logging the checkpoints at regular checkpoint frequency
    if epoch % config['ckpt']['ckpt_frequency'] == 0:
        save_checkpoint(state=save_states, is_best=False,
                        file_folder=config['ckpt']['ckpt_folder'],
                        experiment=config['wandb']['exp_name'],
                        file_name='epoch_{:03d}.pth.tar'.format(epoch)
                        )
if wandb_log:
    wandb.finish()
