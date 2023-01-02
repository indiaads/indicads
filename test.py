from argparse import ArgumentParser
from dataloader import *
from datasets import *
from models import *
from transformers import ViTModel
import torch
from tqdm import tqdm
import pickle
from utils import *
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = load_config('config.yaml')

argument_parser = ArgumentParser()

argument_parser.add_argument(
    '--ckpt_path', type=str, help="Path of the checkpoint", required=True)

arguments = argument_parser.parse_args()

test_dataset = AdsNonAds(images_dir=config['data']['test_dir'],
                          img_height=config['data']['reshape_height'],
                          img_width=config['data']['reshape_width'],
                          seed=42,
                          of_num_imgs=None,
                          overfit_test=False,
                          augment_data=False)

test_dataloader = make_data_loader(dataset=test_dataset,
                                    batch_size=config['model']['batch_size'],
                                    num_workers=4,
                                    sampler=None,
                                    data_augment=False)


pretrained_model = ViTModel.from_pretrained(config['model']['vit_pretrained'])

cls_head = ClassificationHead(
    input_dim=config['model']['vit_feature_dim'], num_classes=2, 
    dropout_prob=config['model']['cls_head_dropout_p'])

model = VitWithCLShead(pretrained_model, cls_head,
                           is_vit_trainable=config['model']['is_vit_trainable'])


ckpt = torch.load(arguments.ckpt_path)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()

gt_labels = []
pred_labels = []
test_loss = []

criterion = nn.CrossEntropyLoss()


for batch in tqdm(test_dataloader):
    inputs, labels = batch
    inputs, labels = inputs.to(device), labels.to(device)

    output = model(inputs)
    loss = criterion(output, labels)

    test_loss.append(loss.item())

    gt_labels.extend(labels.tolist())
    pred_labels.extend(output.argmax(dim=1).tolist())

# Validation loss, f1 score and accuracy
test_loss = np.mean(test_loss)

test_accuracy = accuracy_score(y_true=gt_labels, y_pred=pred_labels)

print("LOSS: {0}, ACC: {1}".format(test_loss, test_accuracy))
