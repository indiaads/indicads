import torch.nn as nn

class SimpleClsHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cls_head = nn.Sequential(
             nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=2),
             nn.MaxPool2d(kernel_size=3),
             nn.LeakyReLU(),
             nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5, stride=(2,3)),
             nn.MaxPool2d(kernel_size=(3,4)),
             nn.LeakyReLU(),
             nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4),
             nn.MaxPool2d(kernel_size=(1,3)),
             nn.LeakyReLU(),
             nn.Flatten(),
             nn.Linear(16,num_classes)
             )
    def forward(self, x):
        return self.cls_head(x)

class VitWithCLShead(nn.Module):
    def __init__(self, vit_model, cls_head):
        super().__init__()
        self.vit_model = vit_model
        self.cls_head = cls_head
    def forward(self, x):
        hidden_state = self.vit_model(x).last_hidden_state
        # (batch, x, y) to (batch, channels=1, x, y) for conv2d in cls_head
        hidden_state.unsqueeze_(1)
        return self.cls_head(hidden_state)