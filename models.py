import torch.nn as nn

class ConvClsHead(nn.Module):
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

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_prob):
        super().__init__()

        # Dropout layers after activation.
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 100),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(50,num_classes)
        )
    def forward(self, x):
        return self.cls_head(x)

class VitWithCLShead(nn.Module):
    def __init__(self, vit_model, cls_head, is_vit_trainable=False):
        super().__init__()
        self.vit_model = vit_model
        if not is_vit_trainable:
            for param in self.vit_model.parameters():
                param.requires_grad = False
        self.cls_head = cls_head
    def trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]
    def forward(self, x):
        cls_token = self.vit_model(x).pooler_output
        return self.cls_head(cls_token)