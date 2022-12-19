from glob import glob
from torch.utils.data import  Dataset
import torchvision.transforms as T
import os
import random
from matplotlib.pyplot import imread


class AdsNonAds(Dataset):
    def __init__(self,
                 images_dir,
                 img_height=224, 
                 img_width=224,
                 seed=42,
                 of_num_imgs=20,
                 overfit_test=False):
        assert os.path.exists(images_dir)

        self.data_dir = images_dir

        if overfit_test:
            self.dataset = self.sample_dataset(seed, of_num_imgs)
        else:
            self.dataset = self.train_dataset(seed)

        self.image_transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((img_height, img_width), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def train_dataset(self, seed):
        ads = glob(self.data_dir+'/ads/*.jpeg')
        non_ads = glob(self.data_dir+'/non-ads/*.jpeg')

        data = []
        data += [[x, 1] for x in ads]
        data += [[x, 0] for x in non_ads]

        random.seed(seed)
        random.shuffle(data)

        return data

    def sample_dataset(self, seed, num_imgs):
        ads = glob(self.data_dir+'/ads/*.jpeg')
        non_ads = glob(self.data_dir+'/non-ads/*.jpeg')

        random.seed(seed)

        ads = random.sample(ads, num_imgs)
        non_ads = random.sample(non_ads, num_imgs)

        data = []
        data += [[x, 1] for x in ads]
        data += [[x, 0] for x in non_ads]

        random.shuffle(data)
        return data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        filename = self.dataset[idx][0]
        image = imread(filename)

        return (self.image_transforms(image), self.dataset[idx][1])
