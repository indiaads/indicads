from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import random
import cv2
import torchvision.transforms as T
from transformers import ViTFeatureExtractor

class AdsNonAds(Dataset):
    def __init__(self,
                 images_dir,
                 img_height=224,
                 img_width=224,
                 seed=42,
                 of_num_imgs=20,
                 overfit_test=False,
                 augment_data=False):
        assert os.path.exists(images_dir)

        self.data_dir = images_dir
        self.augment = augment_data

        if overfit_test:
            self.dataset = self.sample_dataset(seed, of_num_imgs)
        else:
            self.dataset = self.train_dataset(seed)

        self.image_transforms = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    def train_dataset(self, seed):
        ads = glob(self.data_dir+'/ads/*')
        non_ads = glob(self.data_dir+'/non-ads/*')

        data = []
        data += [[x, 1] for x in ads]
        data += [[x, 0] for x in non_ads]

        random.seed(seed)
        random.shuffle(data)

        return data

    def sample_dataset(self, seed, num_imgs):
        ads = glob(self.data_dir+'/ads/*')
        non_ads = glob(self.data_dir+'/non-ads/*')

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
        
        # The shape of the image is (height, width, channels)
        # any image is read as BGR image --> converted to RGB
        image = cv2.cvtColor(cv2.imread(
            filename,
            flags=cv2.IMREAD_COLOR),
            code=cv2.COLOR_BGR2RGB)
        transformed_image = self.image_transforms(image, return_tensors="pt")["pixel_values"].squeeze(0) # 3, H, W
        if self.augment:
            # All the transformation expect the image to be in shape of [...., H, W]
            rotated_90 = T.functional.rotate(transformed_image, angle=90)
            rotated_270 = T.functional.rotate(transformed_image, angle=270) 
            flipped = T.RandomHorizontalFlip(p=1)(transformed_image)

            return_list = [transformed_image, rotated_90, rotated_270, flipped]

            return (return_list, [self.dataset[idx][1]]*len(return_list))
        return (transformed_image, self.dataset[idx][1])
