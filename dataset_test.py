import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
import cv2
from torch.utils.data import DataLoader

def my_transforms():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, flist, mask_flist, test_mask_index, augment, training, input_size):
        super(Dataset, self).__init__()
        self.dataset_name = dataset_name
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.mask_data = self.load_flist(mask_flist)
        self.mask_selected = np.load(test_mask_index)
        if not self.training:
            self.mask_data = self.mask_data[self.mask_selected]
        self.input_size = input_size

    def __len__(self):
        return len(self.data)
        #return 100

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        img = cv2.imread(self.data[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # for celeba dataset we first perform center crop
        if self.dataset_name == 'celeba':
            img = self.resize(img)

        name = self.data[index]
        name = name.split('/')[-1].split('.')[0]

        img_512 = cv2.resize(img, (512, 512))
        img_256 = cv2.resize(img, (256, 256))

        # load mask
        mask = self.load_mask(img, index)

        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img_512 = img_512[:, ::-1, ...]
            img_256 = img_256[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

        return self.to_tensor(img_512), self.to_tensor(img_256), self.to_tensor(mask), index, name


    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]

        # external
        if self.training:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = cv2.imread(self.mask_data[mask_index])
            mask = cv2.resize(mask, (256,256),interpolation=cv2.INTER_NEAREST)
            # mask = self.resize(mask, imgh, imgw)
        else:   # in test mode, there's a one-to-one relationship between mask and image; masks are loaded non random
            # mask = 255 - imread(self.mask_data[index])[:,:,0]    # ICME original (H,W,3) mask: 0 for hole
            mask = cv2.imread(self.mask_data[index])  # mask must be 255 for hole in this InpaintingModel
            mask = cv2.resize(mask, (256,256),interpolation=cv2.INTER_NEAREST)
        # mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
        return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                # print(np.genfromtxt(flist, dtype=np.str))
                # return np.genfromtxt(flist, dtype=np.str)
                try:
                    return np.genfromtxt(flist, dtype=np.str)
                except:
                    return [flist]
        return []


def build_dataloader(dataset_name, flist, mask_flist, test_mask_index, augment, training, input_size, batch_size, \
num_workers, shuffle):

    dataset = Dataset(
        dataset_name=dataset_name,
        flist=flist,
        mask_flist=mask_flist,
        test_mask_index=test_mask_index,
        augment=augment,
        training=training,
        input_size=input_size
        )

    print('Total instance number:', dataset.__len__())

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=shuffle
    )

    return dataloader
