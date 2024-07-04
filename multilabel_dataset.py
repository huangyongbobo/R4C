import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def Multilabel_Dataset(image_txt_path, label_csv_path, train):
    """
    Args:
        image_txt_path (str): The path of Multilabel Dataset.
        label_csv_path (str): The path of label_csv.
        train (bool): If True, returns train dataset class.

    Returns:
        dataset: An instance of "BaseDataset" class, containing the loaded images and label.
    """
    image_df = pd.read_csv(image_txt_path, header=None)
    image_txt_list = image_df.values.tolist()
    label_list = pd.read_csv(label_csv_path)
    label_list.set_index(["IMAGE\LABEL"], inplace=True)

    if train:
        transform = transforms.Compose([transforms.Resize((512, 512)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation(90),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    else:
        transform = transforms.Compose([transforms.Resize((512, 512)),
                                        transforms.CenterCrop(512),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = BaseDataset(image_txt_list, label_list, transform, train)

    return dataset


class BaseDataset(Dataset):
    def __init__(self, image_txt_list, label_list, transform, train):
        self.image_txt_list = image_txt_list
        self.label_list = label_list
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.image_txt_list)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path, current_class = self.image_txt_list[index][0].split(" ")
        img = self.transform((Image.open(img_path)))

        file_path, current_image_name = os.path.split(img_path)
        current_image_name, _ = current_image_name.split('.')
        label_array = self.label_list.loc[current_image_name].array
        label_numpy = np.array(label_array)
        return img, label_numpy
