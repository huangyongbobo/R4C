import os
import numpy as np
import torch
import argparse
import torch.nn as nn
from torchvision import transforms, datasets, utils, models
from tqdm import tqdm
from multilabel_dataset import Multilabel_Dataset
from model.swin_transformer import SwinTransformer
from model.resnet import resnet50
from model.vggnet16 import VggNetModel
from Load_Pretrained import load_pretrained
from validate import multi_test, single_test


def parse_args():
    parser = argparse.ArgumentParser(description='Test of R4C')

    parser.add_argument('--dataset', type=str, default='AID_Multilabel', help='Single/Multi-label Classification dataset',
                        choices=['UCM_Multilabel', 'AID', 'NWPU-RESISC45'])
    parser.add_argument('--test_dir', type=str, default='.../AID_Multilabel/test.txt', help='test set path')
    # UCM_Multilabel: '.../UCM_Multilabel/test.txt'
    # AID: '.../AID/AID(0.2-0.8)/test'
    # NWPU-RESISC45: '.../NWPU-RESISC45/NWPU-RESISC45(0.1-0.9)/test'
    parser.add_argument('--label_path', type=str, default='.../AID_Multilabel/multilabel.csv',
                        help='the path of label file in multi-label dataset')
    parser.add_argument('--label_num', type=str, default=17, help='the number of label')
    # UCM_Multilabel: 17 / AID: 30 / NWPU-RESISC45: 45
    parser.add_argument('--test_batch_size', type=int, default=64, help='batch_size of test')
    parser.add_argument('--model', type=str, default='Resnet-50', help='the backbone of R4C',
                        choices=['VGGNet-16', 'Swin-T'])
    parser.add_argument('--pretrained_path', type=str, default="...",
                        help='the pretrained weight for the model')
    parser.add_argument('--weights_path', type=str, default='./Resnet50.pth', help='the path of weight')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    if args.dataset in ['AID_Multilabel', 'UCM_Multilabel']:
        Multi_label = True

    if Multi_label:
        test_dataset = Multilabel_Dataset(args.test_dir, args.label_path, False)

    else:
        test_transform = transforms.Compose(
            [transforms.Resize((512, 512)),
             transforms.CenterCrop(512),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_dataset = datasets.ImageFolder(root=args.test_dirtest_dir, transform=test_transform)

    test_num = len(test_dataset)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    print("using {} images for test".format(test_num))

    if args.model == "Resnet-50":
        model = resnet50(num_classes=17)
        print('Using ResNet50!')
        load_pretrained(model, args.pretrained_path, 'resnet')
    elif args.model == "Swin-T":
        model = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=17, embed_dim=96,
                                depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
                                mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0.0, drop_path_rate=0.1,
                                ape=False, patch_norm=True, use_checkpoint=False)
        load_pretrained(model, args.pretrained_path, 'swin')
    elif args.model == "VGGNet-16":
        vggNet16 = models.vgg16_bn(pretrained=True)
        model = VggNetModel(vggNet16)

    else:
        raise NotImplementedError(f"Unkown model: {args.model}")

    model.to(device)

    assert os.path.exists(args.weights_path), "file: '{}' dose not exist.".format(args.weights_path)
    model.load_state_dict(torch.load(args.weights_path))

    model.eval()

    if Multi_label:
        multi_test(args.label_num, test_dataset, test_loader, model, device)
    else:
        single_test(args.label_num, test_dataset, test_loader, model, device)


if __name__ == '__main__':
    main()
