import os
import glob
import numpy as np
import torch
import argparse
import torch.nn as nn
from torchvision import transforms, datasets, utils, models
import torch.optim as optim
from tqdm import tqdm
from multilabel_dataset import Multilabel_Dataset
from loss.feature_ranking_loss import Feature_Ranking_Loss
from loss.sample_ranking_loss import Sample_Ranking_Loss
from loss.label_ranking_loss import Label_Ranking_Loss
from model.swin_transformer import SwinTransformer
from model.resnet import resnet50
from model.vggnet16 import VggNetModel
from Load_Pretrained import load_pretrained
from distance_metric import Euclidean_Distance
from validate import multi_validate, single_validate
from log.logger import loadLogger


def parse_args():
    parser = argparse.ArgumentParser(description='Train of R4C')

    parser.add_argument('--dataset', type=str, default='AID_Multilabel', help='Single/Multi-label Classification dataset',
                        choices=['UCM_Multilabel', 'AID', 'NWPU-RESISC45'])
    parser.add_argument('--train_dir', type=str, default='.../AID_Multilabel/train.txt', help='training set path')
    # UCM_Multilabel: '.../UCM_Multilabel/train.txt'
    # AID: '.../AID/AID(0.2-0.8)/train'
    # NWPU-RESISC45: '.../NWPU-RESISC45/NWPU-RESISC45(0.1-0.9)/train'
    parser.add_argument('--val_dir', type=str, default='.../AID_Multilabel/val.txt', help='validate set path')
    parser.add_argument('--label_path', type=str, default='.../AID_Multilabel/multilabel.csv',
                        help='the path of label file in multi-label dataset')
    parser.add_argument('--label_num', type=str, default=17, help='the number of label')
    # UCM_Multilabel: 17 / AID: 30 / NWPU-RESISC45: 45
    parser.add_argument('--model', type=str, default='Resnet-50', help='the backbone of R4C',
                        choices=['VGGNet-16', 'Swin-T'])
    parser.add_argument('--pretrained_path', type=str,  default="...",
                        help='the pretrained weight for the model')
    parser.add_argument('--save_path', type=str, default='./Resnet50.pth', help='the save path of model')
    parser.add_argument('--max_epoch', type=int, default=400, help='max training epoch')
    parser.add_argument('--train_batch_size', type=int, default=16, help='batch_size of train')
    parser.add_argument('--val_batch_size', type=int, default=64, help='batch_size of validate')
    parser.add_argument('--lr', type=float, default=1 * 1e-5, help='the initial learning rate of model')
    parser.add_argument('--seed', type=int, default=3050, help='Random seed')

    parser.add_argument('--number_sample', type=float, default=16, help='the number of virtual neutral samples')
    parser.add_argument('--number_label', type=float, default=4, help='the number of virtual neutral labels')
    parser.add_argument('--range_sample', type=float, default=0.1,
                        help='the range coefficient of virtual neutral samples')
    parser.add_argument('--range_label', type=float, default=0.1,
                        help='the range coefficient of virtual neutral labels')
    parser.add_argument('--lambdas', type=float, default=1, help='teh weight value in total loss')

    parser.add_argument('--not-save', default=False, action='store_true', help='If true, only output log to terminal.')
    parser.add_argument('--work-dir', default='./log', help='the work folder for storing log results')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger = loadLogger(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if args.dataset in ['AID_Multilabel', 'UCM_Multilabel']:
        Multi_label = True

    if Multi_label:
        train_dataset = Multilabel_Dataset(args.train_dir, args.label_path, True)
        val_dataset = Multilabel_Dataset(args.val_dir, args.label_path, False)
        max_F1_score = 0
    else:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(512),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomRotation(90),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

        val_transform = transforms.Compose(
            [transforms.Resize((512, 512)),
             transforms.CenterCrop(512),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_dataset = datasets.ImageFolder(root=args.train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(root=args.train_dir.replace("train", "val"), transform=val_transform)
        max_accuracy = 0

    train_num = len(train_dataset)
    val_num = len(val_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)

    print("using {} images for training".format(train_num))
    print("using {} images for validate".format(val_num))

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

    feature_ranking_loss = Feature_Ranking_Loss()
    sample_ranking_loss = Sample_Ranking_Loss()
    label_ranking_loss = Label_Ranking_Loss()

    params_to_update = model.parameters()
    optimizer = optim.Adam(params_to_update, lr=args.lr)

    for epoch in range(args.max_epoch):
        logger.info(" Training epoch: {}".format(epoch + 1))
        model.train()
        running_loss = 0.0
        loss_feature_num = 0.0
        loss_label_num = 0.0
        loss_sample_num = 0.0
        train_bar = tqdm(train_loader)
        if epoch + 1 == args.max_epoch * 0.5:
            for params in optimizer.param_groups:
                params['lr'] *= 0.1
        if epoch + 1 == args.max_epoch * 0.75:
            for params in optimizer.param_groups:
                params['lr'] *= 0.5
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            if not Multi_label:
                labels = torch.zeros(len(labels), args.label_num,
                                     dtype=torch.int64).to(device).scatter_(1, labels.unsqueeze(dim=1), 1)

            images_feature, y_pred2 = model(images)
            y_pred2 = torch.sigmoid(y_pred2)

            optimizer.zero_grad()

            y_pred = Euclidean_Distance(images_feature)
            loss_feature = feature_ranking_loss(y_pred.to(device), labels)

            # virtual neutral labels
            virtual_middle_class = torch.tile(torch.linspace(0.5 - args.range_label, 0.5 + args.range_label,
                                                             steps=args.number_label).view((1, args.number_label)),
                                              (len(labels), 1))
            virtual_middle_class_label = torch.tile(torch.linspace(1.0, 1.0,
                                                                   steps=args.number_label).view(
                (1, args.number_label)), (len(labels), 1))
            y_pred2_middle_class = torch.cat((y_pred2, virtual_middle_class.to(device)), dim=1).to(device)
            labels_middle_class = torch.cat((labels * 2, virtual_middle_class_label.to(device)), dim=1).to(device)
            loss_label = label_ranking_loss(y_pred2_middle_class, labels_middle_class)

            # virtual neutral samples
            virtual_middle_image = torch.tile(torch.linspace(0.5 - args.range_sample, 0.5 + args.range_sample,
                                                             steps=args.number_sample).view((args.number_sample, 1)),
                                              (1, args.label_num))
            virtual_middle_image_label = torch.tile(torch.linspace(1.0, 1.0,
                                                                   steps=args.number_sample).view(
                (args.number_sample, 1)), (1, args.label_num))
            y_pred2_middle_sample = torch.cat((y_pred2, virtual_middle_image.to(device)), dim=0).to(device)
            labels_middle_sample = torch.cat((labels * 2, virtual_middle_image_label.to(device)), dim=0).to(device)
            loss_sample = sample_ranking_loss(y_pred2_middle_sample, labels_middle_sample)

            loss = loss_feature + args.lambdas * (loss_label + loss_sample)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_feature_num += loss_feature.item()
            loss_label_num += loss_label.item()
            loss_sample_num += loss_sample.item()

            train_bar.desc = "train epoch[{}/{}] total_loss:{:.5f} loss_feature:{:.5f} " \
                             "loss_label:{:.5f} loss_sample:{:.5f}".format(epoch + 1, args.max_epoch,
                                                                           running_loss / len(train_loader),
                                                                           loss_feature_num / len(train_loader),
                                                                           loss_label_num / len(train_loader),
                                                                           loss_sample_num / len(train_loader))

        logger.info(" train epoch[{}/{}] total_loss:{:.5f} feature_ranking_loss:{:.5f} " \
                    "label_ranking_loss:{:.5f} sample_ranking_loss:{:.5f}".format(epoch + 1, args.max_epoch,
                                                                                  running_loss / len(train_loader),
                                                                                  loss_feature_num / len(train_loader),
                                                                                  loss_label_num / len(train_loader),
                                                                                  loss_sample_num / len(train_loader)))

        model.eval()
        logger.info(" Eval epoch: {}".format(epoch + 1))
        if Multi_label:
            F1_score = multi_validate(model, val_loader, device)
            logger.info(" F1_score: {}".format(F1_score))
            if F1_score >= max_F1_score:
                max_F1_score = F1_score
                torch.save(model.state_dict(), args.save_path)
        else:
            val_accuracy = single_validate(model, val_loader, val_num, device)
            logger.info(" val_accuracy: {}".format(val_accuracy))
            if val_accuracy >= max_accuracy:
                max_accuracy = val_accuracy
                torch.save(model.state_dict(), args.save_path)

        if epoch == args.max_epoch - 1:
            torch.save(model.state_dict(), './last.pth')

    print('Finished Training')


if __name__ == '__main__':
    main()
