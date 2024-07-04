import torch
import torch.nn.functional as F


def Euclidean_Distance(feature):
    y_pred = torch.zeros(len(feature), len(feature) - 1).cuda()
    for index in range(len(feature)):
        if index == 0:
            feature_del = feature[1:].squeeze()
            feature_distance = F.pairwise_distance(feature_del, feature[index, :].view((1, feature.size()[1])), p=2)
            y_pred[index] = feature_distance.view((1, len(feature_del)))
        else:
            feature_del = torch.cat((feature[0:index], feature[index + 1:]), dim=0).squeeze()
            feature_distance = F.pairwise_distance(feature_del, feature[index, :].view((1, feature.size()[1])), p=2)
            y_pred[index] = feature_distance.view((1, len(feature_del)))

    return y_pred
