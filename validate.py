import torch
from tqdm import tqdm


def multi_validate(model, val_loader, device):
    pred_num = 0
    label_num = 0
    inter_num = 0
    with torch.no_grad():
        validate_bar = tqdm(val_loader)
        for val_data in validate_bar:
            val_images, val_labels = val_data
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_images_feature, outputs = model(val_images)
            outputs = torch.sigmoid(outputs)
            predict_y = outputs >= 0.5
            predict_y = torch.as_tensor(predict_y, dtype=int)
            pred_num += predict_y.sum().item()
            label_num += val_labels.sum().item()
            inter_num += ((predict_y + val_labels) == 2).sum().item()

    precision = inter_num / pred_num
    print("precision:{}".format(precision))
    recall = inter_num / label_num
    print("recall:{}".format(recall))
    F1_score = (precision * recall / (precision + recall)) * 2
    print("F1 score:{}".format(F1_score))

    return F1_score


def single_validate(model, val_loader, val_num, device):
    accuracy = 0
    with torch.no_grad():
        validate_bar = tqdm(val_loader)
        for val_data in validate_bar:
            val_images, val_labels = val_data
            val_images_feature, val_outputs = model(val_images.to(device))
            val_outputs = torch.sigmoid(val_outputs)
            predict_y = torch.max(val_outputs, dim=1)[1]
            accuracy += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accuracy = accuracy / val_num
        print('val_accuracy: %.5f' % val_accuracy)

    return val_accuracy


def multi_test(num_label, test_dataset, test_loader, model, device):
    pred_num_label = torch.zeros(num_label, ).to(device)
    label_num_label = torch.zeros(num_label, ).to(device)
    inter_num_label = torch.zeros(num_label, ).to(device)

    precision_num = 0
    recall_num = 0
    F1_num = 0
    F2_num = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for test_data in test_bar:
            test_images, test_labels = test_data
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            image_feature, outputs = model(test_images)
            outputs = torch.sigmoid(outputs)

            predict_label = outputs > 0.5
            predict_label = torch.as_tensor(predict_label, dtype=int)

            pred_num_example = torch.sum(predict_label, dim=1)
            label_num_example = torch.sum(test_labels, dim=1)
            inter_num_example = torch.sum((predict_label + test_labels) == 2, dim=1)

            pred_num_label += torch.sum(predict_label, dim=0)
            label_num_label += torch.sum(test_labels, dim=0)
            inter_num_label += torch.sum((predict_label + test_labels) == 2, dim=0)

            cur_precision = torch.nan_to_num(inter_num_example / pred_num_example).sum().item()
            cur_recall = torch.nan_to_num(inter_num_example / label_num_example).sum().item()

            precision_num += cur_precision
            recall_num += cur_recall
            if cur_precision + cur_recall != 0:
                F1_num += (2 * cur_precision * cur_recall) / (cur_precision + cur_recall)
                F2_num += (5 * cur_precision * cur_recall) / (4 * cur_precision + cur_recall)

        precision_example = precision_num / len(test_dataset)
        print("mean precision_example:{}".format(precision_example))
        recall_example = recall_num / len(test_dataset)
        print("mean recall_example:{}".format(recall_example))

        F1_score = F1_num / len(test_dataset)
        print("mean F1:{}".format(F1_score))
        F2_score = F2_num / len(test_dataset)
        print("mean F2:{}".format(F2_score))

        precision_label = torch.nan_to_num(inter_num_label / pred_num_label).sum().item() / num_label
        print("mean precision_label:{}".format(precision_label))
        recall_label = torch.nan_to_num(inter_num_label / label_num_label).sum().item() / num_label
        print("mean recall_lable:{}".format(recall_label))


def single_test(num_label, test_dataset, test_loader, model, device):
    pred_num_label = torch.zeros(num_label, ).to(device)
    label_num_label = torch.zeros(num_label, ).to(device)
    inter_num_label = torch.zeros(num_label, ).to(device)

    precision_num = 0
    recall_num = 0
    F1_num = 0
    F2_num = 0

    acc = 0.0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for test_data in test_bar:
            test_images, test_labels = test_data
            test_images_feature, outputs = model(test_images.to(device))
            outputs = torch.sigmoid(outputs)

            predict_y = torch.max(outputs, dim=1)[1]

            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

            test_labels = torch.zeros(len(test_labels), 30, dtype=torch.int64).scatter_(1, test_labels.unsqueeze(
                dim=1), 1).to(device)

            predict_label = outputs > 0.5
            predict_label = torch.as_tensor(predict_label, dtype=int)

            pred_num_example = torch.sum(predict_label, dim=1)
            label_num_example = torch.sum(test_labels, dim=1)
            inter_num_example = torch.sum((predict_label + test_labels) == 2, dim=1)

            pred_num_label += torch.sum(predict_label, dim=0)
            label_num_label += torch.sum(test_labels, dim=0)
            inter_num_label += torch.sum((predict_label + test_labels) == 2, dim=0)

            cur_precision = torch.nan_to_num(inter_num_example / pred_num_example).sum().item()
            cur_recall = torch.nan_to_num(inter_num_example / label_num_example).sum().item()

            precision_num += cur_precision
            recall_num += cur_recall
            if cur_precision + cur_recall != 0:
                F1_num += (2 * cur_precision * cur_recall) / (cur_precision + cur_recall)
                F2_num += (5 * cur_precision * cur_recall) / (4 * cur_precision + cur_recall)

        F1_score = F1_num / len(test_dataset)
        print("mean F1:{:.5f}".format(F1_score))
        # 计算F2分数
        F2_score = F2_num / len(test_dataset)
        print("mean F2:{:.5f}".format(F2_score))

        precision_example = precision_num / len(test_dataset)
        print("mean precision_example:{:.5f}".format(precision_example))
        recall_example = recall_num / len(test_dataset)
        print("mean recall_example:{:.5f}".format(recall_example))

        precision_label = torch.nan_to_num(inter_num_label / pred_num_label).sum().item() / num_label
        print("mean precision_label:{:.5f}".format(precision_label))
        recall_label = torch.nan_to_num(inter_num_label / label_num_label).sum().item() / num_label
        print("mean recall_lable:{:.5f}".format(recall_label))

    test_accurate = acc / len(test_dataset)
    print('test_accuracy: %.5f' % test_accurate)