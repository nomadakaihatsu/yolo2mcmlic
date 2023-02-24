import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable

'''
language: python, UTF-8
comment language: English
'''

def contains_common_element(list1, list2):
    # list1, list2: check if these lists have common elements
    return len(set(list1) & set(list2)) > 0

def pred_acc(original, predicted, threshold):
    # original:label, predicted:prediction, threshold:threshold
    # apply threshold to prediction and compare with label
    # return accuracy
    prob = torch.sigmoid(predicted)
    pred_classes = (prob >= threshold).long()
    return (pred_classes == original).sum().numpy() / len(original)

def pred_recall(original, predicted, threshold):
    # original:label, predicted:prediction, threshold:threshold
    # apply threshold to prediction and compare with label
    # return recall
    prob = torch.sigmoid(predicted)
    pred_classes = (prob >= threshold).long()
    true_positive = ((pred_classes == original) & (original == 1)).sum().numpy()
    recall = true_positive / (original.sum().numpy() + 1e-10)
    return recall

def pred_precision(original, predicted, threshold):
    # original:label, predicted:prediction, threshold:threshold
    # apply threshold to prediction and compare with label
    # return precision
    prob = torch.sigmoid(predicted)
    pred_classes = (prob >= threshold).long()
    true_positive = ((pred_classes == 1) & (original == 1)).sum().numpy()
    false_positive = ((pred_classes == 1) & (original == 0)).sum().numpy()
    precision = true_positive / (true_positive + false_positive + 1e-10)
    return precision
def pred_acc_each_label(original, predicted, threshold):
    # original:label, predicted:prediction, threshold:threshold
    # apply threshold to prediction and compare with label
    # return accuracy for each label
    prob = torch.sigmoid(predicted)
    pred_classes = (prob >= threshold).long()
    return (pred_classes == original).long().numpy()

def pred_recall_each_label(original, predicted, threshold):
    # original:label, predicted:prediction, threshold:threshold
    # apply threshold to prediction and compare with label
    # return recall for each label
    # if prediction is 0, recall is not defined, so return np.nan
    prob = torch.sigmoid(predicted)
    pred_classes = (prob >= threshold).long()
    mask = (original == 1)
    recall_each = torch.zeros_like(original, dtype=torch.float32)
    recall_each[mask] = ((pred_classes[mask] == 1) & (original[mask] == 1)).float()
    recall_each[~mask] = np.nan

    return recall_each

def pred_precision_each_label(original, predicted, threshold):
    # original:label, predicted:prediction, threshold:threshold
    # apply threshold to prediction and compare with label
    # return precision for each label
    # if prediction is 0, precision is not defined, so return np.nan
    prob = torch.sigmoid(predicted)
    pred_classes = (prob >= threshold).long()
    mask = (pred_classes == 1)
    precision_each = torch.zeros_like(pred_classes, dtype=torch.float32)
    precision_each[mask] = ((pred_classes[mask] == 1) & (original[mask] == 1)).float()
    precision_each[~mask] = np.nan

    return precision_each

def merge_lists(list1, list2):
    # list1, list2: merge these lists
    list = []
    for i in range(len(list1)):
        list.append(list2[i])
        list.append(list1[i])
    return list

def fit_model(model, dataloader,criterion,device_name, optimizer,threshold,phase='training'):
    # model: model to be trained
    # dataloader: dataloader
    # criterion: loss function
    # device_name: device name
    # optimizer: optimizer
    # threshold: threshold for prediction
    # phase: training or validation
    # return loss, accuracy, recall, precision, each accuracy, each recall, each precision
    if phase == 'training':
        model.train()

    if phase == 'validation':
        model.eval()

    running_loss = []
    running_acc = []
    running_recall = []
    running_precision = []
    running_each_acc = []
    running_each_recall = []
    running_each_precision = []
    for i, data in enumerate(tqdm(dataloader)):
        inputs, target = Variable(data[0]),Variable(data[1])

        # for GPU
        if device_name != 'cpu':
            inputs, target = inputs.to(torch.device(device_name)), target.to(torch.device(device_name))

        if phase == 'training':
            optimizer.zero_grad()  # clear gradients for this training step

        output = model(inputs)
        acc_ = []
        recall_ = []
        precision_ = []
        each_acc_ = []
        each_recall_ = []
        each_precision_ = []
        # calculate loss
        for j, d in enumerate(output):
            acc = pred_acc(torch.Tensor.cpu(target[j]), torch.Tensor.cpu(d),threshold)
            acc_.append(acc)
            recall = pred_recall(torch.Tensor.cpu(target[j]), torch.Tensor.cpu(d),threshold)
            recall_.append(recall)
            precision = pred_precision(torch.Tensor.cpu(target[j]), torch.Tensor.cpu(d),threshold)
            precision_.append(precision)
            each_acc = pred_acc_each_label(torch.Tensor.cpu(target[j]), torch.Tensor.cpu(d),threshold)
            each_acc_.append(each_acc)
            each_recall = pred_recall_each_label(torch.Tensor.cpu(target[j]), torch.Tensor.cpu(d),threshold)
            each_recall_.append(np.asarray(each_recall))
            each_precision = pred_precision_each_label(torch.Tensor.cpu(target[j]), torch.Tensor.cpu(d),threshold)
            each_precision_.append(np.asarray(each_precision))

        loss = criterion(output, target)
        running_loss.append(loss.item())
        running_acc.append(np.asarray(acc_).mean())
        running_recall.append(np.asarray(recall_).mean())
        running_precision.append(np.asarray(precision_).mean())
        running_each_acc.append(np.asarray(each_acc_))
        running_each_recall.append(np.asarray(each_recall_))
        running_each_precision.append(np.asarray(each_precision_))

        if phase == 'training':
            loss.backward()  #backward
            optimizer.step()   #update parameters

    total_batch_loss = np.asarray(running_loss).mean()
    total_batch_acc = np.asarray(running_acc).mean()
    total_batch_recall = np.asarray(running_recall).mean()
    total_batch_precision = np.asarray(running_precision).mean()
    running_each_acc = np.concatenate(running_each_acc, axis=0)
    total_batch_each_acc = np.mean(running_each_acc, axis=0)
    running_each_recall = np.concatenate(running_each_recall, axis=0)
    total_batch_each_recall = np.nanmean(running_each_recall, axis=0)
    running_each_precision = np.concatenate(running_each_precision, axis=0)
    total_batch_each_precision = np.nanmean(running_each_precision, axis=0)

    return total_batch_loss, total_batch_acc, total_batch_recall,total_batch_precision, \
        total_batch_each_acc,total_batch_each_recall, total_batch_each_precision