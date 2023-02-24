import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint
import torch.optim as optim
import torch_optimizer

from mcmlic_model import MultiChannelMultiLabelImageCrassifer
from mcmlic_dataset import MCMLICDataset
from modules import contains_common_element, fit_model


def train(opt):
    np.random.seed(0)
    device = torch.device(opt.device_name)
    with open(opt.conf, 'r') as f:
        conf = json.load(f)
        input_channels = conf['input_channels']
        output_labels = conf['output_labels']

    # 訓練用のデータセットを生成する
    np.random.seed(0)
    train_dataset = MCMLICDataset(data_dir=opt.data_dir, label_dir=opt.label_dir, split='train', train_rate=0.8,
                                  device=device)

    # 検証用のデータセットを生成する
    np.random.seed(0)
    val_dataset = MCMLICDataset(data_dir=opt.data_dir, label_dir=opt.label_dir, split='validation', train_rate=0.8,
                                device=device)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                              shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                            shuffle=True, num_workers=0)

    assert not contains_common_element(train_dataset.filenames, val_dataset.filenames), "Lists contain common elements"

    model = MultiChannelMultiLabelImageCrassifer(num_input_channels=len(input_channels),
                                                 num_output_labels=len(output_labels))
    model.to(device)

    torch.autograd.set_detect_anomaly(True)

    # train
    writer = SummaryWriter()
    pos_weight = torch.ones([len(output_labels)]) * 1
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion.to(device)
    if opt.optimizer == 'RAdam':
        optimizer = torch_optimizer.RAdam(model.parameters(), lr=opt.lr)
    elif opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    else:
        assert False, "Invalid optimizer selected."

    num = opt.epoch
    best_val = 1
    trn_losses = []
    trn_acc = []
    trn_each_acc = []
    val_losses = []
    val_acc = []
    val_each_acc = []
    for idx in range(1, num + 1):
        trn_l, trn_a, trn_r, trn_p, trn_each_a, trn_each_r, trn_each_p = fit_model(model=model, dataloader=trainloader,
                                                                                   criterion=criterion,
                                                                                   device_name=opt.device_name,
                                                                                   optimizer=optimizer,
                                                                                   phase='training',
                                                                                   threshold=opt.threshold)
        pprint(f"Epoch: {idx},train loss: {trn_l}. ,train accuracy: {trn_a}.", width=200)
        val_l, val_a, val_r, val_p, val_each_a, val_each_r, val_each_p = fit_model(model=model, dataloader=valloader,
                                                                                   criterion=criterion,
                                                                                   device_name=opt.device_name,
                                                                                   optimizer=optimizer,
                                                                                   phase='validation',
                                                                                   threshold=opt.threshold)
        pprint(f"Epoch: {idx},validation loss: {val_l}. ,validation accuracy: {val_a}.", width=200)
        trn_losses.append(trn_l)
        trn_acc.append(trn_a)
        trn_each_acc.append(trn_each_a)
        val_losses.append(val_l)
        val_acc.append(val_a)
        val_each_acc.append(val_each_a)

        writer.add_scalar('training_loss', trn_l, idx)
        writer.add_scalar('training_acc', trn_a, idx)
        writer.add_scalar('training_recall', trn_r, idx)
        writer.add_scalar('training_precision', trn_p, idx)
        writer.add_scalar('validation_loss', val_l, idx)
        writer.add_scalar('validation_acc', val_a, idx)
        writer.add_scalar('validation_recall', val_r, idx)
        writer.add_scalar('validation_precision', val_p, idx)

        for i in range(len(output_labels)):
            writer.add_scalar(output_labels[i] + '_acc', val_each_a[i], idx)
            writer.add_scalar(output_labels[i] + '_recall', val_each_r[i], idx)
            writer.add_scalar(output_labels[i] + '_precision', val_each_p[i], idx)

        if best_val > val_l:
            torch.save(model.state_dict(), f'model/best_model.pth')
            best_val = val_l
            best_idx = idx

    print('best_idx:', best_idx, 'best_val', best_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi channel multi label image crassifer training')
    parser.add_argument('--data_dir', required=True, help='training data')
    parser.add_argument('--label_dir', required=True, help='training label')
    parser.add_argument('--conf', required=True, help='json file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--device_name', default='cuda')
    parser.add_argument('--optimizer', default='RAdam', help='RAdam,Adam,SGD')
    parser.add_argument('--lr', type=float, default=0.001)

    opt = parser.parse_args()
    train(opt)
