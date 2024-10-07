import os
import sys
import time
import random
import logging
import torch
import numpy as np
import torch.nn as nn
import argparse
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

from torch import log
from tqdm import tqdm
from pathlib import Path
from common.opt import opts
from common.utils import makedir
from models.M2HGR import M2HGR as M2HGR
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader.EMG_Skeleton_loader import load_EMG_Skeleton_data, ESdata_generator, ESConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loss = 0
    for batch_idx, (data1, data2, data3, target) in enumerate(tqdm(train_loader)):
        M, P, N, target = data1.to(device), data2.to(device), data3.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(M, P, N)

        loss = criterion(output, target)
        train_loss += loss.detach().item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            msg = ('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print(msg)
            logging.info(msg)
            if args.dry_run:
                break
    history['train_loss'].append(train_loss)
    return train_loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for _, (data1, data2, data3, target) in enumerate(tqdm(test_loader)):
            M, P, N, target = data1.to(device), data2.to(device), data3.to(device), target.to(device)
            
            output = model(M, P, N)
            
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # output shape (B,Class)
            # target_shape (B)
            # pred shape (B,1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(correct / len(test_loader.dataset))
    msg = ('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(msg)
    logging.info(msg)


if __name__ == '__main__':
    args = opts().parse()

    set_seed(args.manualSeed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    logging.info(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'batch_size': args.batch_size}
    if use_cuda:  
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        device = torch.device("cuda")
        kwargs.update({'num_workers': 8,
                       'pin_memory': True},)
    else:
        device = torch.device("cpu")

    custom_name = f"M2HGR--lr_{args.lr}_bs_{args.batch_size}_epoch_{args.epochs}"
    savedir = Path('./results') / Path(custom_name)
    makedir(savedir)
    logging.basicConfig(filename=savedir/'train.log', level=logging.INFO)
    history = {
        "train_loss": [],
        "test_loss": [],
        "test_acc": []
    }


    Config = ESConfig()
    load_data = load_EMG_Skeleton_data
    data_generator = ESdata_generator

    C = Config
    Train, Test, le = load_data()

    X_0, X_1, X_2, Y = data_generator(Train, C, le)  
    X_0 = torch.from_numpy(X_0).type('torch.FloatTensor')       # JCD data
    X_1 = torch.from_numpy(X_1).type('torch.FloatTensor')       # skeleton data，shape=(64,21,2)
    X_2 = torch.from_numpy(X_2).type('torch.FloatTensor')       # emg data, shape=(64,8)
    Y = torch.from_numpy(Y).type('torch.LongTensor')

    X_0_t, X_1_t, X_2_t, Y_t = data_generator(Test, C, le)
    X_0_t = torch.from_numpy(X_0_t).type('torch.FloatTensor')
    X_1_t = torch.from_numpy(X_1_t).type('torch.FloatTensor')
    X_2_t = torch.from_numpy(X_2_t).type('torch.FloatTensor')
    Y_t = torch.from_numpy(Y_t).type('torch.LongTensor')


    trainset = torch.utils.data.TensorDataset(X_0, X_1, X_2, Y)
    train_sampler = torch.utils.data.RandomSampler(trainset, replacement=False)
    train_loader = torch.utils.data.DataLoader(trainset, sampler=train_sampler, **kwargs)

    testset = torch.utils.data.TensorDataset(X_0_t, X_1_t, X_2_t, Y_t)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size)


    model = M2HGR(C.frame_l, C.joint_n, C.joint_d, C.feat_d, C.emgdata, C.filters, C.clc_num).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer, factor=args.gamma, patience=5, cooldown=0.5, min_lr=5e-6, verbose=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader,
                           optimizer, epoch, criterion)
        test(model, device, test_loader)
        scheduler.step(train_loss)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    ax1.plot(history['train_loss'])
    ax1.plot(history['test_loss'])
    ax1.legend(['Train', 'Test'], loc='upper left')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Loss')

    ax2.set_title('Model accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.plot(history['test_acc'])
    xmax = np.argmax(history['test_acc'])
    ymax = np.max(history['test_acc'])
    text = "x={}, y={:.3f}".format(xmax, ymax)
    ax2.annotate(text, xy=(xmax, ymax))

    ax3.set_title('Confusion matrix')
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_0_t.to(device), X_1_t.to(
            device), X_2_t.to(device)).cpu().numpy()
    Y_test = Y_t.numpy()
    cnf_matrix = confusion_matrix(
        Y_test, np.argmax(Y_pred, axis=1))
    ax3.imshow(cnf_matrix)
    fig.tight_layout()
    fig.savefig(str(savedir / "perf.png"))
    if args.save_model:
        torch.save(model.state_dict(), str(savedir/"model.pt"))
    if args.calc_time:
        device = ['cpu', 'cuda']
        # calc time
        for d in device:
            tmp_X_0_t = X_0_t.to(d)
            tmp_X_1_t = X_1_t.to(d)
            tmp_X_2_t = X_2_t.to(d)
            model = model.to(d)
            # warm up
            _ = model(tmp_X_0_t, tmp_X_1_t, tmp_X_2_t)

            tmp_X_0_t = tmp_X_0_t.unsqueeze(1)
            tmp_X_1_t = tmp_X_1_t.unsqueeze(1)
            tmp_X_2_t = tmp_X_2_t.unsqueeze(1)
            start = time.perf_counter_ns()
            for i in range(tmp_X_0_t.shape[0]):
                _ = model(tmp_X_0_t[i, :, :, :], tmp_X_1_t[i, :, :, :], tmp_X_2_t[i, :, :, :])
            end = time.perf_counter_ns()
            msg = ("total {}ns, {:.2f}ns per one on {}".format((end - start),
                                                               ((end - start) / (X_0_t.shape[0])), d))
            print(msg)
            logging.info(msg)



