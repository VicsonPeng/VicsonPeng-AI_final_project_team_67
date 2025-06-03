from pathlib import Path
import matplotlib.pyplot as plt
from torch import log
from tqdm import tqdm
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import GradScaler, autocast

import itertools
from dataloader.ibn_loader import load_ibn_data, Idata_generator, IConfig
from model.HT_ConvNet import HT_ConvNet as HT_ConvNet # change this line
from utils import makedir
import sys
import time
import numpy as np
import logging
sys.path.insert(0, './pytorch-summary/torchsummary/')
#from torchsummary import summary  # noqa

savedir = Path('ibn_experiments') 
makedir(savedir)
logging.basicConfig(filename=savedir/'train.log', level=logging.INFO)
history = {
    "train_loss": [],
    "test_loss": [],
    "test_acc": []
}
class_names = ['B0A', 'B0B', 'D0X', 'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11']

scaler = GradScaler()

def train(args, model, device, train_loader, optimizer, epoch, criterion, use_cuda):
    model.train()
    train_loss = 0

    for batch_idx, (data1, data2, target) in enumerate(tqdm(train_loader)):
        M, P, target = data1.to(device), data2.to(device), target.to(device)
        optimizer.zero_grad()

        if use_cuda:
            # GPU 上启用 AMP
            with torch.cuda.amp.autocast():
                output = model(M, P)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU 上不使用 AMP
            output = model(M, P)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            msg = ( f'Train Epoch: {epoch} '
                    f'[{batch_idx * len(data1)}/{len(train_loader.dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                    f'Loss: {loss.item():.6f}' )
            print(msg)
            logging.info(msg)
            if args.dry_run:
                break

    history['train_loss'].append(train_loss)
    return train_loss


def test(model, device, test_loader, epoch, best_acc, savedir):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for _, (data1, data2, target) in enumerate(tqdm(test_loader)):
            M, P, target = data1.to(device), data2.to(device), target.to(device)
            output = model(M, P)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_accuracy)
    msg = ('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * test_accuracy))
    print(msg)
    logging.info(msg)

    if test_accuracy > best_acc:
        best_acc = test_accuracy
        best_model_path = savedir / f"model_epoch_{epoch}_acc_{best_acc:.3f}.pt"
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model with accuracy: {best_acc:.3f} saved to {best_model_path}")
    return best_acc  

def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=199, metavar='N',
                        help='number of epochs to train (default: 199)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dataset', type=int, required=True, metavar='N',
                        help='0 for JHMDB, 1 for SHREC coarse, 2 for SHREC fine, others is undefined')
    parser.add_argument('--model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--calc_time', action='store_true', default=False,
                        help='calc calc time per sample')
    args = parser.parse_args()
    logging.info(args)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},)

    # alias
    Config = None
    data_generator = None
    load_data = None
    clc_num = 0

    Config = IConfig()
    load_data = load_ibn_data
    clc_num = Config.class_num
    data_generator = Idata_generator('label')


    C = Config
    Train, Test, le = load_data()
    X_0, X_1, Y = data_generator(Train, C, le)
    X_0 = torch.from_numpy(X_0).type('torch.FloatTensor')
    X_1 = torch.from_numpy(X_1).type('torch.FloatTensor')
    Y = torch.from_numpy(Y).type('torch.LongTensor')

    X_0_t, X_1_t, Y_t = data_generator(Test, C, le)
    X_0_t = torch.from_numpy(X_0_t).type('torch.FloatTensor')
    X_1_t = torch.from_numpy(X_1_t).type('torch.FloatTensor')
    Y_t = torch.from_numpy(Y_t).type('torch.LongTensor')

    trainset = torch.utils.data.TensorDataset(X_0, X_1, Y)
    train_loader = torch.utils.data.DataLoader(trainset, **kwargs)

    testset = torch.utils.data.TensorDataset(X_0_t, X_1_t, Y_t)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size)

    Net = HT_ConvNet(C.frame_l, C.joint_n, C.joint_d,
                C.feat_d, C.filters, clc_num)
    model = Net.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-7, max_lr=args.lr, 
                                               step_size_up=5, mode='exp_range', gamma=0.85, 
                                               cycle_momentum=False)
    best_acc = 0.0 
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader,
                           optimizer, epoch, criterion, use_cuda)
        best_acc = test(model, device, test_loader, epoch, best_acc, savedir)
        scheduler.step(train_loss)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
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

    with torch.no_grad():
        logits = model(X_0_t.to(device), X_1_t.to(device))
    # 再 detach 并转成 numpy
    Y_pred_logits = logits.detach().cpu().numpy()
    Y_pred_labels = np.argmax(Y_pred_logits, axis=1)
    Y_true = Y_t.numpy()
    cnf_matrix = confusion_matrix(Y_true, Y_pred_labels)

    im = ax3.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.set_title('Confusion Matrix')

    # 4. 添加 colorbar
    cbar = fig.colorbar(im, ax=ax3)
    cbar.ax.set_ylabel('Sample Count', rotation=-90, va="bottom")

    # 5. 设置坐标刻度及标签
    tick_marks = np.arange(len(class_names))
    ax3.set_xticks(tick_marks)
    ax3.set_xticklabels(class_names, rotation=45, ha='right')
    ax3.set_yticks(tick_marks)
    ax3.set_yticklabels(class_names)

    # 6. 在每个格子里写上对应的数字，并根据阈值选择文字颜色（白/黑）
    fmt = 'd'
    thresh = cnf_matrix.max() / 2.0
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        count = cnf_matrix[i, j]
        color = "white" if count > thresh else "black"
        ax3.text(j, i, format(count, fmt),
                horizontalalignment="center",
                color=color)

    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')

    # 7. 调整布局，防止标签被遮挡
    fig.tight_layout()

    # 8. 保存图像
    fig.savefig(str(savedir / "perf.png"))
    plt.close(fig)

    if args.save_model:
        torch.save(model.state_dict(), str(savedir/"model.pt"))
    if args.calc_time:
        device = ['cpu', 'cuda']
        # calc time
        for d in device:
            tmp_X_0_t = X_0_t.to(d)
            tmp_X_1_t = X_1_t.to(d)
            model = model.to(d)
            # warm up
            _ = model(tmp_X_0_t, tmp_X_1_t)

            tmp_X_0_t = tmp_X_0_t.unsqueeze(1)
            tmp_X_1_t = tmp_X_1_t.unsqueeze(1)
            start = time.perf_counter_ns()
            for i in range(tmp_X_0_t.shape[0]):
                _ = model(tmp_X_0_t[i, :, :, :], tmp_X_1_t[i, :, :, :])
            end = time.perf_counter_ns()
            msg = ("total {}ns, {:.2f}ns per one on {}".format((end - start),
                                                               ((end - start) / (X_0_t.shape[0])), d))
            print(msg)
            logging.info(msg)


if __name__ == '__main__':
    main()