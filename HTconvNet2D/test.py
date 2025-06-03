#! /usr/bin/env python
#! coding:utf-8
import torch
import argparse
from tqdm import tqdm
from pathlib import Path

from dataloader.ibn_loader import load_ibn_data, Idata_generator, IConfig
from model.HT_ConvNet import HT_ConvNet as HT_ConvNet 
import sys
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    total_inference_time = 0
    start_time = time.time()
    with torch.no_grad():
        for data1, data2, target in tqdm(test_loader):
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            batch_start_time = time.time()
            output = model(data1, data2)
            batch_end_time = time.time()
            total_inference_time += (batch_end_time - batch_start_time)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    end_time = time.time()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * test_accuracy:.2f}%)')
    total_frames = len(test_loader.dataset)
    total_time = end_time - start_time
    fps = total_frames / total_time
    print(f'Processed {total_frames} frames in {total_time:.2f} seconds ({fps:.2f} FPS)')
    print(f'Total inference time for all batches: {total_inference_time:.2f} seconds')
    print(f'Average inference time per batch: {total_inference_time / len(test_loader):.6f} seconds')

def main():
    parser = argparse.ArgumentParser(description='Test a trained model on a dataset.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model file.')
    parser.add_argument('--dataset', type=int, required=True, help='0 for JHMDB, 1 for SHREC coarse, 2 for SHREC fine.')
    parser.add_argument('--no-cuda', action='store_true', help='Disables CUDA testing.')
    args = parser.parse_args()

    # Load the data and initialize the data loaders

    Config = IConfig()
    load_data = load_ibn_data
    clc_num = Config.class_num
    data_generator = Idata_generator('label')


    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    # Load the model
    model = HT_ConvNet(Config.frame_l, Config.joint_n, Config.joint_d,
                  Config.feat_d, Config.filters, clc_num).to(device)
    # Load the saved model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    total_params = count_parameters(model)
    print(f'Total number of parameters in the model: {total_params}')
    _, Test, le = load_data()
    data_gen = data_generator(Test, Config, le)
    X_0_t, X_1_t, Y_t = data_gen
    X_0_t = torch.from_numpy(X_0_t).type(torch.FloatTensor)
    X_1_t = torch.from_numpy(X_1_t).type(torch.FloatTensor)
    Y_t = torch.from_numpy(Y_t).type(torch.LongTensor)
    testset = torch.utils.data.TensorDataset(X_0_t, X_1_t, Y_t)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    test(model, device, test_loader)

if __name__ == '__main__':
    main()
