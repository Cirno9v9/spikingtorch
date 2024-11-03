import torch
from torch import nn
from spikingtorch.data import mnist
import argparse
from torch.nn import functional as F
from tqdm import tqdm
from spikingtorch.network import SingleFeedForward
from spikingtorch import encoding

def train(net, train_loader, optimizer, encoder, args):
    train_samples = 0
    train_loss = 0
    train_acc = 0
    net.train()
    for img, label in tqdm(train_loader, total=len(train_loader), leave=False, desc='training'):
        img = img.to(args.device)
        label = label.to(args.device)
        label_onehot = F.one_hot(label, 10).float()
        out_fr = 0.
        for t in range(args.T):
            encoded_img = encoder(img)
            out_fr += net(encoded_img)
        out_fr = out_fr / args.T
        optimizer.zero_grad()
        loss = F.mse_loss(out_fr, label_onehot)
        loss.backward()
        optimizer.step()
        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (out_fr.argmax(1) == label).float().sum().item()
        net.reset()
    train_loss /= train_samples
    train_acc /= train_samples
    return train_loss, train_acc

def test(net, test_loader, encoder, args):
    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for img, label in tqdm(test_loader, total=len(test_loader), leave=False, desc='testing'):
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()
            out_fr = 0.
            for t in range(args.T):
                encoded_img = encoder(img)
                out_fr += net(encoded_img)
            out_fr = out_fr / args.T
            loss = F.mse_loss(out_fr, label_onehot)
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            net.reset()
    test_loss /= test_samples
    test_acc /= test_samples
    return test_loss, test_acc

def main():
    args = get_hyper_params()
    net = SingleFeedForward(args.tau)
    net.to(args.device)
    train_loader, test_loader = mnist.load_mnist(args.data_dir, args.b, args.j)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    encoder = encoding.PoissonEncoder()

    for epoch in range(args.epochs):
        train_loss, train_acc = train(net, train_loader, optim, encoder, args)
        test_loss, test_acc = test(net, test_loader, encoder, args)
        print(f'epoch ={epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')
        

def get_hyper_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', default=100, type=int, help='脉冲序列时间步长')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='训练设备')
    parser.add_argument('--b', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='训练总轮数')
    parser.add_argument('--j', default=4, type=int, help='读数据线程数')
    parser.add_argument('--data-dir', default='spikingtorch/datasets/mnist', type=str, help='数据集存放位置，如果没有下载数据集会自动下载，但要把文件夹创建好')
    parser.add_argument('--lr', default=1e-3, type=float, help='学习率')
    parser.add_argument('--tau', default=2.0, type=float, help='LIF神经元膜电位时间常数')
    return parser.parse_args()


if __name__ == '__main__': main()