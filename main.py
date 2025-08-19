# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation, visualization, test
from torch.nn import functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_fucntion(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
    return loss


def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(
            F.interpolate(a[item],
                          size=size,
                          mode='bilinear',
                          align_corners=True))
        b_map.append(
            F.interpolate(b[item],
                          size=size,
                          mode='bilinear',
                          align_corners=True))
    a_map = torch.cat(a_map, 1)
    b_map = torch.cat(b_map, 1)
    loss += torch.mean(1 - cos_loss(a_map, b_map))
    return loss


def train(_class_, epochs):
    print(f"🔧 類別: {_class_} | Epochs: {epochs}")
    learning_rate = 0.005
    batch_size = 16
    image_size = 256

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ 使用裝置: {device}")

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = f'./mvtec/{_class_}/train'
    test_path = f'./mvtec/{_class_}'
    ckp_path = f'./checkpoints/wres50_{_class_}.pth'

    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path,
                             transform=data_transform,
                             gt_transform=gt_transform,
                             phase="test")

    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=1,
                                                  shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters()) +
                                 list(bn.parameters()),
                                 lr=learning_rate,
                                 betas=(0.5, 0.999))

    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            loss = loss_fucntion(inputs, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        print(
            f"📘 Epoch [{epoch + 1}/{epochs}] | Loss: {np.mean(loss_list):.4f}")

        if (epoch + 1) == epochs:
            auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder,
                                                      test_dataloader, device)
            print(
                f"✅ 評估結果 | Pixel AUROC: {auroc_px:.3f}, Sample AUROC: {auroc_sp:.3f}, Pixel AUPRO: {aupro_px:.3f}"
            )
            # 建立 checkpoints 資料夾（如果尚未存在）
            os.makedirs('./checkpoints', exist_ok=True)
            # 然後再執行 torch.save()
            torch.save({
                'bn': bn.state_dict(),
                'decoder': decoder.state_dict()
            }, ckp_path)

    return auroc_px, auroc_sp, aupro_px


if __name__ == '__main__':
    import argparse
    import pandas as pd
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default='bottle', type=str)
    parser.add_argument('--epochs', default=25, type=int)
    args = parser.parse_args()

    setup_seed(111)
    auroc_px, auroc_sp, aupro_px = train(args.category, args.epochs)

    df_metrics = pd.DataFrame([{
        'Category': args.category,
        'Pixel_AUROC': auroc_px,
        'Sample_AUROC': auroc_sp,
        'Pixel_AUPRO': aupro_px,
        'Epochs': args.epochs
    }])
    if not os.path.exists('metrics_all.csv'):
        df_metrics.to_csv('metrics_all.csv', index=False)
    else:
        df_metrics.to_csv('metrics_all.csv',
                          mode='a',
                          header=False,
                          index=False)
    # 🔥 加入可視化儲存
    visualization(args.category, save_path=f"results/{args.category}")
