import torch  # 引入 PyTorch
from dataset import get_data_transforms  # 從 dataset.py 載入資料轉換函式
from torchvision.datasets import ImageFolder  # 用於影像資料夾的資料集
import numpy as np  # 數值計算套件
import random  # 亂數控制
import os  # 檔案系統操作
from torch.utils.data import DataLoader  # PyTorch 的資料載入器
from resnet import  wide_resnet50_2  # 引入 ResNet 模型
from de_resnet import de_wide_resnet50_2 # 引入解碼器 ResNet
from dataset import MVTecDataset  # MVTec 資料集類別
# import torch.backends.cudnn as cudnn  # CUDA cuDNN 加速
import argparse  # 命令列參數處理
from test import evaluation, visualization, test  # 測試、評估與可視化函式
from torch.nn import functional as F  # 引入 PyTorch 的函式介面
def setup_seed(seed):
    # 設定隨機種子，確保實驗可重現
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 保證結果可重現
    torch.backends.cudnn.benchmark = False  # 關閉自動最佳化搜尋
def loss_fucntion(a, b):
    # 自訂的損失函式：基於 Cosine 相似度
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        # 將特徵展平後計算 Cosine 相似度
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
    return loss
# 先在全域定義 FullModel，避免 pickle 找不到類別
class FullModel(torch.nn.Module):
    def __init__(self, encoder, bn, decoder):
        super().__init__()
        self.encoder = encoder
        self.bn = bn
        self.decoder = decoder
    def forward(self, x):
        feats = self.encoder(x)
        recons = self.decoder(self.bn(feats))
        return feats, recons
def train(_arch_, _class_, epochs, save_pth_path):
    print(f"🔧 類別: {_class_} | Epochs: {epochs}")
    learning_rate = 0.005
    batch_size = 16
    image_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ 使用裝置: {device}")
    # 資料轉換 gt_ :ground truth 通常是二值或灰階的 defect mask
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = f'./mvtec/{_class_}/train'
    test_path = f'./mvtec/{_class_}'
    # 訓練影像集與載入器
    train_data = ImageFolder(root=train_path, transform=data_transform)
    # 測試影像集與ground truth mask集與載入器
    test_data = MVTecDataset(root=test_path,
                             transform=data_transform,
                             gt_transform=gt_transform,
                             phase="test")
    """ 載入器（DataLoader），用來在訓練與測試階段批次讀取影像資料，並控制資料的順序與批次大小
        num_workers：加速資料載入（例如 num_workers=4）
        pin_memory=True：加速 GPU 傳輸
        drop_last=True：避免最後一批資料太小（訓練時可用）
        從 train_data（通常是 ImageFolder）中讀取訓練影像
        每次讀取 batch_size 張影像（例如 16 張）
        shuffle=True：每個 epoch 都會隨機打亂資料順序，避免模型過度記住資料順序，提升泛化能力 """
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True)
    """ 從 test_data（通常是 MVTecDataset）中讀取測試影像與對應的 ground truth mask
        每次只讀取 1 張影像（batch_size=1），方便做 pixel-level anomaly 檢測與可視化
        shuffle=False：保持原始順序，方便對應 ground truth 與影像名稱 """
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=1,
                                                  shuffle=False)
    # 建立模型
    # 使用 WideResNet50-2 作為 encoder
    """ 載入預訓練的 WideResNet50-2 模型
        encoder 是主幹網路（提取特徵）
        bn 是 BatchNorm 層（可能用於特徵標準化或風格轉換）
        pretrained=True 表示使用 ImageNet 預訓練權重，提升特徵表現力 """
    """ Encoder 提取影像特徵
        Decoder 嘗試重建原始影像
        計算原始影像與重建影像的差異（如 MSE、SSIM）
        差異大的區域可能是異常（defect） """
    encoder, bn = wide_resnet50_2(pretrained=True)
    # 把 encoder 和 bn 移到 GPU 或 CPU（由 device 決定）確保模型在正確的裝置上運行
    encoder = encoder.to(device)
    bn = bn.to(device)
    # 設定 encoder 為 推論模式停用 Dropout、BatchNorm 的更新，確保輸出穩定
    # 表示 encoder 是凍結的（不更新），通常是用預訓練模型做特徵提取，只訓練 decoder 來重建影像或特徵。
    encoder.eval()
    # 並搭配一個對應的 decoder 建立一個 重建式模型
    # 建立對應的 decoder 模型（通常是反向的 WideResNet 結構）pretrained=False：表示 decoder 是隨機初始化的，需自行訓練，移到指定裝置上
    decoder = de_wide_resnet50_2(pretrained=False).to(device)
    """ decoder.parameters()：包含 decoder 模型中所有可訓練的參數（例如卷積核、偏差）
        bn.parameters()：包含 BatchNorm 層的參數（例如均值、方差的學習參數）
        +：將兩者合併成一個參數列表，讓 optimizer 一起更新
        lr=learning_rate：設定學習率，控制每次參數更新的幅度
        betas=(0.5, 0.999)：Adam 的動量參數，影響梯度的平滑與收斂速度 """
    optimizer = torch.optim.Adam(list(decoder.parameters()) +
                                 list(bn.parameters()),
                                 lr=learning_rate,
                                 betas=(0.5, 0.999))
    save_pth_dir = save_pth_path if save_pth_path else 'pths/best'
    os.makedirs(save_pth_dir, exist_ok=True)
    best_ckp_path = os.path.join(save_pth_dir, f'best_{_arch_}_{_class_}.pth')
    best_score = -1
    """ 逐步優化 decoder 和 BatchNorm 層，使模型能夠重建 encoder 的特徵表示，
    進而學習正常樣本的特徵分佈。
    這是 anomaly detection 中常見的自編碼訓練方式。 """
    for epoch in range(epochs):
        # 將 BatchNorm 和 decoder 設為訓練模式
        bn.train()
        decoder.train()
        # 初始化一個 list，用來記錄每個 batch 的 loss，最後可以計算整個 epoch 的平均 loss。
        loss_list = []
        # 逐批讀取訓練影像（不使用標籤），每次處理 batch_size 張影像。
        for img, _ in train_dataloader:
            img = img.to(device)
            # encoder(img)：提取影像的深層特徵（通常是多層 feature map）
            inputs = encoder(img)
            # bn(inputs)：對特徵進行標準化或風格轉換
            # decoder(...)：嘗試重建 encoder 的特徵，學習正常樣本的特徵分佈
            outputs = decoder(bn(inputs))
            # 計算損失：比較 encoder 的原始特徵與 decoder 重建的特徵差異，通常用 Cosine 相似度衡量。
            loss = loss_fucntion(inputs, outputs)
            # zero_grad()：清除舊的梯度
            optimizer.zero_grad()
            # backward()：計算新的梯度
            loss.backward()
            # step()：根據梯度更新 decoder 和 bn 的參數
            optimizer.step()
            # 記錄當前 batch 的 loss，方便後續統計與顯示。
            loss_list.append(loss.item())
        # 顯示每個 epoch 的平均 loss，追蹤訓練進度與收斂情況。
        print(f"📘 Epoch [{epoch+1}/{epochs}] | Loss: {np.mean(loss_list):.4f}")
        # 評估
        auroc_px, auroc_sp, aupro_px = evaluation(
            encoder, bn, decoder, test_dataloader, device
        )
        print(f"🔍 評估 | Pixel AUROC: {auroc_px:.3f}")
        # 更新最佳模型
        if auroc_px > best_score:
            best_score = auroc_px
            # 1️⃣ 存 state_dict
            torch.save({
                'bn': bn.state_dict(),
                'decoder': decoder.state_dict()
            }, best_ckp_path)
            print(f"💾 更新最佳權重檔 → {best_ckp_path}")
            # 2️⃣ 存完整模型物件（推論端可直接 torch.load）
            full_model = FullModel(encoder, bn, decoder).to(device)
            full_model.eval()
            full_model_path = os.path.join(
                save_pth_dir, f'fullmodel_{_arch_}_{_class_}.pth'
            )
            torch.save(full_model, full_model_path)
            print(f"💾 同時保存完整模型 → {full_model_path}")
    return best_ckp_path, best_score, auroc_sp, aupro_px, bn, decoder
if __name__ == '__main__':
    import argparse
    import pandas as pd
    import os
    import torch
    # 解析命令列參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default='bottle', type=str)  # 訓練類別
    parser.add_argument('--epochs', default=1, type=int)  # 訓練回合數
    parser.add_argument('--arch', default='wres50', type=str)  # 模型架構
    args = parser.parse_args()
    setup_seed(111)  # 固定隨機種子
    save_visual_path = f"results/{args.arch}_{args.category}"
    save_pth_path = f"pths/best_{args.arch}_{args.category}"
    # 開始訓練，並接收最佳模型路徑與結果
    best_ckp, auroc_px, auroc_sp, aupro_px, bn, decoder = train(
        args.arch, args.category, args.epochs, save_pth_path)
    print(f"最佳模型: {best_ckp}")
    # 存訓練指標到 CSV
    df_metrics = pd.DataFrame([{
        'Category': args.category,
        'Pixel_AUROC': auroc_px,
        'Sample_AUROC': auroc_sp,
        'Pixel_AUPRO': aupro_px,
        'Epochs': args.epochs
    }])
    metrics_name = f"metrics_{args.arch}_{args.category}.csv"
    df_metrics.to_csv(metrics_name,
                      mode='a',
                      header=not os.path.exists(metrics_name),
                      index=False)
    # 🔥 訓練結束後自動產生可視化結果
    visualization(args.arch,
                  args.category,
                  ckp_path=best_ckp,
                  save_path=save_visual_path)
