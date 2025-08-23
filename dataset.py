from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
#定義並回傳影像與標註（ground truth）的前處理流程，讓模型在訓練與測試時能夠一致地處理輸入資料。
def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),# 將影像縮放到指定大小
        transforms.ToTensor(),# 轉成 PyTorch tensor，並將像素值縮放到 [0,1]
        transforms.CenterCrop(isize),# 從中心裁切成 isize 大小
        #transforms.CenterCrop(args.input_size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])# 使用 ImageNet 的平均值與標準差做標準化
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),# 同樣縮放
        transforms.CenterCrop(isize),# 同樣裁切
        transforms.ToTensor()])# 轉成 tensor，但不做 Normalize（因為是 mask）
    #這部分不做標準化，是因為 ground truth 通常是二值或灰階的 defect mask，不需要改變其分佈。
    return data_transforms, gt_transforms
class MVTecDataset(torch.utils.data.Dataset):
    #根據 phase 決定載入 train 或 test 資料夾，並初始化轉換函式
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
    """   掃描所有子資料夾（如 good, crack, hole），建立影像路徑、標註路徑、標籤與缺陷類型
        讓模型在測試時知道哪些是正常樣本、哪些是異常樣本，並用 mask 來計算 pixel-level 評估指標（如 AUROC、AUPRO） """
    def load_dataset(self): #分類邏輯
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []
        defect_types = os.listdir(self.img_path)
        for defect_type in defect_types:
            #good 類別 → 沒有 ground truth mask → gt_path = 0 → label = 0
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:#其他缺陷類別 → 有對應的 mask → label = 1
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))
        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types
    #回傳資料集大小
    def __len__(self):
        return len(self.img_paths)
    """根據索引讀取影像與對應的 ground truth mask，並套用轉換
    __getitem__ 的資料處理
    讀取影像並轉成 RGB → 套用 transform
    如果是正常樣本（gt == 0）→ 建立全零的 mask
    如果是異常樣本 → 讀取對應的 mask 並套用 gt_transform
    最後回傳 (img, gt, label, img_type)，供模型推論與評估使用 """
    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"
        # img         # Tensor, shape: [3, H, W]
        # gt          # Tensor, shape: [1, H, W]，缺陷位置的 mask
        # label       # int, 0 表正常，1 表異常
        # img_type    # str, 缺陷類型（如 'crack', 'hole', 'good'）
        # 適合 pixel-level anomaly detection，計算 AUROC 和 AUPRO 的評估函式
        return img, gt, label, img_type