import torch  # 引入 PyTorch 深度學習框架
from dataset import get_data_transforms  # 從 dataset.py 匯入資料增強方法與資料載入函式
# from torchvision.datasets import ImageFolder  # 匯入 PyTorch 官方圖片資料集讀取工具
import numpy as np  # 匯入數值計算函式庫 NumPy
# from torch.utils.data import DataLoader  # PyTorch 的資料載入器 (batch/迭代器)
from resnet import  wide_resnet50_2  # 匯入自定義的 ResNet 模型
from de_resnet import  de_wide_resnet50_2  # 匯入 ResNet 的反卷積解碼器
from dataset import MVTecDataset  # 匯入 MVTec 資料集定義類別 (瑕疵檢測用)
from torch.nn import functional as F  # 匯入 PyTorch 常用函數 (如激活、卷積、插值等)
from sklearn.metrics import roc_auc_score  # 匯入 sklearn 的 ROC AUC 評估指標
import cv2  # OpenCV，用於影像處理
import matplotlib.pyplot as plt  # 繪圖工具 Matplotlib
from sklearn.metrics import auc  # AUC 計算函式
from skimage import measure  # 影像處理工具 (區域標記、區域屬性)
import pandas as pd  # 資料分析工具 Pandas
from numpy import ndarray  # 匯入 ndarray 型別別名
from statistics import mean  # Python 內建平均數計算
from scipy.ndimage import gaussian_filter  # 高斯濾波器
from sklearn import manifold  # 曼哈頓/流形學習工具 (t-SNE, Isomap 等)
from matplotlib.ticker import NullFormatter  # Matplotlib 格式工具
from scipy.spatial.distance import pdist  # 計算向量之間距離
# import matplotlib  # Matplotlib 主套件
# import pickle  # Python 內建物件序列化工具
# === 計算異常熱力圖 ===
def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':  # 乘法模式
        anomaly_map = np.ones([out_size, out_size])  # 初始化為全 1
    else:  # 加法模式
        anomaly_map = np.zeros([out_size, out_size])  # 初始化為全 0
    a_map_list = []  # 儲存每一層的 anomaly map
    for i in range(len(ft_list)):
        fs = fs_list[i]  # 特徵來源
        ft = ft_list[i]  # 特徵目標
        a_map = 1 - F.cosine_similarity(fs, ft)  # 計算 cosine 相似度並轉成 anomaly 分數
        a_map = torch.unsqueeze(a_map, dim=1)  # 增加一個維度 (batch, channel, h, w)
        a_map = F.interpolate(a_map,
                              size=out_size,
                              mode='bilinear',
                              align_corners=True)  # 上採樣至輸出大小
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()  # 轉 numpy
        a_map_list.append(a_map)  # 保存 anomaly map
        if amap_mode == 'mul':  # 乘法聚合
            anomaly_map *= a_map
        else:  # 加法聚合
            anomaly_map += a_map
    return anomaly_map, a_map_list  # 回傳總體 anomaly map 以及每層的 anomaly map
# === 疊加 anomaly map 到原圖 ===
def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255  # 正規化後加在原圖上
    cam = cam / np.max(cam)  # 縮放到 0~1
    return np.uint8(255 * cam)  # 回傳 uint8 格式影像
# === 最小-最大正規化 ===
def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)
# === 轉換成熱力圖 (colormap) ===
def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray),
                                cv2.COLORMAP_JET)  # OpenCV colormap
    return heatmap
# === 評估函式 ===
def evaluation(encoder, bn, decoder, dataloader, device, _class_=None):
    bn.eval()  # 設為推論模式
    decoder.eval()
    gt_list_px = []  # pixel-level ground truth
    pr_list_px = []  # pixel-level prediction
    gt_list_sp = []  # image-level ground truth
    pr_list_sp = []  # image-level prediction
    aupro_list = []  # PRO 評估
    with torch.no_grad():
        for img, gt, label, _ in dataloader:  # 從 dataloader 取資料
            img = img.to(device)  # 把圖片送到 GPU/CPU
            inputs = encoder(img)  # 取 encoder 特徵
            outputs = decoder(bn(inputs))  # 解碼器輸出
            anomaly_map, _ = cal_anomaly_map(inputs,
                                             outputs,
                                             img.shape[-1],
                                             amap_mode='a')  # 計算 anomaly map
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)  # 高斯濾波平滑
            # 二值化 ground truth
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.item() != 0:  # 如果是瑕疵類別
                aupro_list.append(
                    compute_pro(
                        gt.squeeze(0).cpu().numpy().astype(int),
                        anomaly_map[np.newaxis, :, :]))
            # 累積像素級 ground truth 與預測
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            # 累積圖片級 (是否有異常)
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)  # 計算像素級 AUC
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)  # 計算圖片級 AUC
    return auroc_px, auroc_sp, round(np.mean(aupro_list), 3)
# === 測試函式 ===
def test(_class_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 判斷是否使用 GPU
    print(device)
    print(_class_)
    data_transform, gt_transform = get_data_transforms(256, 256)  # 資料增強
    test_path = '../mvtec/' + _class_  # 測試資料路徑
    ckp_path = './checkpoints/' + 'rm_1105_wres50_ff_mm_' + _class_ + '.pth'  # 模型 checkpoint 路徑
    test_data = MVTecDataset(root=test_path,
                             transform=data_transform,
                             gt_transform=gt_transform,
                             phase="test")  # 測試資料集
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=1,
                                                  shuffle=False)  # 測試資料 loader
    encoder, bn = wide_resnet50_2(pretrained=True)  # 建立編碼器
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)  # 建立解碼器
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)  # 載入 checkpoint
    for k, v in list(ckp['bn'].items()):  # 移除 batchnorm 的 memory 欄位
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder,
                                              test_dataloader, device,
                                              _class_)  # 執行評估
    print(_class_, ':', auroc_px, ',', auroc_sp, ',', aupro_px)
    return auroc_px
import os  # 載入作業系統模組，用於檔案路徑處理、建立資料夾
# =============================
# 函式：可視化模型輸出結果
# =============================
def visualization(_arch_, _class_, save_path=None, ckp_path=None):
    print(f"🖼️ 開始可視化類別: {_class_}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 判斷使用 GPU 或 CPU
    data_transform, gt_transform = get_data_transforms(256,
                                                       256)  # 取得影像與標註的資料轉換方式
    test_path = f'./mvtec/{_class_}'  # 測試資料集路徑，與 train() 一致
    # ✅ 如果沒有外部傳入權重檔路徑，就使用預設值
    if ckp_path is None:
        ckp_path = f'./checkpoints/{_arch_}_{_class_}.pth'
    # 建立測試資料集
    test_data = MVTecDataset(root=test_path,
                             transform=data_transform,
                             gt_transform=gt_transform,
                             phase="test")
    # 建立 DataLoader (一次一張圖片，不打亂順序)
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=1,
                                                  shuffle=False)
    # 建立編碼器與 BatchNorm
    encoder, bn = wide_resnet50_2(
        pretrained=True)  # 使用 Wide ResNet50_2 作為 backbone
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()  # 設定為推理模式
    decoder = de_wide_resnet50_2(pretrained=False)  # 解碼器 (Decoder)
    decoder = decoder.to(device)
    
    # Fixed line - explicitly set weights_only=False to suppress warning
    ckp = torch.load(ckp_path, map_location=device, weights_only=False)
    
    for k in list(ckp['bn'].keys()):
        if 'memory' in k:  # 移除 batch norm 的 "memory" 欄位，避免載入失敗
            del ckp['bn'][k]
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    # 建立輸出資料夾
    save_dir = save_path if save_path else f'results/{_class_}'
    os.makedirs(save_dir, exist_ok=True)
    count = 0  # 計數器：已處理圖片數
    with torch.no_grad():  # 關閉梯度，節省記憶體
        for img, gt, label, _ in test_dataloader:  # 逐張處理測試資料
            if label.item() == 0:  # 如果是正常樣本，跳過
                continue
            img = img.to(device)
            inputs = encoder(img)  # 編碼影像特徵
            outputs = decoder(bn(inputs))  # 解碼重建影像
            # 計算異常圖 (Anomaly Map)
            anomaly_map, _ = cal_anomaly_map([inputs[-1]], [outputs[-1]],
                                             img.shape[-1],
                                             amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)  # 高斯平滑
            ano_map = min_max_norm(anomaly_map)  # 正規化到 0~1
            ano_map = cvt2heatmap(ano_map * 255)  # 轉換成熱力圖
            # 將原圖轉為 numpy 格式，並標準化
            img_np = img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255
            img_np = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_BGR2RGB)
            img_norm = np.uint8(min_max_norm(img_np) * 255)
            overlay = show_cam_on_image(img_norm, ano_map)  # 疊加熱力圖
            # 儲存原圖與疊加熱力圖
            cv2.imwrite(f"{save_dir}/{count:03d}_org.png", img_norm)
            cv2.imwrite(f"{save_dir}/{count:03d}_ad.png", overlay)
            count += 1
    print(f"✅ 可視化完成，共儲存 {count} 張圖片至 {save_dir}")
import numpy as np
import pandas as pd
from numpy import ndarray
from skimage import measure
from sklearn.metrics import auc
from statistics import mean
# =============================
# 函式：計算 PRO 指標 (Pixel-wise Recall Overlap)
# =============================
def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> float:
    """計算每個區域重疊率（PRO）與 FPR 在 0~0.3 區間的 AUC"""
    # --- 資料驗證 ---
    assert isinstance(amaps, ndarray), "amaps 必須是 ndarray"
    assert isinstance(masks, ndarray), "masks 必須是 ndarray"
    assert amaps.ndim == 3 and masks.ndim == 3, "amaps 和 masks 必須是三維陣列"
    assert amaps.shape == masks.shape, "amaps 和 masks 的形狀必須一致"
    assert set(np.unique(masks)) <= {0, 1}, "masks 必須是二值 (0 或 1)"
    assert isinstance(num_th, int), "num_th 必須是整數"
    # --- 初始化 ---
    df = pd.DataFrame({
        "pro": pd.Series(dtype="float"),
        "fpr": pd.Series(dtype="float"),
        "threshold": pd.Series(dtype="float")
    })
    min_th, max_th = amaps.min(), amaps.max()
    thresholds = np.linspace(min_th, max_th, num_th)
    # --- 閾值掃描 ---
    for th in thresholds:
        binary_amaps = (amaps > th).astype(np.uint8)  # 閾值二值化
        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            labeled_mask = measure.label(mask)  # 標記 mask 區域
            for region in measure.regionprops(labeled_mask):
                coords = region.coords
                tp_pixels = binary_amap[coords[:, 0], coords[:, 1]].sum()
                pros.append(tp_pixels / region.area)  # 區域內 TP 比例
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()  # 偽陽率

        new_row = pd.DataFrame([{
            "pro": mean(pros) if pros else 0,
            "fpr": fpr,
            "threshold": th
        }])
        # ✅ 避免 concat 空或全 NA 的 DataFrame
        if not new_row.isna().all(axis=None) and not new_row.empty:
            df = pd.concat([df, new_row], ignore_index=True)
    # --- FPR 正規化與 AUC 計算 ---
    df = df[df["fpr"] < 0.3]  # 只保留 FPR < 0.3
    if df.empty or df["fpr"].max() == 0:
        return 0.0
    df["fpr"] = df["fpr"] / df["fpr"].max()  # FPR 正規化
    pro_auc = auc(df["fpr"], df["pro"])  # 計算 AUC
    return pro_auc