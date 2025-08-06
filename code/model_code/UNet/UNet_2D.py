# train_age_class.py

import os
import random
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
# -------------------------------
# 在此直接指定參數（不要再用 argparse）
# -------------------------------
data_dir   = r"C:\Users\User\Desktop\unet\data_fish\similar"  
csv_path   = r"C:\Users\User\Desktop\unet\data_fish\patient_data\labels.csv"
num_epochs = 50      
batch_size = 16      
lr         = 1e-4    
slice_idx  = 28      
save_path  = "./checkpoints"

# -------------------------------
# Dataset：讀取 NIfTI、抽 slice、對應 binary label
# -------------------------------
class SliceBinaryDataset(Dataset):
    def __init__(self, data_dir, csv_path, slice_idx, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.slice_idx = slice_idx
        self.transform = transform

        # 只抓 subject_id 與 label
        df = pd.read_csv(csv_path, usecols=["subject_id","label"])
        self.id2label = dict(zip(df.subject_id, df.label))

        all_niis = glob(os.path.join(data_dir, "sub-ACN????_DTIFCT_weighted_similar.nii"))
        self.samples = []
        for nii in all_niis:
            sid = os.path.basename(nii).split("_")[0].replace("sub-","")
            if sid in self.id2label:
                self.samples.append((nii, int(self.id2label[sid])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        nii_path, label = self.samples[idx]
        img = nib.load(nii_path).get_fdata()
        slice_2d = img[:,:,self.slice_idx].astype(np.float32)
        # 1) 先把 NaN 轉成 0
        slice_2d = np.nan_to_num(slice_2d, nan=0.0)

        # Normalize
        m, s = slice_2d.mean(), slice_2d.std() if slice_2d.std()>0 else 1.0
        slice_2d = (slice_2d - m) / s

        x = torch.from_numpy(slice_2d[None])  # shape (1,H,W)
        if self.transform:
            x = self.transform(x)
        return x, torch.tensor(label, dtype=torch.long)

# -------------------------------
# 2D U-Net + Classification Head
# -------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(out_ch),
        )
    def forward(self, x): return self.net(x)

class UNet2D(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.conv1 = DoubleConv(in_ch, base_ch);       
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(base_ch, base_ch*2);    
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(base_ch*2, base_ch*4);  
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(base_ch*4, base_ch*8);  
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(base_ch*8, base_ch*16)
        
        self.up1 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, 2); 
        self.dec1 = DoubleConv(base_ch*16, base_ch*8)
        self.up2 = nn.ConvTranspose2d(base_ch*8,  base_ch*4, 2, 2); 
        self.dec2 = DoubleConv(base_ch*8,  base_ch*4)
        self.up3 = nn.ConvTranspose2d(base_ch*4,  base_ch*2, 2, 2); 
        self.dec3 = DoubleConv(base_ch*4,  base_ch*2)
        self.up4 = nn.ConvTranspose2d(base_ch*2,  base_ch,   2, 2); 
        self.dec4 = DoubleConv(base_ch*2,  base_ch)
    def forward(self, x):
        x1 = self.conv1(x); x2 = self.pool1(x1)
        x3 = self.conv2(x2); x4 = self.pool2(x3)
        x5 = self.conv3(x4); x6 = self.pool3(x5)
        x7 = self.conv4(x6); x8 = self.pool4(x7)
        b  = self.bottleneck(x8)
        d1 = self.dec1(torch.cat([self.up1(b), x7],1))
        d2 = self.dec2(torch.cat([self.up2(d1),x5],1))
        d3 = self.dec3(torch.cat([self.up3(d2),x3],1))
        d4 = self.dec4(torch.cat([self.up4(d3),x1],1))
        return d4

class ClassHead(nn.Module):
    def __init__(self, in_ch=32, num_classes=2):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc  = nn.Linear(in_ch, num_classes)
    def forward(self, x):
        x = self.gap(x).view(x.size(0),-1)
        return self.fc(x)

# -------------------------------
# training / evaluation loops
# -------------------------------
def train_epoch(net, head, opt, crit, loader, device):
    net.train(); head.train()
    tot_loss = tot_corr = tot_n = 0
    for x,y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        feats = net(x); logits = head(feats)
        loss = crit(logits, y); loss.backward(); opt.step()
        preds = logits.argmax(1)
        tot_loss += loss.item() * x.size(0)
        tot_corr += (preds==y).sum().item()
        tot_n += x.size(0)
    return tot_loss/tot_n, tot_corr/tot_n

def eval_epoch(net, head, crit, loader, device):
    net.eval(); head.eval()
    tot_loss = tot_corr = tot_n = 0
    with torch.no_grad():
        for x,y in tqdm(loader, desc="Eval", leave=False):
            x, y = x.to(device), y.to(device)
            feats = net(x); logits = head(feats)
            loss = crit(logits, y)
            preds = logits.argmax(1)
            tot_loss += loss.item() * x.size(0)
            tot_corr += (preds==y).sum().item()
            tot_n += x.size(0)
    return tot_loss/tot_n, tot_corr/tot_n

# -------------------------------
# main training script
# -------------------------------
def main():
    # 參數設定
    debug = True
    debug_size = 32

    # 確保存檔資料夾存在
    os.makedirs(save_path, exist_ok=True)

    # 固定隨機種子
    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)
    torch.cuda.manual_seed_all(2025)

    # 建立完整資料集
    full_ds = SliceBinaryDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        slice_idx=slice_idx,
        transform=transforms.RandomHorizontalFlip(0.5)
    )

    # Debug 模式：只取 debug_size 筆樣本
    if debug:
        full_ds.samples = full_ds.samples[:debug_size]

    # 分割 train / val
    N = len(full_ds)
    idxs = list(range(N))
    random.shuffle(idxs)
    split = int(0.8 * N)
    train_idxs, val_idxs = idxs[:split], idxs[split:]

    train_loader = DataLoader(
        torch.utils.data.Subset(full_ds, train_idxs),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(full_ds, val_idxs),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # 模型、Loss、Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    net = UNet2D().to(device)
    head = ClassHead().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        list(net.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=1e-4
    )

    # 紀錄指標
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    best_val_acc = 0.0
    for ep in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch(net, head, optimizer, criterion, train_loader, device)
        va_loss, va_acc = eval_epoch(net, head, criterion, val_loader, device)

        # 紀錄並輸出
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        print(f"Epoch {ep}/{num_epochs} "
              f"Train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"Val   loss={va_loss:.4f} acc={va_acc:.3f}")

        # 儲存最佳模型
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "net": net.state_dict(),
                "head": head.state_dict(),
                "acc": best_val_acc,
                "epoch": ep
            }, os.path.join(save_path, "best_model.pth"))

    print("Training complete. Best Val Acc:", best_val_acc)

    # 使用真實資料（debug=False）或直接畫圖
    epochs = range(1, len(train_losses) + 1)

    # Loss 曲線
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val   Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")
    plt.show()

    # Accuracy 曲線
    plt.figure()
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs,   label="Val   Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over Epochs")
    plt.show()

if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
