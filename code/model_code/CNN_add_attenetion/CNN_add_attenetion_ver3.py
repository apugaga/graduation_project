# Updated CNN_add_attention.py with Debug Mode restored

import os
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# --- Debug helper (restored) ---
def debug_data_loading(data_dir: str, csv_path: str):
    print("CSV exists:", os.path.isfile(csv_path))
    print("Data dir exists:", os.path.isdir(data_dir))
    try:
        df = pd.read_csv(csv_path)
        print("CSV loaded:", df.shape)
        print(df.head())
    except Exception as e:
        print("CSV load error:", e)
        return
    pattern = os.path.join(data_dir, "sub-ACN????_DTIFCT_weighted_similar.nii.gz")
    niis = glob(pattern)
    print("Found NIfTIs:", len(niis))
    for ni in niis[:5]:
        print(" ", ni)
    if niis:
        try:
            img = nib.load(niis[0]).get_fdata()
            print("Sample shape:", img.shape)
            print("Slice z30 shape:", img[:,:,30].shape)
        except Exception as e:
            print("NIfTI read error:", e)

# -------- Dataset --------
class SliceDataset(Dataset):
    def __init__(self, file_paths, labels, slice_idx=28, target_size=128):
        self.file_paths = file_paths
        self.labels = labels
        self.slice_idx = slice_idx
        self.target_size = target_size
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        nii = nib.load(self.file_paths[idx]).get_fdata()
        slice_2d = np.nan_to_num(nii[:, :, self.slice_idx], nan=0.0)
        m, s = slice_2d.mean(), slice_2d.std() if slice_2d.std()>0 else 1.0
        slice_2d = (slice_2d - m) / s
        h,w = slice_2d.shape
        pad = (( (self.target_size-h)//2, self.target_size-h - (self.target_size-h)//2 ),
               ( (self.target_size-w)//2, self.target_size-w - (self.target_size-w)//2 ))
        slice_2d = np.pad(slice_2d, pad, mode='constant', constant_values=0)
        x = torch.from_numpy(slice_2d).unsqueeze(0).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x,y

# -------- Attention & Model --------
class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,kernel_size=k_size,padding=(k_size-1)//2,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1,-2)
        y = self.conv(y).sigmoid().transpose(-1,-2).unsqueeze(-1)
        return x * y

class TraditionalCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            ECA(32),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            ECA(64),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            ECA(128)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(128, num_classes)
    def forward(self,x):
        x = self.features(x)
        x = self.gap(x).view(x.size(0),-1)
        return self.fc(x)

# -------- Metrics & Loops --------
def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    sen = tp/(tp+fn) if tp+fn>0 else 0
    spe = tn/(tn+fp) if tn+fp>0 else 0
    pre = tp/(tp+fp) if tp+fp>0 else 0
    return acc,sen,spe,pre

# 1. 修改 train_one_epoch (刪除 loader 上的 tqdm)
# 1. 修改 train_one_epoch (刪除 loader 上的 tqdm)
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(y.cpu().numpy())
    acc = correct / total
    sen, spe, pre = compute_metrics(all_trues, all_preds)[1:]
    return total_loss / total, (acc, sen, spe, pre)
# -------- Main --------
def main(data_dir, csv_path,
         debug=False, debug_num_samples=None,
         n_splits=5, batch_size=2, epochs=30, lr=1e-4):
    
    # --- Added: 準備 summary 檔案 ---
    os.makedirs("results2", exist_ok=True)
    summary_path = os.path.join("results2", "results_summary.txt")
    summary_file = open(summary_path, "w", encoding="utf-8")
    # -------------------------------

    # 用於收集每個腦區的平均指標
    region_avg = {}
    # Debug check
    if debug:
        debug_data_loading(data_dir, csv_path)

    # Brain regions & slices
    # 原本只有 5 個腦區，改成 10 個：
    brain_regions = [
        ("Posterior corona radiata R", 46),
        ("Retrolenticular part of internal capsule R", 39),
        ("Anterior corona radiata R", 44),
        ("Tapetum L", 41),
        ("Superior corona radiata R", 52),
        ("Superior fronto-occipital fasciculus (ant. internal capsule) L", 46),  # ← Added
        ("Genu of corpus callosum", 37),                                        # ← Added
        ("Anterior corona radiata L", 44),                                       # ← Added
        ("Body of corpus callosum", 50),                                         # ← Added
        ("Posterior corona radiata L", 48)                                       # ← Added
    ]



    # Load labels
    df=pd.read_csv(csv_path,usecols=["subject_id","label"])
    id2lab=dict(zip(df.subject_id.astype(str),df.label.astype(int)))

    # Find samples
    files=glob(os.path.join(data_dir,"sub-ACN????_DTIFCT_weighted_similar.nii.gz"))
    samples=[(f,id2lab[os.path.basename(f).split("_")[0].replace("sub-","")])
             for f in files if os.path.basename(f).split("_")[0].replace("sub-","") in id2lab]

    # Debug limit
    if debug and debug_num_samples:
        print(f"Debug: only {debug_num_samples} samples")
        samples=samples[:debug_num_samples]

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for region_name,slice_idx in brain_regions:
        print(f"這是 {region_name} 的結果 (使用 z={slice_idx})")
        
        # --- Added: 寫入折結果標頭 ---
        summary_file.write(f"{region_name} 各折結果:\n")
        # -------------------------------
        
        region_dir=os.path.join("results2",region_name.replace(" ","_"))
        os.makedirs(region_dir,exist_ok=True)
        file_paths,labels=zip(*samples)
        file_paths,labels=list(file_paths),list(labels)
        skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=2025)
        fold_metrics=[]
        for fold,(tidx,teidx) in enumerate(skf.split(file_paths,labels),1):
            tr,va=train_test_split(tidx,test_size=0.1,stratify=[labels[i] for i in tidx],random_state=2025)
            fold_dir=os.path.join(region_dir,f"fold_{fold}")
            os.makedirs(fold_dir,exist_ok=True)
            # datasets
            tr_ds=SliceDataset([file_paths[i] for i in tr],[labels[i] for i in tr],slice_idx)
            va_ds=SliceDataset([file_paths[i] for i in va],[labels[i] for i in va],slice_idx)
            te_ds=SliceDataset([file_paths[i] for i in teidx],[labels[i] for i in teidx],slice_idx)
            tr_ld=DataLoader(tr_ds,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
            va_ld=DataLoader(va_ds,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
            te_ld=DataLoader(te_ds,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
            # model
            model=TraditionalCNN().to(device)
            crit=nn.CrossEntropyLoss()
            opt=optim.Adam(model.parameters(),lr=lr)
            sched=optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.5,patience=3,min_lr=1e-5)
            writer=SummaryWriter(log_dir=fold_dir)
            best_acc=0
            for ep in tqdm(range(1, epochs + 1), desc=f"Fold{fold} Epochs", leave=True):
                tr_loss, tr_acc = train_one_epoch(model, tr_ld, crit, opt, device)
                va_loss, (va_acc, va_sen, va_spe, va_pre) = eval_one_epoch(model, va_ld, crit, device)
                sched.step(va_loss)
            
                # 一次性更新 epoch-level 指標
                tqdm.write(
                    f"Epoch {ep}/{epochs} → "
                    f"tr_loss: {tr_loss:.4f}, tr_acc: {tr_acc:.4f}, "
                    f"va_loss: {va_loss:.4f}, va_acc: {va_acc:.4f}"
                )
            
                writer.add_scalar("Loss/train", tr_loss, ep)
                writer.add_scalar("Loss/val",   va_loss,   ep)
                writer.add_scalar("Acc/train",  tr_acc,    ep)
                writer.add_scalar("Acc/val",    va_acc,    ep)
            
                if va_acc > best_acc:
                    best_acc = va_acc
                    torch.save(model.state_dict(), os.path.join(fold_dir, f"best_model_fold{fold}.pth"))
            writer.close()
            # test
            _, (te_acc, te_sen, te_spe, te_pre) = eval_one_epoch(model, te_ld, crit, device)
            fold_metrics.append((te_acc, te_sen, te_spe, te_pre))
            print(f"{region_name} Fold{fold} Test → ACC:{te_acc:.4f},SEN:{te_sen:.4f},SPE:{te_spe:.4f},PRE:{te_pre:.4f}")
            # --- Added: 各折結果寫入 & 顯示 ---
        
        print(f"{region_name} 各折結果:")
        for i, (acc, sen, spe, pre) in enumerate(fold_metrics, start=1):
            line = f"  Fold{i}: ACC:{acc:.4f}, SEN:{sen:.4f}, SPE:{spe:.4f}, PRE:{pre:.4f}\n"
            summary_file.write(line)
            print(line.strip())
            
        avg = np.mean(fold_metrics, axis=0)
        print( f"{region_name} 平均結果 → ACC:{avg[0]:.4f}, SEN:{avg[1]:.4f}, SPE:{avg[2]:.4f}, PRE:{avg[3]:.4f}\\n")
        avg_line = f"{region_name} 平均結果 → ACC:{avg[0]:.4f}, SEN:{avg[1]:.4f}, SPE:{avg[2]:.4f}, PRE:{avg[3]:.4f}\\n"
        summary_file.write(avg_line)
        print(avg_line.strip())
        region_avg[region_name] = avg
        
    # 10. 所有切面平均結果總結
    summary_file.write("所有切面平均結果：\n")
    print("所有切面平均結果：")
    for region_name, metrics in region_avg.items():
        acc, sen, spe, pre = metrics
        print(f"{region_name} 平均 → ACC:{acc:.4f}, SEN:{sen:.4f}, SPE:{spe:.4f}, PRE:{pre:.4f}\n")
        final_line = f"{region_name} 平均 → ACC:{acc:.4f}, SEN:{sen:.4f}, SPE:{spe:.4f}, PRE:{pre:.4f}\n"
        summary_file.write(final_line)
        print(final_line.strip())
    summary_file.close()

# =============================================================================
#             model.load_state_dict(torch.load(os.path.join(fold_dir,f"best_model_fold{fold}.pth")))
#             _,(ta,tse,tspe,tpr)=eval_one_epoch(model,te_ld,crit,device)
#             print(f"{region_name} Fold{fold} Test → ACC:{ta:.4f},SEN:{tse:.4f},SPE:{tspe:.4f},PRE:{tpr:.4f}")
#             fold_metrics.append((ta,tse,tspe,tpr))
#         m=np.mean(fold_metrics,axis=0)
#         print(f"{region_name} 平均 → ACC:{m[0]:.4f},SEN:{m[1]:.4f},SPE:{m[2]:.4f},PRE:{m[3]:.4f}\n")
# =============================================================================

if __name__=="__main__":
    data_dir=r"D:\laboratory\Graduation_thesis\data\data_fish\no_nan_JHU"
    csv_path=r"D:\laboratory\Graduation_thesis\data\data_fish\paitient_data\label.csv"
    # To debug: set debug=True, debug_num_samples=n
    main(data_dir,csv_path, debug=True, debug_num_samples=100)
