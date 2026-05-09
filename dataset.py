# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

class QoEBurstDataset(Dataset):
    def __init__(self, X_data, y_data, scaler=None, is_train=True):
        N, seq_len, feat_dim = X_data.shape
        
        # 1. 提取 Attention Mask: 只要有一个特征不是 0，就是真实的 Burst (1)，全 0 则是填充 (0)
        self.mask = (np.abs(X_data).sum(axis=-1) > 1e-6).astype(int)
        
        X_flatten = X_data.reshape(-1, feat_dim)
        
        if is_train:
            self.scaler = StandardScaler()
            # 只对真实的 Burst 进行统计和拟合，不让 Padding 参与
            valid_data = X_flatten[self.mask.reshape(-1) == 1]
            if len(valid_data) > 0:
                self.scaler.fit(valid_data)
            X_scaled = self.scaler.transform(X_flatten)
        else:
            self.scaler = scaler
            X_scaled = self.scaler.transform(X_flatten)
            
        X_scaled = X_scaled.reshape(N, seq_len, feat_dim)
        
        # 将缩放后 Padding 位置被污染的值，强制清零！
        mask_expanded = np.expand_dims(self.mask, axis=-1)
        X_scaled = X_scaled * mask_expanded
        
        self.X = torch.FloatTensor(X_scaled)
        self.mask_tensor = torch.LongTensor(self.mask)  # Transformer 需要 LongTensor
        self.y = torch.FloatTensor(y_data).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # 现在多返回一个 mask
        return self.X[idx], self.mask_tensor[idx], self.y[idx]
    
    def get_scaler(self):
        return self.scaler