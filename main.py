import os
import numpy as np
import pandas as pd
from nfstream import NFStreamer, NFPlugin

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
import copy
from sklearn.model_selection import KFold


class MultiFileVideoStallDataset(Dataset):
    def __init__(self, file_pairs, window_size=10, resolution=64, stride=10, channel_mode='ALL', is_train=False):
        super().__init__()
        self.samples = []
        self.window_size = window_size
        self.stride = stride
        self.resolution = resolution
        self.channel_mode = channel_mode
        self.is_train = is_train

        for packet_csv, label_csv in file_pairs:
            print(f"\n 正在构建图像: {packet_csv} <--> {label_csv}")

            label_dict = self._parse_beijing_time_labels(label_csv)
            df_pkts = self._preprocess_tshark_csv(packet_csv)
            if df_pkts.empty:
                continue

            times = df_pkts['time'].values
            sizes = df_pkts['size'].values
            protos = df_pkts['proto'].values
            is_dls = df_pkts['is_dl'].values

            start_time = times[0]
            end_time = times[-1]
            current_start = start_time
            windows_processed = 0

            while current_start + self.window_size <= end_time:
                current_end = current_start + self.window_size
                start_idx = np.searchsorted(times, current_start, side='left')
                end_idx = np.searchsorted(times, current_end, side='left')

                w_times = times[start_idx:end_idx]
                w_sizes = sizes[start_idx:end_idx]
                w_protos = protos[start_idx:end_idx]
                w_is_dls = is_dls[start_idx:end_idx]

                num_channels = 3 if self.channel_mode == 'ALL' else 1
                mat = np.zeros((num_channels, self.resolution, self.resolution), dtype=np.float32)

                if len(w_times) > 0:
                    t_bins = ((w_times - current_start) / (self.window_size / float(self.resolution))).astype(np.int32)
                    t_bins = np.clip(t_bins, 0, self.resolution - 1)

                    s_bins = ((w_sizes / 1500.0) * float(self.resolution - 1)).astype(np.int32)
                    s_bins = np.clip(s_bins, 0, self.resolution - 1)

                    c_bins = np.zeros_like(w_protos, dtype=np.int32)
                    valid_mask = np.zeros_like(w_protos, dtype=bool)

                    if self.channel_mode in ['ALL', 'R']:
                        mask_r = (w_protos == 17) & w_is_dls
                        c_bins[mask_r] = 0
                        valid_mask[mask_r] = True

                    if self.channel_mode in ['ALL', 'G']:
                        mask_g = ~w_is_dls
                        c_bins[mask_g] = 1 if self.channel_mode == 'ALL' else 0
                        valid_mask[mask_g] = True

                    if self.channel_mode in ['ALL', 'B']:
                        mask_b = (w_protos == 6) & w_is_dls
                        c_bins[mask_b] = 2 if self.channel_mode == 'ALL' else 0
                        valid_mask[mask_b] = True

                    np.add.at(
                        mat,
                        (c_bins[valid_mask], s_bins[valid_mask], t_bins[valid_mask]),
                        w_sizes[valid_mask]
                    )

                mat = np.log1p(mat)

                end_T_sec = int(current_end)
                future_window = range(end_T_sec, end_T_sec + 10)

                stall_found = False
                valid_label_found = False

                for sec in future_window:
                    if sec in label_dict:
                        valid_label_found = True
                        if label_dict[sec] == 1:
                            stall_found = True
                            break

                if valid_label_found:
                    final_label = 1 if stall_found else 0

                    if final_label == 1:
                        self.samples.append((mat, 1))

                        if self.is_train:
                            for _ in range(5):
                                noise = np.random.normal(0, 0.05, mat.shape).astype(np.float32)
                                noisy_mat = mat + noise
                                noisy_mat = np.clip(noisy_mat, 0, None)
                                self.samples.append((noisy_mat, 1))
                    else:
                        self.samples.append((mat, 0))

                current_start += self.stride
                windows_processed += 1

            print(f"文件全局划窗数量: {windows_processed}，累计提取有效样本: {len(self.samples)}")

    def _preprocess_tshark_csv(self, packet_csv):
        df = pd.read_csv(packet_csv, low_memory=False)

        rename_dict = {
            'frame.time_epoch': 'time',
            'frame.len': 'size',
            'ip.proto': 'proto',
            'tcp.srcport': 'tcp_src',
            'udp.srcport': 'udp_src',
            'tcp.dstport': 'tcp_dst',
            'udp.dstport': 'udp_dst'
        }

        df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})

        if 'srcport' not in df.columns:
            df['srcport'] = df.get('tcp_src', pd.Series(dtype=float)).fillna(
                df.get('udp_src', pd.Series(dtype=float))
            )

        if 'dstport' not in df.columns:
            df['dstport'] = df.get('tcp_dst', pd.Series(dtype=float)).fillna(
                df.get('udp_dst', pd.Series(dtype=float))
            )

        df = df[df['proto'].isin([6, 17])]
        df = df[(df['srcport'] == 443) | (df['dstport'] == 443)]

        df['is_dl'] = (df['srcport'] == 443)

        df = df.dropna(subset=['time', 'size', 'proto', 'is_dl'])
        df = df.sort_values(by='time').reset_index(drop=True)

        return df

    def _parse_beijing_time_labels(self, csv_path):
        df = pd.read_csv(csv_path)

        parsed_time = pd.to_datetime(df['时间']).dt.tz_localize('Asia/Shanghai')
        unix_series = parsed_time.apply(lambda x: int(x.timestamp()))

        df_merged = pd.DataFrame({
            'unix_sec': unix_series,
            'is_stall': df['预测结果(流畅/卡顿;0/1)']
        })

        df_merged = df_merged.groupby('unix_sec')['is_stall'].max().reset_index()
        label_dict = dict(zip(df_merged['unix_sec'], df_merged['is_stall']))

        return label_dict

    def __getitem__(self, idx):
        mat, label = self.samples[idx]
        return (
            torch.tensor(mat, dtype=torch.float32),
            torch.tensor([label], dtype=torch.long)
        )

    def __len__(self):
        return len(self.samples)


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=nIn,
            hidden_size=nHidden,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        output, _ = self.rnn(x)
        return output


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[3, 2], stride=1, padding=[1, 1], dilation=[1, 2]):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size[0],
            stride=stride,
            padding=padding[0],
            dilation=dilation[0]
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size[1],
            stride=stride,
            padding=padding[1],
            dilation=dilation[1]
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.downsample(residual)

        out += residual
        out = F.relu(out)

        return out


class CRNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, num_hidden=32, resolution=64):
        super().__init__()

        self.cnn = nn.Sequential(
            BasicBlock(in_channels, num_hidden // 2),
            nn.MaxPool2d(4),

            BasicBlock(num_hidden // 2, num_hidden),
            nn.MaxPool2d(4),

            BasicBlock(num_hidden, num_hidden * 2),
            nn.MaxPool2d((4, 1)),

            BasicBlock(num_hidden * 2, num_hidden * 4),
            nn.MaxPool2d((4, 1)),

            nn.BatchNorm2d(num_hidden * 4)
        )

        self.rnn = nn.Sequential(
            BidirectionalLSTM(num_hidden * 4, num_hidden),
            BidirectionalLSTM(num_hidden * 2, num_hidden)
        )

        seq_len = resolution // 16

        self.fc1 = nn.Linear(seq_len * num_hidden * 2, 64)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        conv = self.cnn(x)

        b, c, h, w = conv.shape

        conv = torch.squeeze(conv, 2)
        conv = conv.permute(0, 2, 1)

        bilstm = self.rnn(conv)

        bilstm = bilstm.reshape(bilstm.shape[0], -1)

        feats = F.relu(self.fc1(bilstm))
        output = self.fc2(self.dropout(feats))

        return output, feats


class FocalLoss(nn.Module):
    def __init__(self, num_classes=2, reduction='mean', alpha=0.25, gamma=2.0, name=None):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma
        self.name = name

    def forward(self, input, label):
        label = torch.squeeze(label, 1)
        label_onehot = F.one_hot(label, num_classes=self.num_classes).float()

        bce_loss = F.binary_cross_entropy_with_logits(
            input,
            label_onehot,
            reduction='none'
        )

        prob = torch.sigmoid(input)
        pt = label_onehot * prob + (1 - label_onehot) * (1 - prob)

        alpha_factor = label_onehot * self.alpha + (1 - label_onehot) * (1 - self.alpha)
        focal_weight = alpha_factor * ((1 - pt) ** self.gamma)

        loss = focal_weight * bce_loss

        fg_num = torch.sum(label_onehot >= 1.0).float()
        fg_num = torch.clamp(fg_num, min=1.0)

        loss = loss / fg_num

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def evaluate_model(model, loader, threshold=None, mode="val", alpha=0.25, device=None):
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    criterion_focal = FocalLoss(num_classes=2, alpha=alpha)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits, feats = model(images)

            loss = criterion_focal(logits, labels)
            total_loss += loss.item()

            probs = F.softmax(logits, dim=1)

            current_thresh = threshold if threshold is not None else 0.5
            preds = (probs[:, 1] > current_thresh).long()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader), 1)

    if threshold is not None and len(set(all_labels)) > 1:
        cm = confusion_matrix(all_labels, all_preds)
        print(
            f"\n【{mode} 集混淆矩阵 (阈值={threshold:.2f})】:\n"
            f"流畅对: {cm[0][0]} | 卡顿漏: {cm[1][0]}\n"
            f"流畅错: {cm[0][1]} | 卡顿准: {cm[1][1]}"
        )

    stall_f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    model.train()

    return avg_loss, stall_f1, acc, all_labels, all_probs


def find_best_threshold(val_labels, val_probs):
    best_threshold = 0.5
    best_score = 0.0

    for threshold in np.arange(0.05, 0.95, 0.05):
        preds = (np.array(val_probs) >= threshold).astype(int)

        precision = precision_score(val_labels, preds, pos_label=1, zero_division=0)
        recall = recall_score(val_labels, preds, pos_label=1, zero_division=0)

        if precision + recall == 0:
            score = 0
        else:
            score = f1_score(val_labels, preds, pos_label=1, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def plot_training_curves(history, save_path="saved_models/training_curves.png"):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o', color='blue')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', marker='s', color='orange')
    axes[0].plot(epochs, history['test_loss'], label='Test Loss', marker='^', color='green')
    axes[0].set_title('Loss Curve', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(epochs, history['train_acc'], label='Train Acc', marker='o', color='blue')
    axes[1].plot(epochs, history['val_acc'], label='Val Acc', marker='s', color='orange')
    axes[1].plot(epochs, history['test_acc'], label='Test Acc', marker='^', color='green')
    axes[1].set_title('Accuracy Curve', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].plot(epochs, history['train_f1'], label='Train F1', marker='o', color='blue')
    axes[2].plot(epochs, history['val_f1'], label='Val F1', marker='s', color='orange')
    axes[2].plot(epochs, history['test_f1'], label='Test F1', marker='^', color='green')
    axes[2].set_title('F1 Score Curve (Stall)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n 线已成功绘制并保存至: {save_path}")
    plt.close()


def train_one_fold(
    fold_id,
    train_file_pairs,
    val_file_pairs,
    test_file_pairs,
    resolution=64,
    epochs=30,
    train_batch_size=32,
    test_batch_size=64,
    learning_rate=0.001,
    num_hidden=64,
    alpha=0.75,
    lmbda=0.0,
    experiment_mode='ALL',
    save_root='saved_models_cv'
):
    print(f" \n 开始训练第 {fold_id} 折")
    print(f"\n Train files: {len(train_file_pairs)} | Val files: {len(val_file_pairs)} | Test files: {len(test_file_pairs)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")

    cnn_in_channels = 3 if experiment_mode == 'ALL' else 1

    train_dataset = MultiFileVideoStallDataset(
        train_file_pairs,
        window_size=10,
        resolution=resolution,
        stride=10,
        channel_mode=experiment_mode,
        is_train=False
    )

    val_dataset = MultiFileVideoStallDataset(
        val_file_pairs,
        window_size=10,
        resolution=resolution,
        stride=10,
        channel_mode=experiment_mode,
        is_train=False
    )

    test_dataset = MultiFileVideoStallDataset(
        test_file_pairs,
        window_size=10,
        resolution=resolution,
        stride=10,
        channel_mode=experiment_mode,
        is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    model = CRNN(
        in_channels=cnn_in_channels,
        num_classes=2,
        num_hidden=num_hidden,
        resolution=resolution
    ).to(device)

    criterion_focal = FocalLoss(num_classes=2, alpha=alpha)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-3
    )

    fold_save_dir = os.path.join(save_root, f"fold_{fold_id}")
    os.makedirs(fold_save_dir, exist_ok=True)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': []
    }

    best_val_f1 = 0
    best_threshold = 0.5
    best_state_dict = None

    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

        epoch_total_loss = 0.0
        all_train_preds, all_train_labels = [], []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits, feats = model(images)

            loss_f = criterion_focal(logits, labels)
            loss = loss_f

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_total_loss += loss.item()

            probs = F.softmax(logits, dim=1)
            preds = (probs[:, 1] > 0.5).long()

            all_train_preds.extend(preds.detach().cpu().numpy().tolist())
            all_train_labels.extend(labels.detach().cpu().numpy().flatten().tolist())

        train_avg_loss = epoch_total_loss / max(len(train_loader), 1)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds, pos_label=1, zero_division=0)

        val_loss, _, val_acc_default, val_labels, val_probs = evaluate_model(
            model,
            val_loader,
            threshold=None,
            mode="Val",
            alpha=alpha,
            device=device
        )

        current_thresh, current_val_f1 = find_best_threshold(val_labels, val_probs)

        val_preds_best = (np.array(val_probs) >= current_thresh).astype(int)
        current_val_acc = accuracy_score(val_labels, val_preds_best)

        history['train_loss'].append(train_avg_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(current_val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(current_val_f1)

        print(
            f"Fold {fold_id} | Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss={train_avg_loss:.4f}, Train F1={train_f1:.4f}, "
            f"Val Loss={val_loss:.4f}, Val F1={current_val_f1:.4f}, Best Th={current_thresh:.2f}"
        )

        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            best_threshold = current_thresh
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0

            torch.save(best_state_dict, os.path.join(fold_save_dir, "best_model.pth"))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停")
            break

    model.load_state_dict(best_state_dict)

    test_loss, test_f1, test_acc, test_labels, test_probs = evaluate_model(
        model,
        test_loader,
        threshold=best_threshold,
        mode="Test",
        alpha=alpha,
        device=device
    )

    result = {
        "fold": fold_id,
        "best_val_f1": best_val_f1,
        "best_threshold": best_threshold,
        "test_loss": test_loss,
        "test_f1": test_f1,
        "test_acc": test_acc,
        "history": history,
        "n_train_samples": len(train_dataset),
        "n_val_samples": len(val_dataset),
        "n_test_samples": len(test_dataset)
    }

    print(f"\n 第 {fold_id} 折结束：")
    print(f"Best Val F1 = {best_val_f1:.4f}")
    print(f"Best Threshold = {best_threshold:.2f}")
    print(f"Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.4f}, Test F1 = {test_f1:.4f}")

    return result


def run_kfold_cross_validation(file_pairs, n_splits=3):
    resolution = 64
    epochs = 30
    train_batch_size = 32
    test_batch_size = 64
    learning_rate = 0.001
    num_hidden = 64
    alpha = 0.5
    lmbda = 0.0
    experiment_mode = 'ALL'
    save_root = f"saved_models_{n_splits}fold"

    os.makedirs(save_root, exist_ok=True)

    outer_kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_results = []

    file_pairs = list(file_pairs)
    indices = np.arange(len(file_pairs))

    for fold_id, (trainval_idx, test_idx) in enumerate(outer_kf.split(indices), start=1):
        trainval_pairs = [file_pairs[i] for i in trainval_idx]
        test_pairs = [file_pairs[i] for i in test_idx]

        inner_splits = min(3, len(trainval_pairs))

        inner_kf = KFold(n_splits=inner_splits, shuffle=True, random_state=100 + fold_id)

        inner_indices = np.arange(len(trainval_pairs))
        inner_train_idx, inner_val_idx = next(inner_kf.split(inner_indices))

        train_pairs = [trainval_pairs[i] for i in inner_train_idx]
        val_pairs = [trainval_pairs[i] for i in inner_val_idx]

        result = train_one_fold(
            fold_id=fold_id,
            train_file_pairs=train_pairs,
            val_file_pairs=val_pairs,
            test_file_pairs=test_pairs,
            resolution=resolution,
            epochs=epochs,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            learning_rate=learning_rate,
            num_hidden=num_hidden,
            alpha=alpha,
            lmbda=lmbda,
            experiment_mode=experiment_mode,
            save_root=save_root
        )

        if result is not None:
            all_results.append(result)

    test_f1s = [r["test_f1"] for r in all_results]
    test_accs = [r["test_acc"] for r in all_results]
    val_f1s = [r["best_val_f1"] for r in all_results]

    print(f"\n{n_splits}-Fold 交叉验证完成")

    for r in all_results:
        print(
            f"Fold {r['fold']}: "
            f"Val F1={r['best_val_f1']:.4f}, "
            f"Test F1={r['test_f1']:.4f}, "
            f"Test Acc={r['test_acc']:.4f}, "
            f"Train/Val/Test Samples={r['n_train_samples']}/{r['n_val_samples']}/{r['n_test_samples']}"
        )

    print("\n===== Overall =====")
    print(f"Mean Val F1 : {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
    print(f"Mean Test F1: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print(f"Mean Test Acc: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")


def main():
    all_file_pairs = [
        ("./csv_data/N3_Facebook_129_0909_100526.csv", "./labels/N3_Facebook_129_20250909_100526.csv"),
        ("./csv_data/N3_Facebook_130_0909_100526.csv", "./labels/N3_Facebook_130_20250909_100526.csv"),
        ("./csv_data/N3_Facebook_175_0909_100526.csv", "./labels/N3_Facebook_175_20250909_100526.csv"),
        ("./csv_data/N3_Facebook_188_0909_100526.csv", "./labels/N3_Facebook_188_20250909_100526.csv"),
        ("./csv_data/N3_Facebook_129_0910_094907.csv", "./labels/N3_Facebook_129_20250910_094907.csv"),
        ("./csv_data/N3_Facebook_130_0910_094907.csv", "./labels/N3_Facebook_130_20250910_094907.csv"),
        ("./csv_data/N3_Facebook_175_0910_094907.csv", "./labels/N3_Facebook_175_20250910_094907.csv"),
        ("./csv_data/N3_Facebook_188_0910_094907.csv", "./labels/N3_Facebook_188_20250910_094907.csv")
    ]

    N_SPLITS = 3

    if len(all_file_pairs) < N_SPLITS:
        raise ValueError(f"\n文件对数量不足，当前只有{len(all_file_pairs)}个文件对，不能做{N_SPLITS}折")

    run_kfold_cross_validation(all_file_pairs, n_splits=N_SPLITS)


if __name__ == '__main__':
    main()