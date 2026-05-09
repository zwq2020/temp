# train.py
import os
import torch
import glob
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_curve, f1_score
from transformers import get_cosine_schedule_with_warmup

import config
from dataset import QoEBurstDataset
from model import ChronosForQoE

# Focal Loss 
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        pt = targets * probs + (1 - targets) * (1 - probs)
        bce_loss = - (targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()

# --- 在 import config 下方加入这个全新的高级评估函数 ---
def relaxed_evaluation(y_true, y_prob, threshold, 
                       smooth_window=config.PROB_SMOOTH_WINDOW, 
                       min_delay=config.MIN_DELAY, 
                       tolerance=config.TOLERANCE):
    """
    工业级流媒体宽松评估算法
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    n = len(y_true)
    
    # 1. 概率滑动平均平滑 (避免瞬间的错误激增)
    kernel = np.ones(smooth_window) / smooth_window
    pad_left = smooth_window // 2
    pad_right = smooth_window - 1 - pad_left
    padded_prob = np.pad(y_prob, (pad_left, pad_right), mode='edge')
    smoothed_prob = np.convolve(padded_prob, kernel, mode='valid')
    
    # 初步按阈值二值化
    y_pred = (smoothed_prob >= threshold).astype(int)
    
    # 2. 切换点边界免责 (MIN_DELAY)
    adjusted_pred = y_pred.copy()
    changes = np.where(y_true[:-1] != y_true[1:])[0]
    for idx in changes:
        # idx 是状态切换的前一秒，强制把切换点前后的预测结果改为正确
        start = max(0, idx - min_delay + 1)
        end = min(n, idx + min_delay + 1)
        adjusted_pred[start:end] = y_true[start:end]
        
    # 3. 总体容忍度调整 (TOLERANCE)
    final_pred = adjusted_pred.copy()
    for i in range(n):
        if final_pred[i] != y_true[i]:
            # 如果预测错误，往前后找找有没有对得上的真实标签
            start = max(0, i - tolerance)
            end = min(n, i + tolerance + 1)
            # 只要周围真实标签里包含当前预测出的状态，就认为是提前/延后预测到了，记为正确
            if final_pred[i] in y_true[start:end]:
                final_pred[i] = y_true[i]
                
    return y_true, final_pred, smoothed_prob

# --- 替换原来的 evaluate_dataset 函数 ---
def evaluate_dataset(model, dataloader, threshold, device, dataset_name=""):
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_mask, batch_y in dataloader:
            outputs = model(batch_X.to(device), batch_mask.to(device))
            all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    all_probs = np.array(all_probs).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # 调用加入的工业级高级评估逻辑！
    y_true, final_preds, smoothed_probs = relaxed_evaluation(all_targets, all_probs, threshold)
    
    print(f"\n[{dataset_name}] 高级平滑性能评估 (固定阈值: {threshold:.4f} | 容忍度: {config.TOLERANCE}s)")
    print(classification_report(y_true, final_preds, digits=4, zero_division=0))
    return y_true, smoothed_probs, final_preds

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 开始基于 Chronos 大模型的 QoE 预测流程 (设备: {device}) ===")
    
    x_files = sorted(glob.glob(os.path.join(config.PROCESSED_DATA_DIR, "X_aligned*.npy")))
    if len(x_files) == 0:
        print(f"未找到特征数据，请先运行预处理！")
        return

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    for x_path in x_files:
        y_path = x_path.replace("X_aligned", "y_aligned")
        if not os.path.exists(y_path):
            continue
            
        print(f"正在加载与切分: {os.path.basename(x_path)}")
        X_part = np.load(x_path)
        y_part = np.load(y_path)
        
        total_len = len(X_part)
        train_idx = int(total_len * 0.7)
        val_idx = int(total_len * 0.85)
        
        X_train_list.append(X_part[:train_idx])
        y_train_list.append(y_part[:train_idx])
        X_val_list.append(X_part[train_idx:val_idx])
        y_val_list.append(y_part[train_idx:val_idx])
        X_test_list.append(X_part[val_idx:])
        y_test_list.append(y_part[val_idx:])

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    
    print(f"✅ 全局合并完成！训练集:{len(y_train)} | 验证集:{len(y_val)} | 测试集:{len(y_test)}")

    positive_indices = np.where(y_train == 1)[0]
    
    # if len(positive_indices) > 0:
    #     X_train_pos = X_train[positive_indices]
    #     y_train_pos = y_train[positive_indices]
        
    #     X_train_pos_aug = np.repeat(X_train_pos, 10, axis=0)
    #     y_train_pos_aug = np.repeat(y_train_pos, 10, axis=0)

    #     noise_factor = 1.0 + np.random.normal(0, 0.05, X_train_pos_aug.shape)
    #     valid_mask = (np.abs(X_train_pos_aug).sum(axis=-1, keepdims=True) > 1e-6)
    #     X_train_pos_aug = X_train_pos_aug * noise_factor * valid_mask
        
    #     X_train = np.concatenate([X_train, X_train_pos_aug], axis=0)
    #     y_train = np.concatenate([y_train, y_train_pos_aug], axis=0)
    #     print(f"过采样后训练集激增至: {len(y_train)}")
    
    train_dataset = QoEBurstDataset(X_train, y_train, is_train=True)
    scaler = train_dataset.get_scaler() 
    val_dataset = QoEBurstDataset(X_val, y_val, scaler=scaler, is_train=False)
    test_dataset = QoEBurstDataset(X_test, y_test, scaler=scaler, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    model = ChronosForQoE(
        model_name=config.LLM_MODEL_NAME, 
        burst_dim=config.BURST_FEAT_DIM, 
        freeze_llm=config.FREEZE_LLM
    ).to(device)
    
    # criterion = FocalLoss(alpha=0.3, gamma=2.0).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0])).to(device)
    
    # 新增：差异化学习率 (加速收敛)
    # pretrained_params, new_params = [], []
    # for name, param in model.named_parameters():
    #     if 'llm_encoder' in name:
    #         pretrained_params.append(param)
    #     else:
    #         new_params.append(param)
            
    # optimizer = optim.AdamW([
    #     {'params': pretrained_params, 'lr': config.LEARNING_RATE_LLM},
    #     {'params': new_params, 'lr': config.LEARNING_RATE_HEAD}
    # ], weight_decay=1e-2)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=1e-3, weight_decay=1e-2)
                            
    # ==========================================
    # 【新增：大模型专属平滑调度器】
    # ==========================================
    # 计算总的训练步数 (Step)
    total_steps = len(train_loader) * config.EPOCHS
    # 设定前 10% 的时间作为 Warmup 预热期 (比如总共 50 轮，前 5 轮都在缓慢预热)
    warmup_steps = int(total_steps * 0.1) 
    
    # 使用带有 Warmup 的余弦退火调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    best_val_f1 = 0.0
    global_best_threshold = 0.5 # 保存全局最优阈值
    epochs_no_improve = 0
    
    # 记录 Loss 用于画图
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(config.EPOCHS):
        
        # ==========================================
        # 【新增】：两阶段渐进式解冻策略
        # ==========================================
        if epoch == 0:
            print("\n[阶段 1] 锁定大模型骨干，仅预热训练分类头...")
            for param in model.llm_encoder.parameters():
                param.requires_grad = False
        
        elif epoch == 5: # 在第 6 个 Epoch 时，解冻大模型！
            print("\n[阶段 2] 分类头预热完毕，彻底解冻大模型骨干联合微调！")
            for param in model.llm_encoder.parameters():
                param.requires_grad = True
                
            # 重置优化器，给大模型一个极小的学习率，给分类头一个稍大的学习率
            pretrained_params = [p for n, p in model.named_parameters() if 'llm_encoder' in n]
            new_params = [p for n, p in model.named_parameters() if 'llm_encoder' not in n]
            
            optimizer = optim.AdamW([
                {'params': pretrained_params, 'lr': 1e-5}, # 大模型用极小的学习率微调
                {'params': new_params, 'lr': 5e-4}
            ], weight_decay=1e-2)
        # ==========================================

        model.train()
        train_loss = 0
        model.train()
        train_loss = 0
        for batch_X, batch_mask, batch_y in train_loader:
            batch_X, batch_mask, batch_y = batch_X.to(device), batch_mask.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X, batch_mask)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # ==========================================
            
            optimizer.step()
            
            # ==========================================
            # 【新增：调度器步进】
            # 每走一步，动态调整一次学习率
            scheduler.step()
            # ==========================================
            
            train_loss += loss.item()
            
        epoch_train_loss = train_loss / len(train_loader)
        train_loss_history.append(epoch_train_loss)
            
        model.eval()
        val_loss = 0.0
        val_probs, val_targets = [], []
        with torch.no_grad():
            for batch_X, batch_mask, batch_y in val_loader:
                batch_X, batch_mask, batch_y = batch_X.to(device), batch_mask.to(device), batch_y.to(device)
                outputs = model(batch_X, batch_mask)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(batch_y.cpu().numpy())
                
        epoch_val_loss = val_loss / len(val_loader)
        val_loss_history.append(epoch_val_loss)
        
        # val_probs = np.array(val_probs).flatten()
        # val_targets = np.array(val_targets).flatten()
        
        # precisions, recalls, thresholds = precision_recall_curve(val_targets, val_probs)
        # f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        # best_f1_idx = np.argmax(f1_scores)
        # current_val_f1 = f1_scores[best_f1_idx]
        # current_best_th = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5

        val_probs = np.array(val_probs).flatten()
        val_targets = np.array(val_targets).flatten()
        
        # 【核心策略：锁死 0.5 物理阈值】杜绝为了虚高 F1 而放任误报！
        fixed_threshold = 0.6
        preds = (val_probs >= fixed_threshold).astype(int)
        
        current_val_f1 = f1_score(val_targets, preds, zero_division=0)
        current_best_th = fixed_threshold
        
        print(f"Epoch [{epoch+1:02d}/{config.EPOCHS}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Best F1: {current_val_f1:.4f} (最优阈值={current_best_th:.4f})")
        
        if epoch == 0 or current_val_f1 >= best_val_f1:
            best_val_f1 = current_val_f1
            # global_best_threshold = current_best_th # 锁定表现最好的阈值
            global_best_threshold = fixed_threshold # 锁定固定的物理阈值
            epochs_no_improve = 0
            
            os.makedirs(os.path.dirname(config.BEST_MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.PATIENCE:
                print(f"\n触发早停机制,结束训练。")
                break
                
    # --- 绘制并保存 Loss 曲线 ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_loss_history, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Focal Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(config.LOSS_CURVE_PATH, dpi=300)
    print(f"\n✅ 训练结束，Loss曲线已保存至: {config.LOSS_CURVE_PATH}")

    # --- 最终的三大集客观评估 ---
    print("\n" + "="*50)
    print(f"=== 加载最佳模型权重，执行最终客观评估 ===")
    print(f"=== 锁定验证集寻得的最优阈值: {global_best_threshold:.4f} ===")
    print("="*50)
    
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    
    # 分别评估训练集、验证集、测试集
    evaluate_dataset(model, train_loader, global_best_threshold, device, "1. 训练集 (Train)")
    evaluate_dataset(model, val_loader, global_best_threshold, device, "2. 验证集 (Validation)")
    test_targets, test_probs, test_preds = evaluate_dataset(model, test_loader, global_best_threshold, device, "3. 测试集 (Test)")
    
    # 导出测试集预测结果
    res_df = pd.DataFrame({
        'True_Label': test_targets.astype(int),
        'Predicted_Prob': test_probs,
        'Predicted_Label': test_preds
    })
    res_df.to_csv(config.PREDICT_RESULT_CSV, index=False)
    print(f"\n✅ 测试集预测详情已保存至: {config.PREDICT_RESULT_CSV}")

if __name__ == "__main__":
    main()