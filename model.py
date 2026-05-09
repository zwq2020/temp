# model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class ChronosForQoE(nn.Module):
    def __init__(self, model_name, burst_dim=6, freeze_llm=True):
        super(ChronosForQoE, self).__init__()
        
        print(f"正在加载大模型结构与预训练权重: {model_name} ...")
        self.config = AutoConfig.from_pretrained(model_name)
        self.llm_encoder = AutoModel.from_pretrained(model_name).encoder
        
        # 1. 【核心策略：绝对冻结大模型！】
        # 只有冻结骨干，才能保住大模型抵抗数据漂移的通用泛化能力
        for param in self.llm_encoder.parameters():
            param.requires_grad = False
        print("✅ 已严格冻结大模型骨干网络参数，启动 Adapter 适配器模式。")
                
        # 2. 【核心升级：多层非线性适配器 (Adapter)】
        # 替代原本单薄的 Linear，用一个小网络把物理特征优雅地“翻译”成大模型能懂的 Token 语义
        self.input_projection = nn.Sequential(
            nn.Linear(burst_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.config.d_model)
        )
        
        # 3. 健壮的分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.config.d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.4), # 加大 Dropout 强效防死记硬背
            nn.Linear(64, 1)
        )

    def forward(self, bursts_sequence, attention_mask):
        # 4. 【核心策略：时序特征随机遮挡 (Sequence Dropout)】
        # 训练时，随机把 30% 的时间窗口直接变黑(抹零)。
        # 逼迫大模型用剩下的 70% 历史去“推理”局势，彻底打碎它的死记硬背机制！
        if self.training:
            # 生成 0 或 1 的掩码，保留概率 70%
            drop_mask = (torch.rand(bursts_sequence.shape[:-1], device=bursts_sequence.device) > 0.3).float()
            bursts_sequence = bursts_sequence * drop_mask.unsqueeze(-1)

        inputs_embeds = self.input_projection(bursts_sequence)
        
        # 喂给大模型 (它现在看到的是被翻译好且部分遮挡的高级特征)
        outputs = self.llm_encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state 
        
        # 提取最后一个时刻的 Token 进行判决
        pooled_output = hidden_states[:, -1, :] 
        
        logits = self.classifier(pooled_output)
        return logits