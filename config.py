# config.py
import os


# 路径配置
RAW_PCAP_CSV = "./pcap/N3_Facebook_188_0909_100526.csv"    # tshark 导出的原始包数据
LABEL_CSV = "./label/N3_Facebook_188_20250909_100526.csv"        # 包含 timestamp 和 label 的每秒真实标签表
PROCESSED_DATA_DIR = "./processed_data"   # 预处理后矩阵的保存目录
BEST_MODEL_PATH = "./weight/best_chronos_qoe.pth"  # 最佳模型权重保存路径
PREDICT_RESULT_CSV = "./result/test_predictions.csv" # 最终测试集预测结果
LOSS_CURVE_PATH = "./result/training_loss_curve.png" # 新增：Loss 曲线保存路径

# 流量特征提取超参数
BURST_THRESHOLD_SEC = 0.05  # 划分 Burst 的时间间隔阈值 (50ms)
WINDOW_SEC = 40           # 每次预测回看的时间窗口长度 (15秒)
MAX_BURSTS = 200             # 强制对齐的最大 Burst 序列长度

# 模型超参数
LLM_MODEL_NAME = "./chronos-t5-small" # 调用的开源时序大模型
BURST_FEAT_DIM = 6                         # 提取的物理特征维度
FREEZE_LLM = False                         # 是否冻结大模型骨干参数

# 训练超参数
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE_LLM = 2e-5    # 大模型骨干的温柔学习率
LEARNING_RATE_HEAD = 1e-3   # 分类头的冲刺学习率
PATIENCE = 10                # 早停容忍度 (基于 F1 分数)

# 预处理阶段的标签修正
MIN_STALL_LEN = 3        # 持续短于2秒的卡顿记为流畅
MIN_SMOOTH_LEN = 4       # 持续短于3秒的流畅记为卡顿
SHIFT_SEC = 1            # 标签前移时间(秒)，用于提前预测

# 评估阶段的平滑与容忍度
PROB_SMOOTH_WINDOW = 3   # 预测概率的滑动平均窗口大小(秒)
MIN_DELAY = 3            # 切换点前后免责的窗口大小(秒)
TOLERANCE = 3            # 只要前后容忍度内有对应标签即算正确的窗口(秒)

# 创建数据保存目录
if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)