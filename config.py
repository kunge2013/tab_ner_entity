# 配置文件
import os

# 获取脚本所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    # 模型配置
    MODEL_NAME = "bert-base-chinese"  # 使用中文BERT模型
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 50  

    # 数据配置（使用绝对路径）
    TRAIN_FILE = os.path.join(BASE_DIR, "dataset/train.txt")
    DEV_FILE = os.path.join(BASE_DIR, "dataset/dev.txt")
    TEST_FILE = os.path.join(BASE_DIR, "dataset/test.txt")
    OUTPUT_DIR = os.path.join(BASE_DIR, "models")
    
    # 标签体系
    LABELS = ['O', 'B-FIELD', 'I-FIELD', 'B-TIME', 'I-TIME', 
              'B-METRIC', 'I-METRIC', 'B-DIMENSION', 'I-DIMENSION']
    NUM_LABELS = len(LABELS)
    LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
    ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}
    
    # 训练配置
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1.0
    SAVE_STEPS = 500
    LOGGING_STEPS = 50
    
    # 设备配置
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    NUM_WORKERS = 0  # Windows兼容性设置为0
    
    # 随机种子
    SEED = 42
