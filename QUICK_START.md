# 快速使用指南

## 项目位置

所有代码已保存在: `/workspace/ner_tab_entity/`

## 文件结构

```
ner_tab_entity/
├── config.py              # 配置文件（模型参数、标签定义等）
├── model.py               # BERT NER模型定义
├── data_processor.py      # 数据处理（数据加载、标签对齐等）
├── train.py               # 训练脚本
├── test.py                # 测试脚本
├── inference.py           # 推理脚本（交互式/批量）
├── utils.py               # 工具函数
├── dataset/               # 数据集目录
│   ├── train.txt          # 训练数据（19个句子）
│   ├── dev.txt            # 验证数据（14个句子）
│   └── test.txt           # 测试数据（14个句子）
├── models/                # 模型保存目录（训练后生成）
├── requirements.txt       # Python依赖
├── README.md              # 详细文档
├── run_demo.sh            # 快速开始脚本
├── verify_setup.py        # 环境验证脚本
└── demo_test.py           # 演示脚本
```

## 功能说明

### 1. 实体识别功能

模型可以识别以下四种实体类型：

| 实体类型 | 标签 | 示例 | 说明 |
|---------|------|------|------|
| 字段名 | FIELD | 商品、订单、价格 | 数据库字段名 |
| 时间 | TIME | 昨天、本月、2023-01-01 | 时间表达 |
| 指标 | METRIC | 购买数量、销售额、增长率 | 度量指标 |
| 维度 | DIMENSION | 张山、北京、上海地区 | 查询维度 |

### 2. 示例查询

```
输入: 昨天 张山 购买了 多少 商品

输出:
- 时间: 昨天
- 维度: 张山
- 字段名: 商品
- 指标: 未识别到
```

## 快速开始

### 步骤1: 安装依赖

```bash
cd /workspace/ner_tab_entity
pip install -r requirements.txt
```

### 步骤2: 训练模型

```bash
python train.py
```

训练过程会：
- 自动下载bert-base-chinese模型（首次运行）
- 训练10个epoch
- 在验证集上评估
- 保存最佳模型到 `models/best_model/`
- 保存最终模型到 `models/final_model/`

### 步骤3: 测试模型

```bash
python test.py
```

测试脚本会：
- 加载最佳模型
- 在测试集上评估性能
- 显示各项指标（准确率、精确率、召回率、F1）
- 生成详细分类报告
- 可视化部分预测结果

### 步骤4: 使用推理

#### 交互式推理

```bash
python inference.py
```

运行后可以输入自然语言查询，系统会实时显示识别结果。

示例输入：
```
昨天张山购买了多少商品
本月北京的销售额是多少
```

#### 批量推理

```bash
# 创建查询文件
cat > queries.txt << EOF
昨天张山购买了多少商品
本月北京的销售额是多少
2023年第三季度上海地区用户增长率
EOF

# 运行批量推理
python inference.py queries.txt results.json
```

结果会保存为JSON格式。

## 一键运行

使用快速开始脚本（包含所有步骤）：

```bash
bash run_demo.sh
```

## 数据集说明

### 训练数据格式

采用BIO标注格式，每行一个字符：

```
昨    B-TIME
天    I-TIME
张    B-DIMENSION
山    I-DIMENSION
购    O
买    O
了    O
多    O
少    O
商    B-FIELD
品    I-FIELD

(空行表示句子分隔)
```

### 数据集统计

| 数据集 | 句子数 | 总行数 |
|--------|--------|--------|
| 训练集 | 19 | 261 |
| 验证集 | 14 | 193 |
| 测试集 | 14 | 200 |

### 数据覆盖范围

- ✅ 相对时间（昨天、今天、本月、上周五等）
- ✅ 绝对时间（2023-01-01、2023年第三季度等）
- ✅ 人员维度（张山、李四、王五等）
- ✅ 地区维度（北京、上海、杭州等）
- ✅ 企业维度（中国移动、中国联通等）
- ✅ 字段实体（商品、订单、产品、用户等）
- ✅ 指标实体（购买数量、销售额、增长率等）

## 配置参数

主要参数在 `config.py` 中：

```python
MODEL_NAME = "bert-base-chinese"    # BERT模型
MAX_LENGTH = 128                    # 最大序列长度
BATCH_SIZE = 16                     # 批次大小
LEARNING_RATE = 2e-5                # 学习率
NUM_EPOCHS = 10                     # 训练轮数
```

可根据需要修改这些参数。

## 验证环境

运行环境验证脚本：

```bash
python verify_setup.py
```

会检查：
- Python版本
- 依赖包安装情况
- 文件完整性
- 数据集统计
- 配置参数

## 输出说明

### 训练输出

```
models/
├── best_model/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── vocab.txt
└── final_model/
    └── ...
```

### 测试输出

```
总体评估指标:
- Accuracy:  准确率
- Precision: 精确率
- Recall:    召回率
- F1 Score:  F1分数

详细分类报告:
- 每种实体类型的精确率、召回率、F1

test_results.json:
- JSON格式的完整测试结果
```

## 常见问题

### Q1: 训练时内存不足怎么办？

A: 减小BATCH_SIZE，在config.py中修改：
```python
BATCH_SIZE = 8  # 或更小
```

### Q2: 如何使用自己的数据集？

A: 按照BIO格式准备数据，修改config.py中的路径：
```python
TRAIN_FILE = "your_train.txt"
DEV_FILE = "your_dev.txt"
TEST_FILE = "your_test.txt"
```

### Q3: 预测结果不准确怎么办？

A: 可以尝试：
1. 增加训练数据量
2. 调整学习率
3. 增加训练轮数
4. 使用更大的模型（如BERT-Large）

### Q4: 如何在新数据上使用？

A: 使用inference.py：
```bash
# 交互式
python inference.py

# 批量
python inference.py input.txt output.json
```

## 评估指标说明

使用seqeval库计算序列级指标：

- **Accuracy**: 整体准确率
- **Precision**: 精确率（预测为正的样本中有多少是真的正样本）
- **Recall**: 召回率（真正的正样本中有多少被预测为正样本）
- **F1 Score**: 精确率和召回率的调和平均

指标按整体和各实体类型分别计算。

## 扩展建议

1. **数据增强**: 增加训练数据量，提高模型泛化能力
2. **模型升级**: 使用BERT-Large或RoBERTa等更强大的模型
3. **领域适配**: 使用特定领域的预训练模型
4. **后处理**: 添加规则后处理提高时间标准化准确性
5. **性能优化**: 模型量化和推理加速

## 项目特点

✅ 完整的训练-测试-推理流程
✅ 标准化的BIO标注格式
✅ 灵活的配置系统
✅ 详细的评估指标
✅ 交互式和批量推理
✅ 完善的文档和示例
✅ 时间标准化功能
✅ 可视化预测结果

## 联系支持

如有问题，请参考：
1. README.md - 详细文档
2. PROJECT_SUMMARY.md - 项目总览
3. verify_setup.py - 环境检查

## 许可证

MIT License
