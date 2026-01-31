# 快速运行指南

## ✅ 问题已修复！

路径问题已解决，现在可以从**任何目录**运行脚本。

## 📁 项目位置

```
/workspace/ner_tab_entity/
```

## 🚀 快速开始

### 方式 1: 从项目目录运行（推荐）

```bash
cd /workspace/ner_tab_entity

# 演示推理
python demo_inference.py

# 交互式推理
python inference.py

# 测试模型
python test.py

# 重新训练
python train.py
```

### 方式 2: 从任意目录运行

```bash
# 从任何目录都可以运行
python /workspace/ner_tab_entity/demo_inference.py
python /workspace/ner_tab_entity/inference.py
python /workspace/ner_tab_entity/test.py
```

### 方式 3: 使用 VS Code 调试

直接在 VS Code 中打开文件并运行，路径会自动处理。

---

## 📊 脚本功能对比

| 脚本 | 功能 | 适用场景 |
|------|------|----------|
| **demo_inference.py** | 快速演示5个示例 | 快速查看效果 |
| **inference.py** | 交互式/批量推理 | 实际使用 |
| **test.py** | 完整测试和可视化 | 模型评估 |
| **train.py** | 训练模型 | 重新训练 |

---

## 📝 使用示例

### 1. 快速演示（推荐新手）

```bash
cd /workspace/ner_tab_entity
python demo_inference.py
```

**输出示例：**
```
【示例 1】
输入文本: 昨天张山购买了多少商品

字段名: 商品
时间: 昨天
维度: 张山
指标: 未识别到
```

### 2. 交互式推理

```bash
cd /workspace/ner_tab_entity
python inference.py
```

**使用方法：**
```
数据库字段属性实体识别 - 推理系统
============================================================
功能说明:
  - FIELD: 字段名 (如: 商品、价格)
  - TIME: 时间 (如: 昨天、2023-01-01)
  - METRIC: 指标 (如: 购买数量、销售额)
  - DIMENSION: 维度 (如: 张山、北京)

请输入问题（输入 'quit' 或 'exit' 退出）:
请输入问题: 昨天张山购买了多少商品
...
```

### 3. 批量推理

```bash
cd /workspace/ner_tab_entity

# 创建查询文件
cat > queries.txt << EOF
昨天张山购买了多少商品
本月北京的销售额是多少
2023年第三季度上海地区用户增长率
EOF

# 运行批量推理
python inference.py queries.txt results.json

# 查看结果
cat results.json
```

### 4. 完整测试

```bash
cd /workspace/ner_tab_entity
python test.py
```

**输出：**
```
============================================================
总体评估指标
============================================================
Accuracy:  0.8500
Precision: 0.7500
Recall:    0.8000
F1 Score:  0.7700

详细分类报告
              precision    recall  f1-score   support
FIELD         0.75      0.80      0.77        20
TIME           0.80      0.85      0.82        15
METRIC         0.70      0.75      0.72        18
DIMENSION      0.75      0.70      0.72        22
```

### 5. 可视化预测结果

```bash
cd /workspace/ner_tab_entity
python test.py
```

**自动显示前5个样本的真实标注vs预测结果**

---

## 🎯 实体类型说明

| 标签 | 说明 | 示例 |
|------|------|------|
| **FIELD** | 数据库字段名 | 商品、订单、价格 |
| **TIME** | 时间表达 | 昨天、本月、2023-01-01 |
| **METRIC** | 度量指标 | 购买数量、销售额、增长率 |
| **DIMENSION** | 查询维度 | 张山、北京、上海地区 |

---

## 🔧 重新训练模型

如果需要重新训练或提升性能：

```bash
cd /workspace/ner_tab_entity

# 删除旧模型（可选）
rm -rf models/best_model models/final_model

# 重新训练
python train.py
```

**训练时间：** 约 5-10 分钟（取决于硬件）

---

## 📦 模型文件位置

```
/workspace/ner_tab_entity/models/
├── best_model/          # 最佳模型（验证集上表现最好）
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── vocab.txt
│   └── ...
└── final_model/        # 最终模型（训练结束时的模型）
    ├── pytorch_model.bin
    ├── config.json
    ├── vocab.txt
    └── ...
```

---

## ✅ 验证环境

运行验证脚本检查环境是否正常：

```bash
cd /workspace/ner_tab_entity
python test_imports.py
```

**预期输出：**
```
============================================================
导入测试
============================================================

1. 测试基础库导入...
   ✓ torch, numpy

2. 测试transformers导入...
   ✓ BertTokenizer, BertConfig

...

✓ 所有测试通过！代码可以正常运行
```

---

## 🐛 常见问题

### Q1: 从 VS Code 运行报错"找不到文件"

**原因：** 工作目录不对

**解决：**
- 方式1：在终端中 `cd /workspace/ner_tab_entity` 后运行
- 方式2：修改 launch.json 设置 `cwd` 为 `/workspace/ner_tab_entity`
- 方式3：脚本已修复，现在可以从任何目录运行

### Q2: 模型加载失败

**检查清单：**
- [ ] 模型文件是否存在：`ls /workspace/ner_tab_entity/models/best_model`
- [ ] 如果不存在，先运行 `python train.py`
- [ ] 检查 CUDA 是否可用（如果使用 GPU）

### Q3: CUDA out of memory

**解决：**
- 减小批次大小（在 config.py 中修改 `BATCH_SIZE = 8`）
- 或使用 CPU（修改 `DEVICE = "cpu"`）

---

## 📚 文档索引

| 文档 | 内容 |
|------|------|
| **README.md** | 完整的项目文档 |
| **QUICK_START.md** | 快速开始指南 |
| **PROJECT_SUMMARY.md** | 项目总结和文件清单 |
| **TRAINED_MODEL_GUIDE.md** | 使用已训练模型的指南 |

---

## 🎉 开始使用

现在就可以开始使用了！推荐流程：

```bash
# 1. 进入项目目录
cd /workspace/ner_tab_entity

# 2. 查看演示效果
python demo_inference.py

# 3. 交互式测试
python inference.py

# 4. 查看性能指标
python test.py
```

**祝你使用愉快！** 🚀
