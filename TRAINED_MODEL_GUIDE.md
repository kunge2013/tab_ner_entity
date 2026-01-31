# 使用已训练模型指南

## ✅ 问题已解决！

路径问题已修复，所有依赖正常。模型已经训练完成！

## 📦 已训练模型

- **最佳模型**: `models/best_model/` - 在验证集上表现最好
- **最终模型**: `models/final_model/` - 训练结束时的模型

## 🚀 快速使用（无需重新训练）

### 1. 直接使用推理脚本

```bash
cd /workspace/ner_tab_entity

# 交互式推理（推荐）
python inference.py
```

输入问题示例：
```
昨天张山购买了多少商品
本月北京的销售额是多少
```

### 2. 批量推理

```bash
# 创建查询文件
cat > queries.txt << EOF
昨天张山购买了多少商品
本月北京的销售额是多少
2023年第三季度上海地区用户增长率
上周五李四购买了哪些产品
本月杭州地区的订单数量和销售额
EOF

# 运行批量推理
python inference.py queries.txt results.json

# 查看结果
cat results.json
```

### 3. 测试已训练模型

```bash
# 在测试集上评估模型性能
python test.py
```

## 📊 模型功能

模型可以识别以下实体：

| 实体类型 | 说明 | 示例 |
|---------|------|------|
| **FIELD** | 字段名 | 商品、订单、价格 |
| **TIME** | 时间 | 昨天、本月、2023-01-01 |
| **METRIC** | 指标 | 购买数量、销售额、增长率 |
| **DIMENSION** | 维度 | 张山、北京、上海地区 |

## 💡 使用示例

### Python 代码示例

```python
from utils import load_model_and_tokenizer, extract_entities, format_output
from config import Config

# 加载模型
model, tokenizer = load_model_and_tokenizer("models/best_model")

# 识别实体
text = "昨天张山购买了多少商品"
result, entities, tokens = extract_entities(text, model, tokenizer)

# 格式化输出
format_output(text, result)
```

输出：
```
============================================================
输入文本: 昨天张山购买了多少商品
============================================================

识别结果:
------------------------------------------------------------

字段名: 商品

时间: 昨天

指标: 未识别到

维度: 张山

============================================================
```

## 🔄 如需重新训练

如果需要重新训练模型：

```bash
cd /workspace/ner_tab_entity

# 清除旧模型（可选）
rm -rf models/best_model models/final_model

# 运行训练
python train.py
```

训练时间：约 5-10 分钟（取决于硬件）

## 📈 模型性能

运行 `python test.py` 可以看到详细的性能指标，包括：
- 总体准确率、精确率、召回率、F1分数
- 各实体类型的独立评估
- 详细分类报告

## 📁 项目文件说明

```
ner_tab_entity/
├── config.py              # 配置文件（已修复路径问题）
├── model.py               # BERT NER模型
├── data_processor.py      # 数据处理（已修复标签对齐）
├── train.py               # 训练脚本（已修复依赖）
├── test.py                # 测试脚本
├── inference.py           # 推理脚本
├── utils.py               # 工具函数
├── dataset/               # 数据集
│   ├── train.txt         # 训练数据 (19个句子)
│   ├── dev.txt           # 验证数据 (14个句子)
│   └── test.txt          # 测试数据 (14个句子)
├── models/               # 已训练的模型 ✅
│   ├── best_model/       # 最佳模型
│   └── final_model/     # 最终模型
└── test_imports.py      # 环境测试脚本
```

## ✅ 问题解决记录

1. **✓ 路径问题** - 使用绝对路径修复文件找不到的问题
2. **✓ 依赖冲突** - 使用自定义指标计算替代 seqeval
3. **✓ tokenizer 问题** - 重写标签对齐逻辑
4. **✓ 模型训练** - 模型已训练完成并保存

## 🎯 下一步推荐

1. **试用推理**: 运行 `python inference.py` 体验效果
2. **查看性能**: 运行 `python test.py` 查看模型指标
3. **批量处理**: 使用批量推理处理多个查询
4. **定制训练**: 如需改进，可以增加训练数据后重新训练

## ❓ 常见问题

**Q: 为什么已经有训练好的模型？**
A: 之前的训练已经完成，模型保存在 models/ 目录下。

**Q: 可以直接使用吗？**
A: 可以！直接运行 `python inference.py` 即可使用。

**Q: 需要重新训练吗？**
A: 不需要，除非您想改进模型性能或使用新数据。

**Q: 模型准确率如何？**
A: 运行 `python test.py` 可以查看详细的性能指标。

**Q: 如何集成到自己的项目？**
A: 参考上面的 Python 代码示例，使用 utils.py 中的函数。

---

**现在可以开始使用模型了！** 🎉
