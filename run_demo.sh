#!/bin/bash

# 数据库字段属性实体识别模型 - 快速开始脚本

echo "============================================================"
echo "数据库字段属性实体识别模型 - 快速开始"
echo "============================================================"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

echo ""
echo "步骤 1/4: 安装依赖包..."
echo "------------------------------------------------------------"
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "错误: 依赖安装失败"
    exit 1
fi

echo ""
echo "步骤 2/4: 训练模型..."
echo "------------------------------------------------------------"
python train.py

if [ $? -ne 0 ]; then
    echo "错误: 训练失败"
    exit 1
fi

echo ""
echo "步骤 3/4: 测试模型..."
echo "------------------------------------------------------------"
python test.py

echo ""
echo "步骤 4/4: 运行推理示例..."
echo "------------------------------------------------------------"
echo "请选择推理模式："
echo "1. 交互式推理"
echo "2. 批量推理"
read -p "请输入选项 (1 或 2): " choice

if [ "$choice" = "1" ]; then
    python inference.py
elif [ "$choice" = "2" ]; then
    # 创建测试查询文件
    cat > test_queries.txt << EOF
昨天张山购买了多少商品
本月北京的销售额是多少
2023年第三季度上海地区用户增长率
上周五李四购买了哪些产品
本月杭州地区的订单数量和销售额
EOF
    python inference.py test_queries.txt test_results.json
    echo ""
    echo "批量推理结果已保存到 test_results.json"
else
    echo "无效选项"
fi

echo ""
echo "============================================================"
echo "快速开始完成！"
echo "============================================================"
echo ""
echo "您可以："
echo "- 运行 python train.py 重新训练模型"
echo "- 运行 python test.py 测试模型性能"
echo "- 运行 python inference.py 进行交互式推理"
echo "- 运行 python inference.py input.txt output.json 进行批量推理"
echo ""
