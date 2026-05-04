# SVM-RDKit Descriptor Predictor

基于 RDKit 分子描述符和预训练 SVM 模型，对 SMILES 分子进行活性预测。

## 环境要求

- Python ≥ 3.8
- 依赖包：

```bash
pip install numpy pandas scikit-learn==1.3.2 rdkit
```

> **注意：** scikit-learn 版本应与训练模型时一致，否则 pickle 加载可能报错。

## 文件说明

```
├── predict.py                  # 预测脚本
├── SVMRDKitDesc.pickle         # 预训练 SVM 模型
├── descriptor_columns.json     # 描述符列名列表（训练时固定的 208 个描述符）
├── smile.csv                   # 示例输入文件
└── README.md
```

## 输入格式

CSV 文件，**第一列**为 SMILES 字符串，示例：

```csv
SMILES,name
CCO,ethanol
c1ccccc1,benzene
```

## 快速使用

```bash
# 使用默认文件名
python predict.py

# 指定输入/输出
python predict.py -i my_molecules.csv -o my_results.csv

# 指定模型和描述符文件
python predict.py -i input.csv -o output.csv -m model.pickle -d descriptors.json
```

### 命令行参数

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | `smile.csv` | 输入 CSV 文件路径 |
| `--output` | `-o` | `result.csv` | 输出 CSV 文件路径 |
| `--model` | `-m` | `SVMRDKitDesc.pickle` | 模型文件路径 |
| `--descriptors` | `-d` | `descriptor_columns.json` | 描述符列名 JSON 文件路径 |

## 输出格式

在原始数据基础上新增两列：

| 列名 | 说明 |
|------|------|
| `predict` | 预测类别（0 或 1） |
| `predict_proba` | 预测为正类的概率 |

含有无效描述符的分子会被自动剔除，并在终端输出被剔除的行索引。


