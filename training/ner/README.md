# 医疗 NER(BiLSTM-CRF · PyTorch)

单文件训练脚本 `train.py`,自包含:自动下载数据 → 字符级 BiLSTM-CRF → 训练 → 保存。

## 安装
```bash
pip install -r requirements.txt   # torch + pytorch-crf
```

## 训练
```bash
python train.py                   # 全量(首次运行自动下载 cMedQANER 到 ./data/)
python train.py --limit 200       # 冒烟(只用前 200 句)
```
可选参数:`--epochs`(默认 20)、`--batch_size`(默认 64)、`--lr`(默认 1e-3)。

## 产物
`./checkpoint/best_ner.pt`、`vocab.json`、`tag2id.json`

## 说明
- 数据:cMedQANER(ChineseBLUE),字符级 BIO,首次运行由 `train.py` 自动下载。
- 模型:随机初始化字向量从零训练,**无需基座预训练模型**。
- 实体类型:body / crowd / department / disease / drug / feature / physiology / symptom / test / time / treatment。
- 大数据/权重不入库(见 `.gitignore`)。
