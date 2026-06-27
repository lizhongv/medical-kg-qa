# 意图识别(RBT3 + TextCNN · PyTorch)

单文件训练脚本 `train.py`,自包含:读数据 → RBT3 编码 + TextCNN → 训练 → 保存。

## 安装
```bash
pip install -r requirements.txt   # torch + transformers + pandas
```

## 基座模型
脚本用 `hfl/rbt3`(3 层 RoBERTa-wwm-ext)。两种方式任选其一:
- **联网自动下载**:直接训练即可,`train.py` 会从 Hub 拉取 `hfl/rbt3`。
- **离线/手动**:把 `hfl/rbt3` 的文件(`config.json`/`pytorch_model.bin`/`vocab.txt`)放到本目录的 `base_model/`,脚本会优先用本地。

## 训练
```bash
python train.py                   # 全量
python train.py --limit 100       # 冒烟
```
可选参数:`--epochs`(默认 5)、`--batch_size`(默认 32)、`--lr`(默认 2e-5)。

## 数据
读取 `../data/{train,test}.csv`(仓库已自带,字段 `text,label_class,label`,13 类)。

## 产物
`./checkpoint/best_model.pt`、`label2id.json`

## 说明
- 基座权重与训练产物均不入库(见 `.gitignore`)。
