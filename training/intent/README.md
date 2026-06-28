# 意图识别(BERT + TextCNN · PyTorch)

单文件训练脚本 `train.py`,自包含:读数据 → BERT 编码 + TextCNN → 训练 → 保存。

## 安装
```bash
pip install -r requirements.txt   # torch + transformers + pandas + numpy
```

## 基座模型
默认 `hfl/chinese-roberta-wwm-ext`(**12 层**,精度高)。可切换:
- **更快更小**:`python train.py --base hfl/rbt3`(3 层蒸馏版,精度略低)。
- **联网自动下载**:直接训练即可,首次从 Hub 拉取。
- **离线/手动**:把基座文件(`config.json`/`pytorch_model.bin`/`vocab.txt`)放到本目录 `base_model/`,脚本会优先用本地(忽略 `--base`)。

## 训练
```bash
python train.py                       # 全量(默认 wwm-ext,8 epoch)
python train.py --class_weight        # 开类别权重(治类别不均衡,提尾部类召回)
python train.py --base hfl/rbt3       # 用 3 层小模型
python train.py --limit 100 --epochs 1  # 冒烟
```
可选参数:`--base`、`--epochs`(默认 8)、`--batch_size`(默认 32)、`--lr`(默认 2e-5)、`--warmup_ratio`(默认 0.1)、`--class_weight`。

## 数据
读取 `../data/{train,test}.csv`(仓库已自带,字段 `text,label_class,label`,13 类)。

## 产物
`./checkpoint/best_model.pt`、`label2id.json`、`meta.json`

## 说明
- 训练用了 **warmup + 线性衰减** 学习率调度(BERT 微调标配)。
- `meta.json` 记录所用基座 `base`,**推理侧(`app/nlu/intent.py`)会读它用相同结构加载** —— 换基座后无需改推理代码,但 `best_model.pt` 必须和 `meta.json` 配套使用。
- 基座权重与训练产物均不入库(见 `.gitignore`)。
- 调精度建议:默认(12 层 + 调度)通常比 3 层 rbt3 显著更高;若尾部类(传染性/治疗时间等)召回低,加 `--class_weight`。
