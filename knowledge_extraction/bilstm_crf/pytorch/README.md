# NER(BiLSTM-CRF)PyTorch 训练包

## 安装

```bash
pip install -r requirements.txt
```

## 数据

```bash
python download_data.py   # 下载 cMedQANER 到 ./data/
```

## 训练

```bash
python train.py --epochs 20 --batch_size 64        # 全量
python train.py --limit 200 --epochs 1             # 冒烟
```

## 产物
checkpoint/best_ner.pt, vocab.json, tag2id.json

## 说明
本模型用随机初始化字向量从零训练,无需基座预训练模型。
实体类型:body/crowd/department/disease/drug/feature/physiology/symptom/test/time/treatment。

运行测试请在仓库根目录执行 `python -m pytest <路径>`。
