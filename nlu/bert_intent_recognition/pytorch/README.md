# 意图识别(RBT3 + TextCNN)PyTorch 训练包

## 安装

```bash
pip install -r requirements.txt
```

## 基座模型

```bash
python download_base.py   # 尝试下载 hfl/rbt3 到 ./base_model/
# 若失败,train.py 会自动用 Hub 名 'hfl/rbt3'(需联网)
```

## 训练

```bash
python train.py --epochs 5 --batch_size 32         # 全量
python train.py --limit 100 --epochs 1             # 冒烟
```

## 产物
checkpoint/best_model.pt, checkpoint/label2id.json

## 数据
读取 ../data/{train,test}.csv(仓库已自带,字段 text,label_class,label)

运行测试请在仓库根目录执行 `python -m pytest <路径>`。
