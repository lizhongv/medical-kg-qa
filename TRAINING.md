# 训练总说明(PyTorch 迁移)

三个模型的训练包,均可在 GPU(或 CPU 回退)运行。本文件只覆盖训练侧;
推理侧改造(app.py/modules.py 加载新权重)为后续单独步骤。

## 1. 意图识别(RBT3 + TextCNN)

```bash
cd nlu/bert_intent_recognition/pytorch
pip install -r requirements.txt
python download_base.py        # 下载 hfl/rbt3 基座(失败则训练时自动从 Hub 拉)
python train.py --epochs 5 --batch_size 32
# 产物: checkpoint/best_model.pt, label2id.json
```

## 2. NER(BiLSTM-CRF)

```bash
cd knowledge_extraction/bilstm_crf/pytorch
pip install -r requirements.txt
python download_data.py        # 下载 cMedQANER
python train.py --epochs 20 --batch_size 64
# 产物: checkpoint/best_ner.pt, vocab.json, tag2id.json
```

## 3. 闲聊分类(sklearn)

```bash
cd nlu/sklearn_Classification
python train_modern.py
# 产物: model_file/{vec,LR,gbdt,id2label}.pkl
```

## 冒烟测试(快速验证管线)

```bash
意图: python train.py --limit 100 --epochs 1
NER:  python train.py --limit 200 --epochs 1
```

## 训练完成后
把三组产物保留在各自 checkpoint/model_file 目录,等待推理侧改造对接。
