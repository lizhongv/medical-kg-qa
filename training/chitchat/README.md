# 闲聊意图分类(sklearn)

单文件训练脚本 `train_modern.py`:字符级 TF-IDF + LogisticRegression + GBDT 集成。
用于区分 greet / goodbye / deny / isbot 等闲聊意图。

## 安装
```bash
pip install scikit-learn numpy
```

## 训练
```bash
python train_modern.py
```
（数据已自带:`./data/intent_recog_data.txt`,每行「文本,标签」。）

## 产物
覆盖更新 `./model_file/{vec,LR,gbdt,id2label}.pkl`(兼容当前版本 sklearn)。

## 说明
- 重训目的:让 pickle 兼容新版 sklearn(原 0.24 版本的 pkl 在新版可能加载失败)。
- 预测时:`LR` 与 `GBDT` 的概率取平均(见 `clf_model.py`)。
