# 设计文档:三个模型的 PyTorch 训练包迁移

- 日期:2026-06-27
- 范围:**仅训练侧**(数据 + 训练脚本 + 基座模型),供用户在 GPU 环境训练
- 不在范围:推理侧改造(`app.py` / `modules.py` 加载新权重)——训练产出后单独立项

## 背景与动机

原项目的两个核心模型绑定了过时技术栈(`bert4keras` / `tensorflow==1.14` / `Keras==2.3.1`),无法在现代 Python(3.10+)运行,且现有权重为框架专有格式、含自定义层,转换不划算。由于三个模型的训练数据均可获得(意图、闲聊数据在仓库内;NER 数据为公开 cMedQANER),决定**用 PyTorch 重新训练**,彻底摆脱 TF。

需要 TF 的仅有两个服务:意图识别(`nlu/bert_intent_recognition/`)、NER(`knowledge_extraction/bilstm_crf/`)。闲聊分类是 sklearn,不绑 TF,但因 pickle 跨版本(0.24→新版)风险一并重训。

## 总体原则

1. 新代码放进各模型下的 `pytorch/` 子目录,**不修改现有 TF 代码**(保留作对照/回退)。
2. 训练脚本设备无关:`device = 'cuda' if torch.cuda.is_available() else 'cpu'`。
3. 每个训练包自带 `requirements.txt` 与 `README.md`(训练命令、产物路径、依赖)。
4. 复刻原模型架构,保证后续推理改造时行为对齐。

## 模型 1:意图识别(BERT + TextCNN,13 类)

- **数据**:`nlu/bert_intent_recognition/data/{train,test}.csv`(已在仓库,7274 / 810 条;字段 `text,label_class,label`,label 为 0–12)。脚本直接读,无需下载。
- **基座模型**:`hfl/rbt3`(3 层 RoBERTa-wwm-ext)。
  - 风险:本地代理封锁 HuggingFace 大文件 CDN,`pytorch_model.bin`(~150MB)可能无法预先下载。
  - 对策:训练脚本优先读本地 `base_model/`,否则回退 `from_pretrained('hfl/rbt3')` 在 GPU 机器联网自动下载;另附 `download_base.py` 辅助脚本。
- **架构(复刻原版 `bert_model.py`)**:
  - RBT3 编码器 → `last_hidden_state`
  - CLS 特征 = `last_hidden_state[:, 0]`
  - TextCNN 特征 = 对 `last_hidden_state[:, 1:-1]` 做 3 路 `Conv1d`(256 通道,核 3/4/5)+ `GlobalMaxPool` 后 concat
  - `concat(cls, textcnn)` → `Linear(→512, ReLU)` → `Dropout` → `Linear(→13)`
  - 损失:`CrossEntropyLoss`(原版 softmax + 多分类等价)
- **产物**:`best_model.pt`(state_dict)+ `label2id.json`。

## 模型 2:NER(BiLSTM-CRF,字符级)

- **数据**:公开 **cMedQANER**(ChineseBLUE),从 GitHub raw 下载至 `pytorch/data/`(GitHub 通道已验证可用)。实现时确认其 BIO / 列格式并适配 loader。
- **基座模型**:**无**。字符 Embedding 随机初始化、从零训练(与原版一致)。README 注明此模型无"初始 model"。
- **架构**:字符 `Embedding` → `BiLSTM` → `Linear` → `CRF`(依赖 `pytorch-crf`,`import torchcrf`)。
  - 词表(`vocab.json`)与标签集(`tag2id.json`,BIO)由训练数据动态构建。
- **产物**:`best_ner.pt`(state_dict)+ `vocab.json` + `tag2id.json`。

## 模型 3:闲聊分类(sklearn,重训)

- **数据**:`nlu/sklearn_Classification/data/intent_recog_data.txt`(已在仓库,207 条;`文本<制表符>标签`)。无需下载。
- **说明**:不依赖 TF;重训仅为兼容新版 sklearn 的 pickle。沿用原架构:字符级 `TfidfVectorizer` + `LogisticRegression` + `GradientBoostingClassifier`,预测时两者概率平均。
- **产物**:覆盖式更新 `model_file/{vec.pkl, LR.pkl, gbdt.pkl, id2label.pkl}`。

## 目录布局

```
nlu/bert_intent_recognition/pytorch/
    model.py            # RBT3 + TextCNN(PyTorch)
    train.py            # 训练 + 验证 + 保存 best_model.pt
    predict.py          # 单条推理自检
    requirements.txt
    README.md
    base_model/         # hfl/rbt3 本地文件(若可下载)
    download_base.py    # 基座模型下载辅助

knowledge_extraction/bilstm_crf/pytorch/
    data/               # cMedQANER train/dev/test
    model.py            # BiLSTM-CRF(PyTorch)
    train.py
    predict.py
    requirements.txt
    README.md

nlu/sklearn_Classification/
    train_modern.py     # 现代 sklearn 重训脚本(数据已就位)

TRAINING.md             # 顶层总说明:三个模型如何训练、产物位置、后续推理改造预告
```

## 验收标准

1. 三个 `train.py` / `train_modern.py` 在装好各自 `requirements.txt` 后可在 GPU(及 CPU 回退)上启动训练,无导入/路径错误。
2. 意图与 NER 训练脚本能正确加载各自数据并跑通至少 1 个 epoch、保存产物。
3. cMedQANER 数据已下载并放入 NER `data/`,loader 与其格式对齐。
4. 每个训练包的 `README.md` 写明:依赖安装、训练命令、产物文件、基座模型获取方式。
5. `TRAINING.md` 串起三者并预告后续推理改造。

## 风险与对策

- **基座模型下载受阻**:见模型 1 对策(本地优先 + Hub 回退 + 辅助脚本)。
- **cMedQANER 格式未知**:实现阶段先下载样本核对列/分隔/BIO 风格,再定 loader;若结构与预期差异大,在 README 记录适配说明。
- **CPU 本机无法实测完整训练**:本机仅验证脚本可启动、前向/数据管线无误(小步快验);完整收敛由用户在 GPU 环境完成。
```
