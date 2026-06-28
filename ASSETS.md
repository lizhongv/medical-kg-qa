# 未入库资产说明(大文件 / 权重 / 密钥)

为控制仓库体积、规避 GitHub 100MB/文件限制、避免密钥泄露,以下文件**不提交到 git**(已在 `.gitignore` 中)。
clone 本仓库后需按下表自行获取或生成,放到对应目录,系统才能完整运行。

## 一、未入库清单

| 文件 | 大小 | 不入库原因 | 如何获取 / 恢复 |
|---|---|---|---|
| `training/intent/checkpoint/best_model.pt` | **403 MB** | 超 GitHub 100MB/文件硬限制 | 从发布的权重包(zip / Release / 网盘)取,放回该目录 |
| `kg/medical.json` | **45 MB** | 文件大,且可重新下载 | 从源仓库下载(见下),放到 `kg/` |
| `training/ner/checkpoint/best_ner.pt` | 1.6 MB | 训练生成物;在线服务用词典分词,不依赖它 | `cd training/ner && python train.py` 重新训练 |
| `training/ner/data/{train,dev,test}.txt`(cMedQANER) | ~1.5 MB | 公开数据集,训练时自动下载 | NER 首次训练时自动拉取(`train.py` 内置) |
| `run_service.bat` | — | **含 LLM API 密钥**,不可入库 | 按 README「第 3 步」自建,填入你的密钥 |

> 已入库的小元数据:`label2id.json`、`meta.json`(意图)、`vocab.json`、`tag2id.json`(NER)——它们是用模型必需的标签/基座信息,体积小,已随代码提交。

## 二、获取方式

### best_model.pt(意图权重,403MB)
无法走 GitHub 普通提交,选其一分发:
- **GitHub Release** 附件(单文件可达 2GB);
- **网盘 / 对象存储**;
- **Git LFS**(若要纳入版本管理)。

放置位置:`training/intent/checkpoint/best_model.pt`
(同目录的 `label2id.json` / `meta.json` 已在仓库中,`meta.json` 记录了基座 `hfl/chinese-roberta-wwm-ext`,推理侧据此对齐加载。)

### medical.json(知识图谱源数据,45MB)
来自经典中文医疗结构化数据集(约 8808 种疾病)。可从公开仓库获取,例如:
```
https://raw.githubusercontent.com/liuhuanyong/QASystemOnMedicalKG/master/data/medical.json
```
放置位置:`kg/medical.json`,然后:
```bash
cd kg && python import_kg.py     # 导入 Neo4j
```

### run_service.bat(启动脚本 + 密钥)
参考 README「四、快速开始 / 第 3 步」自建,内容形如:
```bat
@echo off
set KBQA_LLM_API_KEY=你的密钥
set KBQA_LLM_BASE_URL=https://api.deepseek.com
set KBQA_LLM_MODEL=deepseek-chat
set KBQA_NEO4J_URI=bolt://127.0.0.1:7687
set KBQA_NEO4J_USER=neo4j
set KBQA_NEO4J_PASSWORD=你的Neo4j密码
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

## 三、全新 clone 后的最小恢复步骤

1. `pip install -r requirements.txt`
2. 放入 `best_model.pt` → `training/intent/checkpoint/`(意图模型可用;否则诊断问句自动降级走 LLM 慢路)
3. 放入 `medical.json` → `kg/`,启动 Neo4j 后 `python kg/import_kg.py`
4. 自建 `run_service.bat`(填密钥)→ 运行,启动服务
5. (可选)`python training/ner/train.py` 重训 NER

> 仅 **best_model.pt** 与 **medical.json** 必须"另外搞到";其余要么已入库,要么训练/导入时自动生成。
