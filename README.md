# END4Rec（MindSpore）重构版

本仓库已完成以下基础整理：
- 已将 `End4Rec.zip` 解压为源码目录 `End4Rec/`；
- 清理了 macOS 打包冗余文件（`__MACOSX`、`.DS_Store`）；
- 重构训练入口、训练流程、损失函数与数据加载逻辑；
- 补充了可执行的项目说明文档与引用信息。

---

## 1. 项目简介

END4Rec 是一个多行为序列推荐模型，核心思想是通过噪声解耦机制（硬噪声 + 软噪声）提升表示质量。当前实现基于 MindSpore。

## 2. 重构内容概览

### 2.1 工程可运行性修复
- 统一了训练入口：`run.py -> train.py`；
- 删除了大量未启用/不可运行的注释旧逻辑；
- 修复了原损失函数中对不存在属性（如 `item_embeddings`）的依赖。

### 2.2 训练流程重构
- 实现 4-stage 训练调度：
  1) `embedding + ebm`
  2) `hard_noise`
  3) `soft_noise`
  4) 全参数联合微调
- 提供统一 `fit_end4rec()` 与 `train_stage()`。

### 2.3 数据集加载重构
- 支持两种训练模式：
  - CSV 数据文件（含 `item_ids/behavior_ids/positions/label`）；
  - 无数据文件时自动使用 dummy 数据快速验证流程。

### 2.4 文档完善
- 提供详细配置说明、运行步骤、常见问题与引用信息。

---

## 3. 环境准备

> 建议：Linux + Python 3.9/3.10 + MindSpore GPU 版本。

1. 创建环境并安装依赖（按你机器上的 MindSpore 版本匹配）。
2. 保证 `python` 能正确导入 `mindspore`。

示例：

```bash
cd End4Rec
python -c "import mindspore; print(mindspore.__version__)"
```

---

## 4. 数据格式

训练 CSV 需包含以下列：
- `item_ids`：逗号分隔的 item 序列
- `behavior_ids`：逗号分隔的行为类型序列
- `positions`：位置编码序列
- `label`：目标 item id

示例：

```csv
item_ids,behavior_ids,positions,label
10,0,0,42
```

> 注意：真实训练时每一列应为等长序列，如 `10,11,23,...`。

---

## 5. 使用步骤

### 5.1 使用默认配置（快速 smoke test）

```bash
cd End4Rec
python run.py
```

默认会自动使用 dummy 数据，执行完整 4-stage 训练，并保存 `end4rec_final.ckpt`。

### 5.2 使用自定义配置

1) 新建配置文件 `config.json`：

```json
{
  "num_items": 10000,
  "num_behaviors": 4,
  "seq_length": 50,
  "d_model": 128,
  "num_blocks": 4,
  "epsilon": 0.001,
  "batch_size": 64,
  "train_data_file": "./train.csv",
  "eval_data_file": "./eval.csv",
  "topk": 10,
  "stage1_epochs": 1,
  "stage2_epochs": 1,
  "stage3_epochs": 1,
  "stage4_epochs": 1,
  "learning_rate_stage1": 0.001,
  "learning_rate_stage2": 0.001,
  "learning_rate_stage3": 0.001,
  "learning_rate_stage4": 0.0005,
  "reg_weight": 0.01,
  "contrast_weight": 0.1,
  "num_dummy_samples": 512
}
```

2) 运行：

```bash
cd End4Rec
python run.py --config config.json
```

---

## 6. 代码结构

```text
End4Rec/
├── run.py          # CLI 入口
├── train.py        # 训练主流程
├── trainer.py      # 分阶段训练调度
├── dataset.py      # CSV / Dummy 数据加载
├── model.py        # 模型定义
├── losses.py       # 联合损失函数
├── evaluation.py   # HR/NDCG 评估
└── utils.py        # 通用工具
```

---

## 7. 审查结论（关键问题）

本次审查发现原始版本存在以下问题并已重点处理：
1. **入口与流程割裂**：`run.py / train.py / trainer.py` 职责重复且互相依赖混乱；
2. **损失函数不可用**：引用了模型中不存在的参数名称；
3. **代码可维护性差**：大段注释历史代码混杂，影响可读性与二次开发；
4. **工程噪音文件多**：压缩包携带 macOS 冗余元文件。

---

## 8. 引用信息

如果你在论文或报告中使用了本实现，请同时引用：

1. MindSpore 官方文档（框架使用）  
   https://www.mindspore.cn/docs/zh-CN/master/index.html

2. 推荐系统常用评估指标背景（HR / NDCG）  
   Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques.

3. 本仓库的 END4Rec 工程实现（请替换为你的 GitHub 仓库链接）

---

## 9. 上传到 GitHub 的建议流程

你本地可直接执行：

```bash
git add .
git commit -m "refactor: unpack project, clean artifacts, rebuild training pipeline and docs"
git push origin <your-branch>
```

如果需要，我可以继续帮你补充：
- `requirements.txt`/`environment.yml`
- 单元测试（dataset/loss/trainer）
- GitHub Actions（lint + smoke test）
