# END4Rec (MindSpore)

本仓库提供论文 **Efficient Noise-Decoupling for Multi-Behavior Sequential Recommendation (WWW 2024)** 的 MindSpore 实现代码。

- 论文（arXiv）：https://arxiv.org/abs/2403.17603
- 论文（ACM DOI）：https://doi.org/10.1145/3589334.3645380

## 1. 论文简介

END4Rec 面向**多行为序列推荐**场景，核心目标是同时解决两类问题：

1. 多行为序列变长后建模效率下降；
2. 多行为数据中的噪声会干扰用户兴趣建模。

论文提出三部分关键设计：

- **EBM (Efficient Behavior Sequence Miner)**：在频域建模行为序列，兼顾效率与表达能力；
- **行为感知去噪模块**：包含 token 级的硬噪声消除与表征级的软噪声过滤；
- **Noise-Decoupling Contrastive Learning + 4阶段训练策略**：逐步提升信号/噪声解耦效果。

## 2. 代码结构（与实现对应）

```text
End4Rec/
├── run.py                # 训练入口（CLI）
├── train.py              # 主训练流程与配置加载
├── trainer.py            # 4-stage 训练调度
├── model.py              # END4Rec 模型主体
├── ebm_enhanced.py       # EBM 相关实现
├── block_mlp.py          # 模块层定义
├── losses.py             # 推荐损失 + 对比损失 + 正则
├── contrastive_loss.py   # 对比学习损失
├── data.py / dataset.py  # 数据读取与构建
└── evaluation.py         # HR / NDCG 评估
```

## 3. 环境准备

建议环境：

- Python 3.9+
- MindSpore（建议与本机 CUDA/Ascend 环境匹配）

快速检查：

```bash
cd End4Rec
python -c "import mindspore; print(mindspore.__version__)"
```

## 4. 数据格式

训练 CSV 需包含以下字段：

- `item_ids`：逗号分隔的 item 序列
- `behavior_ids`：逗号分隔的行为类型序列
- `positions`：位置序列
- `label`：目标 item

示例：

```csv
item_ids,behavior_ids,positions,label
10,11,23,42
```

> 注：实际训练中 `item_ids/behavior_ids/positions` 应为等长序列（如 `10,11,23,...`）。

## 5. 快速开始

### 5.1 默认配置（快速跑通）

```bash
cd End4Rec
python run.py
```

若未提供数据文件，代码会自动回退到 dummy 数据生成器以完成完整训练流程。

### 5.2 使用自定义配置

```bash
cd End4Rec
python run.py --config config.json
```

`config.json` 可覆盖的常用参数包括：

- 模型参数：`num_items`、`num_behaviors`、`seq_length`、`d_model`、`num_blocks`、`epsilon`
- 训练参数：`batch_size`、`stage1~4_epochs`、`learning_rate_stage1~4`
- 损失权重：`reg_weight`、`contrast_weight`
- 数据参数：`train_data_file`、`eval_data_file`、`num_dummy_samples`
- 评估参数：`topk`

## 6. 训练流程（4 Stages）

当前实现对应四阶段训练：

1. `embedding + ebm`
2. `hard_noise`
3. `soft_noise`
4. 全模块联合微调（含 `output_layer`）

## 7. 评估与产物

- 若提供 `eval_data_file`，训练后会输出 `HR@K` 与 `NDCG@K`；
- 默认在训练结束后保存模型为：`End4Rec/end4rec_final.ckpt`。

## 8. 引用

如果你的工作使用了本仓库，请引用原论文：

```bibtex
@inproceedings{han2024efficient,
  title={Efficient Noise-Decoupling for Multi-Behavior Sequential Recommendation},
  author={Han, Yongqiang and Wang, Hao and Wang, Kefan and Wu, Likang and Li, Zhi and Guo, Wei and Liu, Yong and Lian, Defu and Chen, Enhong},
  booktitle={Proceedings of the ACM Web Conference 2024 (WWW '24)},
  year={2024},
  doi={10.1145/3589334.3645380}
}
```
