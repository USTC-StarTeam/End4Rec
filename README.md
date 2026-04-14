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
