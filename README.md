<div align="center">

# **More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models (ICLR 2026)**

[![Project Page](https://img.shields.io/badge/🌐%20Project%20Page-Visit-blue)](https://xytian1008.github.io/VAPO/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.25848-b31b1b.svg)](https://arxiv.org/abs/2509.25848)
[![GitHub](https://img.shields.io/badge/💻%20GitHub-Code-green)](https://github.com/xytian1008/VAPO)

</div>

本仓库是 [VAPO](https://github.com/xytian1008/VAPO)（Vision-Anchored Policy Optimization，ICLR 2026）官方实现的 fork，也是本项目「Forgetting-Aware Visual Anchoring」的开发仓库。VAPO 是 GRPO 的多模态替代算法：在推理路径中嵌入视觉锚点（visual anchors），通过感知奖励（perception reward）鼓励模型在长推理中保持视觉依据，缓解视觉遗忘（visual forgetting）。官方数据集、权重等资源见下方[资源链接](#资源链接)。

---

# 📋 项目研究计划：Forgetting-Aware Visual Anchoring

> **项目 DDL：2026-12-13** ｜ 参考硬件：单机 8× A100 80GB（官方 7B 训练脚本默认配置）
> 提案文件：[proposal-Forgetting-Aware Visual Anchoring for Hallucination Mitigation in Vision-Language Models.pdf](./proposal-Forgetting-Aware%20Visual%20Anchoring%20for%20Hallucination%20Mitigation%20in%20Vision-Language%20Models.pdf)

本项目基于 VAPO，提出**遗忘感知的视觉锚定**：用轻量级遗忘风险预测器将 VAPO 固定时间权重 β 替换为输入自适应的 β(x)，缓解 VLM 长推理中的视觉遗忘。

## 任务分工与进度规划（2026-11-18 → 2026-12-13）

| 模块 | 成员 | 工作内容 | 大致时间线 | 里程碑 / 产出 |
|------|------|----------|------|----------------|
| 环境与数据准备 | 全员 | 下载 36K 训练集、1K 验证集、Qwen2.5-VL-7B-Instruct 与 VAPO-Thinker-7B 权重 | 11.18 – 11.20 | 环境就绪、示例脚本跑通 |
| 基线复现 + 视觉遗忘分析 | chw、whm（评测：zly） | ① 复现 Qwen2.5-VL-7B 评测基线、GRPO 与原始 VAPO-7B 训练；② 利用视觉 Claims/Labels 对比训练轨迹早期 vs 后期视觉事实检查结果，构建视觉遗忘数据集 | 11.20 – 12.02 | 三组基线结果；遗忘数据集 v1 |
| 遗忘风险预测器 + 自适应奖励机制 | lh、jyc | ① 用冻结的图像/问题特征训练轻量预测器（held-out 任务类别验证迁移）；② 将 β(x) 接入 VAPO 奖励（预测器冻结），训练改进版 VAPO-7B | 12.01 – 12.10 | 预测器 v1；改进模型 checkpoint |
| 模型评测与收尾 | zly  | ① 基线评测（基线就绪后提前开展）；② 在 HallusionBench、MMStar、MathVista 上完成四方法对比评测；③ 结果分析、报告/论文撰写 | 12.08 – 12.13 | 最终报告 |

> 各模块时间有意重叠：预测器在遗忘数据集 v1 产出后启动，评测在基线就绪后即可提前开展；12.11–12.13 留作报告撰写与缓冲。

> 可选扩展（时间允许时）：将改进方法迁移到 Qwen3.5-9B 视觉语言模型上验证。

## 改进思路

VAPO 的感知奖励使用固定时间权重 β（手工调参）。本项目将其替换为输入依赖的 β(x)（预测器保持冻结、仅训练时使用，推理时无额外开销）：

```
R_perc(x, o) = Σ_k e^{β(x)·t_k/T} · s_k / Σ_k e^{β(x)·t_k/T}
R(x, o)     = R_acc + R_fmt + γ · 1[R_acc = 1] · R_perc(x, o)
```

- β(x) 越大，越强调推理后期（t_k/T 较大）的视觉检查；接近 0 时各检查权重趋于均匀。
- β(x) 由轻量预测器给出，以「早期正确、后期答错的视觉事实比例」为监督信号，输入为冻结的图像/问题特征。

## 资源链接

### VAPO（基础方法）

| 资源 | 链接 |
|------|------|
| 论文（ICLR 2026） | https://arxiv.org/abs/2509.25848 |
| 项目主页 | https://xytian1008.github.io/VAPO/ |
| 官方代码仓库 | https://github.com/xytian1008/VAPO |
| 训练数据（约 36K） | https://huggingface.co/datasets/xytian1008/VAPO-Thinker-train36k |
| 验证数据（约 1K） | https://huggingface.co/datasets/xytian1008/VAPO-Thinker-val1k |
| VAPO-Thinker-7B 基线权重 | https://huggingface.co/xytian1008/VAPO-Thinker-7B |

### 基础模型与数据

- Qwen2.5-VL-7B-Instruct：https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- ViRL39K（VAPO 训练数据来源）：https://huggingface.co/datasets/TIGER-Lab/ViRL39K
- Qwen3.5（可选扩展目标）：https://qwen.ai/blog?id=qwen3.5

### 评测基准

- HallusionBench：https://github.com/tianyi-lab/HallusionBench
- MMStar：https://github.com/MMStar-Benchmark/MMStar
- MathVista：https://github.com/lupantech/MathVista
- VLMEvalKit（评测框架，本仓库已内置 `vlmeval/`）：https://github.com/open-compass/VLMEvalKit

---

# 快速上手

## 安装

**Requirements:** Python ≥ 3.9, CUDA-compatible GPUs, `torch`, `vllm >= 0.8.0`, `transformers >= 4.51.0`.

```bash
git clone https://github.com/julia-lars/VAPO_pro.git
cd VAPO_pro
pip install -e .
```

## 训练

7B 训练脚本（官方参考配置：单机 8×A100-80G）：

```bash
bash examples/qwen2_5_vl_7b_train36k_vapo.sh
```

## 评测

评测基于内置的 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)：

```bash
python run.py --model <模型名> --data HallusionBench MMStar MathVista
```

`python run.py --help` 可查看支持的模型与数据集列表。VAPO-Thinker 为标准 Qwen2.5-VL checkpoint，可直接用 transformers / vLLM 加载推理。

---

# Citation🎓

```bibtex
@article{tian2025more,
  title={More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models},
  author={Tian, Xinyu and Zou, Shu and Yang, Zhaoyuan and He, Mengqi and Waschkowski, Fabian and Wesemann, Lukas and Tu, Peter and Zhang, Jing},
  journal={arXiv preprint arXiv:2509.25848},
  year={2025}
}
```

# License📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
