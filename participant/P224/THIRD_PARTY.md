# 第三方依赖与许可声明（P224）

## 结论

本提交的**策略实现与推理过程不包含任何第三方源码**，也不含任何训练产物。
`entry.py` 仅依赖：

- **NumPy**（BSD-3-Clause）—— 官方评测环境已提供，见仓库 `requirements-official.lock`
  （锁定版本 `numpy==2.5.3`），因此 `requirements-infer.lock` 未列额外依赖。

## 未使用的内容

为避免歧义，明确说明本提交**不包含、不依赖**以下内容：

| 内容 | 状态 |
| --- | --- |
| PyTorch / Stable-Baselines3 / SuperSuit | 未使用（评测环境也不提供） |
| 任何预训练模型权重或检查点 | 未使用（`artifacts/` 为空，`checkpoint_manifest: []`） |
| 任何第三方复制的代码片段 | 无 |
| 任何外部下载资源 | 无（静态审计禁止且实现无网络调用） |

## 对官方接口的引用

实现过程中阅读并依据了仓库内的官方实现来确认物理与协议语义，均为**阅读参考**，
未复制其代码，也未修改任何官方文件：

- `coverage_bench/envs/scenario.py`、`coverage_bench/envs/physics.py`、
  `coverage_bench/envs/motion.py`：确认机器人动力学的实际执行路径（MPE2
  `World.integrate_state`）与目标分段线性运动模型；
- `coverage_bench/protocol.py`、`coverage_bench/observations.py`：确认观测字段、
  归一化尺度与可见性语义；
- `coverage_bench/metrics.py`、`coverage_bench/scoring.py`：确认覆盖按二分图最大匹配
  计数、以及 `performance_score` 的归一化公式。

`entry.py` 内的 `_reflect_axis` 是按 `coverage_bench/envs/motion.py` 的
`reflect_coordinate` 的**行为语义**独立实现（用于策略内部预测目标位置），
不是该模块的复制粘贴；其正确性通过数值对照验证。

## 许可

本人可授权部分采用 MIT 许可，见同目录 `LICENSE`。
