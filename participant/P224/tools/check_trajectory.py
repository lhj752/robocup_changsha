"""目标轨迹口径核对：本工具复现的轨迹必须与真实 env.step 逐位一致。

================================================================================
背景
================================================================================
E009 上界工具曾经有一个隐蔽 bug：推进目标时**早了一步转向**。
真实环境的顺序（`coverage_bench/envs/parallel_env.py:132-134`）是

    advance_robots(...)  ->  advance_targets(...)  ->  step_index += 1

而转向判定（`coverage_bench/envs/motion.py:40`）用的是**自增前**的步号

    if state.step_index == state.target_next_turn[j]:

`target_next_turn[j]` 初值 ∈ [5,10]（`envs/scenario.py:121`），`step_index` 从 0 起。
所以推进第 k 步时 `advance_targets` 必须看到 k-1。若写成"先自增再推进"，就会提前
一步触发转向，目标轨迹与真实环境不符，上界随之失效。

本文件对公开套件每个场景做如下对照：
    A) 真实轨迹：env.reset(seed) 后连续 env.step(全零动作) 10 次
    B) 工具轨迹：env.reset(seed) 后连续 [advance_targets, step_index += 1] 10 次
两者应当**逐位相同**（max|Δ| = 0.000000）。同时核对 reach 表的期望数值。

用法：
    python check_trajectory.py --repo <仓库根>
    ROBOCUP_ROOT=<仓库根> python check_trajectory.py
"""
from __future__ import annotations

import sys

# 必须在导入本仓库任何模块之前设置，避免在提交目录内生成 __pycache__
sys.dont_write_bytecode = True

import os                      # noqa: E402
from pathlib import Path       # noqa: E402

import numpy as np             # noqa: E402


def find_root() -> Path:
    """定位比赛仓库根：只认 --repo 参数与 ROBOCUP_ROOT 环境变量。"""
    candidates: list[Path] = []
    argv = sys.argv
    for i, a in enumerate(argv):
        if a in ("--repo", "--root") and i + 1 < len(argv):
            candidates.append(Path(argv[i + 1]))
    if os.environ.get("ROBOCUP_ROOT"):
        candidates.append(Path(os.environ["ROBOCUP_ROOT"]))
    for cand in candidates:
        for parent in [cand, *cand.parents]:
            if (parent / "configs" / "public-suite-v1.yaml").is_file():
                return parent
    raise SystemExit(
        "找不到比赛仓库根目录（需含 configs/public-suite-v1.yaml）。\n"
        "请用 --repo <路径> 或环境变量 ROBOCUP_ROOT 指定。"
    )


ROOT = find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coverage_bench import make_training_env           # noqa: E402
from coverage_bench.envs.motion import advance_targets  # noqa: E402
from coverage_bench.suites import load_suite            # noqa: E402

# reach 期望值（公开参数 dt=0.1, damping=0.25, drive_force=1.0, mass=1.0）
# 注意：这些是 6 位小数四舍五入后的展示值，因此比较容差取 5e-7
# （reach(5) 精确值 0.07796875，reach(10) 精确值 0.24901000）
EXPECTED_REACH = {1: 0.0, 2: 0.01, 5: 0.077969, 10: 0.249010}
TOL = 1e-12          # 轨迹逐位比对的容差
REACH_TOL = 5e-7     # reach 展示值比对容差


def ref_reach_v0(public, n: int, v0: float) -> float:
    """独立递推基准：初速投影 v0、n 步内的真实最大位移。

    位置先用旧速度，故 v0 在 n 步里的贡献是 v0 * Σ_{i=0}^{n-1} ga^i = v0*(1-ga^n)/(1-ga)。
    本函数**不引用 entry.py 的任何实现**，是与它相互独立的参照。
    """
    ga = 1.0 - float(public.damping)
    ca = float(public.drive_force) / float(public.robot_mass) * float(public.dt)
    v, tot = float(v0), 0.0
    for _ in range(int(n)):
        tot += v
        v = ga * v + ca
    return float(public.dt) * tot


def entry_reach(task_config, n: int, v0: float) -> float:
    """调用提交物 entry.py 里 _Kinematics.reach(n, v0)，与基准比对。

    注意两点：
    - 就地导入 entry 会在提交目录内生成 __pycache__，因此本文件开头已设
      `sys.dont_write_bytecode = True`（R1.3 的硬约束）。
    - 这里用**静态** `import entry`（本文件只加载固定的提交目录，不做候选切换），
      不使用 importlib.* 之类的动态载入——那会触发审计硬拒绝
      UNREGISTERED_DYNAMIC_LOAD（coverage_bench/audit.py:52-55）。
    """
    sub_dir = str(ROOT / "participant" / "P224")
    saved = list(sys.path)
    sys.path.insert(0, sub_dir)
    try:
        import entry as entry_module          # noqa: PLC0415 - 需要先准备好 sys.path
        kin = entry_module._Kinematics(task_config.public, int(task_config.horizon))
        return float(kin.reach(int(n), float(v0)))
    finally:
        sys.path[:] = saved
        sys.modules.pop("entry", None)



def reach_true(public, horizon: int) -> np.ndarray:
    """与 bound_correct.py 完全一致的真实 reach（迭代累加 n 项旧速度）。"""
    ga = 1.0 - float(public.damping)
    ca = float(public.drive_force) / float(public.robot_mass) * float(public.dt)
    dt = float(public.dt)
    out = np.zeros(horizon + 1, dtype=np.float64)
    v, tot = 0.0, 0.0
    for n in range(1, horizon + 1):
        tot += v
        v = ga * v + ca
        out[n] = dt * tot
    return out


def real_trajectory(task_config, seed: int) -> list[np.ndarray]:
    """真实轨迹：env.step(全零动作) 逐步记录目标位置（含 t=0 初始帧）。"""
    env = make_training_env(task_config)
    try:
        env.reset(seed=seed)
        frames = [env._scenario_state.target_positions.copy()]
        for _ in range(int(task_config.horizon)):
            acts = {a: np.zeros(2, dtype=np.float32) for a in env.agents}
            env.step(acts)
            frames.append(env._scenario_state.target_positions.copy())
        return frames
    finally:
        env.close()


def tool_trajectory(task_config, seed: int):
    """工具轨迹：env.reset(seed) 后按 [advance_targets, step_index += 1] 推进。"""
    env = make_training_env(task_config)
    try:
        env.reset(seed=seed)
        st = env._scenario_state
        frames = [st.target_positions.copy()]
        for _ in range(int(task_config.horizon)):
            advance_targets(st)      # 此时 step_index 仍为 k-1
            st.step_index += 1
            frames.append(st.target_positions.copy())
        return frames
    finally:
        env.close()


def main() -> None:
    suite = load_suite(ROOT / "configs" / "public-suite-v1.yaml")
    print("repo =", ROOT)

    # ---- 1. reach 表核对 ----
    print("=" * 78)
    print("reach 表核对（真实位移，迭代法；期望值来自实测）")
    print("=" * 78)
    cfg0 = suite.groups[0].cases[0].task_config
    reach = reach_true(cfg0.public, int(cfg0.horizon))
    print("  n   reach(n)      期望值(6位小数)  偏差")
    reach_ok = True
    for n, exp in EXPECTED_REACH.items():
        dev = abs(float(reach[n]) - exp)
        ok = dev <= REACH_TOL
        reach_ok &= ok
        print(f" {n:2d}   {reach[n]:.6f}     {exp:.6f}       {dev:.2e}  {'OK' if ok else 'FAIL'}")

    # ---- 2. 轨迹逐位对照 ----
    print()
    print("=" * 78)
    print("目标轨迹逐位对照：工具复现 vs 真实 env.step（全零动作）")
    print("=" * 78)
    all_ok = True
    worst = 0.0
    for group in suite.groups:
        for case in group.cases:
            real = real_trajectory(case.task_config, case.scenario_seed)
            tool = tool_trajectory(case.task_config, case.scenario_seed)
            assert len(real) == len(tool), "帧数不一致"
            max_dev = max(float(np.max(np.abs(r - t))) for r, t in zip(real, tool))
            # 首次偏离的步号（便于定位）
            first_bad = next((k for k, (r, t) in enumerate(zip(real, tool))
                              if float(np.max(np.abs(r - t))) > TOL), None)
            all_ok &= (max_dev <= TOL)
            worst = max(worst, max_dev)
            print(f"  case={case.case_id:8s} seed={case.scenario_seed}  "
                  f"帧数={len(real)}  max|Δ| = {max_dev:.6f}  "
                  f"首次偏离={'-' if first_bad is None else f'k={first_bad}'}  "
                  f"{'OK' if max_dev <= TOL else 'FAIL'}")

    # ---- 3. reach(n, v0) 对照（v0 != 0）----
    # 这一组断言的存在理由：曾经 entry.py 把 v0 的系数写成 n（正确值是 (1-ga^n)/(1-ga)），
    # 导致"可达性判定"在回合后段偏乐观（n=10、v0=0.4 时高估 1.62 倍），
    # 而只校验 v0=0 的断言完全看不出来。加这一组后，同类 bug 不可能再溜过去。
    print()
    print("=" * 78)
    print("reach(n, v0) 对照：独立递推基准 vs entry.py 的 _Kinematics.reach（容差 1e-12）")
    print("=" * 78)
    print("  基准递推：v 从 v0 起，逐位 tot += v; v = ga*v + ca，最后 dt*tot")
    print("  （v0 在 n 步里的贡献是 v0*sum_{i=0}^{n-1} ga^i = v0*(1-ga^n)/(1-ga)，不是 v0*n）")
    print()
    print(f"  {'n':>3}  {'v0':>5}  {'基准(独立递推)':>18}  {'entry.reach':>18}  {'偏差':>10}  判定")
    reach_v0_ok = True
    for n in (1, 2, 5, 10):
        for v0 in (0.1, 0.2, 0.4):
            base = ref_reach_v0(cfg0.public, n, v0)
            got = entry_reach(cfg0, n, v0)
            dev = abs(base - got)
            ok = dev <= TOL
            reach_v0_ok &= ok
            print(f"  {n:>3}  {v0:>5.1f}  {base:>18.12f}  {got:>18.12f}  {dev:>10.2e}  "
                  f"{'OK' if ok else 'FAIL'}")
    if not reach_v0_ok:
        print()
        print("  [!] v0 != 0 时 entry.py 的 reach 与真实递推不一致 —— 这会让'可达性判定'偏乐观。")
        print("      v0 的系数必须是 (1-ga^n)/(1-ga)；若写成 n，误差随 n 线性放大。")

    print()
    print(f"  reach 表(v0=0): {'全部通过' if reach_ok else '存在偏差'}")
    print(f"  reach(n,v0) 对照: {'全部通过' if reach_v0_ok else '存在偏差'}")
    print(f"  轨迹对照: 最大偏差 {worst:.12f}")
    if all_ok and reach_ok and reach_v0_ok:
        print("结论：目标轨迹逐位一致，且 entry.py 的 reach 与真实递推一致（退出码 0）。")
    else:
        bad = []
        if not all_ok:
            bad.append("目标轨迹与真实环境不一致")
        if not reach_ok:
            bad.append("reach(v0=0) 表与期望值不符")
        if not reach_v0_ok:
            bad.append("reach(n,v0) 与真实递推不一致（v0 系数错误）")
        print("结论：存在不一致 - " + "；".join(bad) + "（退出码 1）。")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
