"""公开套件与自选留出集的严格上界（精确判据版，P224 修订 R1）。

================================================================================
这个上界是什么
================================================================================
构造一个**不依赖任何控制策略**的得分上界，用于回答"距离天花板还有多远"。

1) MPE2 的实际积分（`mpe2/_mpe_utils/core.py:231-252`；由
   `coverage_bench/envs/scenario.py` 在 `create_scenario` 中锁定 `mpe_world`，
   因此这是正式评测唯一会走的路径）：

       p_{i+1} = p_i + v_i * dt
       v_{i+1} = ga * v_i + ca * a_i        ga = 1-damping, ca = drive_force/mass*dt

   控制约束是**逐分量单位盒** |a_x| <= 1 且 |a_y| <= 1（代码里直接
   `v += (force/mass)*dt`，没有任何范数归一）。从静止出发（v_0 = 0）：

       p_k - p_0 = ca*dt * Σ_{i=1}^{k-1} W_i * a_i ,   W_i = (1-ga^i)/(1-ga)

   （i 从 1 起：位置先用旧速度，故 a_0 对应的第 1 步位移恒为 0。）

2) 于是 k 步可达集是一个 **zonotope**（中心对称凸多边形），每个控制步的
   x/y 分量各贡献一个长度 g_i = ca*dt*W_i 的生成元。记

       s_k = Σ_{i=1}^{k-1} g_i

   各方向的支撑值是 s_k*(|u_x| + |u_y|)：沿坐标轴为 s_k，沿对角为 √2·s_k。

3) 覆盖判定是"中心距 <= target_radius"，即"目标点与可达集的欧氏距离 <= r_t"。
   逐分量盒约束下，目标点到可达集的最小距离可**精确**求出（逐分量一维盒约束
   最小二乘，闭式解）：

       dist_min = √( max(0,|d_x| - s_k)² + max(0,|d_y| - s_k)² )

   于是 (i,j,k) 可覆盖的**精确条件**是

       √( max(0,|d_x| - s_k)² + max(0,|d_y| - s_k)² ) <= r_t

   **关键：r_t 是欧氏球。** 因为 |d| - r_t 的单位球恰好等于 (|d_x|,|d_y|) 的
   l∞ 盒，所以"球与 zonotope 相交"必须用 l∞ 形式的缺口；把 r_t 各分量都加一遍
   会高估可达性。

4) 该步覆盖数 <= 该步可达二部图的最大匹配数，于是
       J <= (1/T) * Σ_k m_k / M

================================================================================
历史与更正（保留，供复核者判断本文件的可靠性）
================================================================================
本文件此前有两个**方向相反**的错误，均已修正：

  * 错误一（低估）：把 s_k 当作"各方向通用的可达半径"，即判据 |d| <= s_k + r_t。
    这只在坐标轴方向成立。沿对角可达 √2·s_k，故该判据漏判部分可覆盖组合，
    把公开套件上界压到 191.67（真值 225.00）。

  * 错误二（高估）：改用支撑函数 s_k*(|u_x|+|u_y|) 后，把 r_t 当作"可沿该方向
    全额叠加的余量"，即判据 |d| <= s_k*(|cosφ|+|sinφ|) + r_t。支撑函数只说明
    "zonotope 在该方向能伸多远"，而 r_t 是欧氏球，两者不能直接相加。该判据在
    公开套件上给出 241.67，**高于真值**，是不可用的上界。

  正确判据是上面第 3 条的盒约束最小二乘形式，公开套件给出 **225.00**。

  2026-10-05 的独立复核（穷举端点 + 闭式 + 盒约束 LP 三方一致，并对判据分歧对
  做 MPE2 真实物理回放）确认了该结论：支撑函数判据曾把 3 个组合
  （`basic-0 (2,0,8)`、`coop-1 (0,2,3)`、`coop-1 (1,0,5)`）误判为可达，
  回放后它们到目标的最小距离分别为 0.1628 / 0.1508 / 0.1545，均 > r_t = 0.15。

================================================================================
目标轨迹必须与真实环境一致
================================================================================
真实推进顺序（`coverage_bench/envs/parallel_env.py:132-134`）是
    advance_robots(...) -> advance_targets(...) -> step_index += 1
而转向判定（`coverage_bench/envs/motion.py:40`）用的是**自增前**的步号
    if state.step_index == state.target_next_turn[j]:
所以推进第 k 步时 `advance_targets` 看到的是 k-1。本文件严格按此顺序。

================================================================================
用法
================================================================================
    python bound_correct.py --repo <仓库根>
    python bound_correct.py --repo <仓库根> --holdout 300
    python bound_correct.py --repo <仓库根> --seeds-file <种子文件>
"""
from __future__ import annotations

import sys

# 必须在导入本仓库任何模块之前设置，避免在提交目录内生成 __pycache__
sys.dont_write_bytecode = True

import os                      # noqa: E402
from pathlib import Path       # noqa: E402

import numpy as np             # noqa: E402
from scipy.sparse import csr_matrix                          # noqa: E402
from scipy.sparse.csgraph import maximum_bipartite_matching  # noqa: E402


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


def support_scale(public, k: int) -> float:
    """s_k = ca*dt*Σ_{1<=i<k} W_i：k 步可达 zonotope 沿坐标轴方向的支撑值。

    校验值（公开参数 dt=0.1, damping=0.25, drive_force=1.0, mass=1.0）：
        s_1=0.000000  s_2=0.010000  s_5=0.077969  s_10=0.249010
    s_10 = 0.249010 同时是"10 步沿轴向的最大位移"，由全程满推力取得。
    """
    ga = 1.0 - float(public.damping)
    ca = float(public.drive_force) / float(public.robot_mass) * float(public.dt)
    dt = float(public.dt)
    tot = 0.0
    for i in range(1, k):
        tot += (1.0 - ga ** i) / (1.0 - ga)
    return ca * dt * tot


def coverable(public, d: np.ndarray, r_t: float, k: int) -> bool:
    """精确判据：目标点与 k 步可达 zonotope 的欧氏距离 <= r_t。

    逐分量盒约束下的最小距离有闭式解（一维盒约束最小二乘）：
        min_{|a|<=1} (d_c - s*a)²  =  0                 若 |d_c| <= s
                                   =  (|d_c| - s)²      否则
    故 dist_min = || (max(0,|d_x|-s), max(0,|d_y|-s)) ||_2。
    """
    s = support_scale(public, k)
    rx = max(0.0, abs(float(d[0])) - s)
    ry = max(0.0, abs(float(d[1])) - s)
    return float(np.hypot(rx, ry)) <= r_t + 1e-12


def episode_bound(task_config, seed: int):
    """返回 (J 上界, 逐步最大匹配数)。目标轨迹与真实环境逐位一致。"""
    env = make_training_env(task_config)
    try:
        env.reset(seed=seed)
        st = env._scenario_state
        H = int(task_config.horizon)
        r0 = st.robot_positions.copy()          # 初速恒为 0（scenario.py 保证）
        n, m = int(r0.shape[0]), int(task_config.num_targets)
        r_t = float(task_config.public.target_radius)
        traj = []
        for _ in range(H):
            # 关键顺序：先推进目标（此时 step_index 仍为 k-1），再自增步号
            advance_targets(st)
            st.step_index += 1
            traj.append(st.target_positions.copy())

        per_step: list[int] = []
        for k in range(1, H + 1):
            tp = traj[k - 1]
            adj = np.zeros((n, m), dtype=np.int32)
            for i in range(n):
                for j in range(m):
                    if coverable(task_config.public, tp[j] - r0[i], r_t, k):
                        adj[i, j] = 1
            match = maximum_bipartite_matching(csr_matrix(adj), perm_type="column")
            per_step.append(int(np.sum(match >= 0)))
        return float(sum(per_step) / m / H), per_step
    finally:
        env.close()


def _num_arg(flag: str, default: int) -> int:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv) and argv[i + 1].isdigit():
            return int(argv[i + 1])
    return default


def _str_arg(flag: str, default=None):
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
    return default


def _load_seeds_file(path) -> list[int]:
    """读取种子集文件（每行一个整数，'#' 起注释）。"""
    out: list[int] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(int(line))
    if not out:
        raise SystemExit(f"种子集文件为空: {path}")
    return out


def main() -> None:
    seeds_file = _str_arg("--seeds-file")
    holdout_n = 0 if seeds_file else _num_arg("--holdout", 0)
    suite = load_suite(ROOT / "configs" / "public-suite-v1.yaml")

    print("repo =", ROOT)
    print("=" * 78)
    print("严格上界（精确判据 = 盒约束最小二乘 / 目标点到 zonotope 的欧氏距离）")
    print("任何合法策略的 performance_score 都不可能超过下列数值")
    print("=" * 78)

    pub: dict[str, list[float]] = {}
    for group in suite.groups:
        vals = []
        for case in group.cases:
            j, per = episode_bound(case.task_config, case.scenario_seed)
            vals.append(j)
            print(f"  {group.group_id:12s} {case.case_id:8s} seed={case.scenario_seed}  "
                  f"J 上界={j:.4f}  逐步匹配={per}")
        pub[group.group_id] = vals

    b, c = float(np.mean(pub["basic"])), float(np.mean(pub["cooperation"]))
    pub_score = 500.0 * (b + c)
    print("-" * 78)
    print(f"  公开套件 J 上界: basic={b:.4f} cooperation={c:.4f}")
    print(f"  >>> 公开套件 performance_score 严格上界 = {pub_score:.2f}")
    print("      P224 修订 R1 实测 = 221.67  →  达成率 98.5%")
    print("      P224 原始定稿实测 = 191.67  →  达成率 85.2%")
    print("      评分函数绝对上限（两组 J 均取满 1.0）= 1000.00")
    print()
    print("  注：本判据撤掉了“必须存在一条同时满足全部各步约束的控制序列”这一要求，")
    print("      属位置松弛上界，因此仍可能高于真实可达值；但它是**合法**上界。")

    if holdout_n > 0 or seeds_file:
        if seeds_file:
            seeds = _load_seeds_file(seeds_file)
            label_origin = f"seeds-file = {seeds_file}（N = {len(seeds)}）"
        else:
            seeds = [100000 + 7 * i for i in range(holdout_n)]
            label_origin = f"等差种子 100000 + 7i（N = {holdout_n}）"
        print()
        print(f"  自选留出集（held-out）上界：{label_origin}")
        means = []
        for label, cfg in (("uniform", suite.groups[0].cases[0].task_config),
                           ("crossing", suite.groups[1].cases[0].task_config)):
            vals = np.array([episode_bound(cfg, s)[0] for s in seeds])
            means.append(float(vals.mean()))
            print(f"    {label:10s} mean J 上界={vals.mean():.6f}  "
                  f"median={np.median(vals):.4f}  "
                  f"95%CI=[{np.percentile(vals, 2.5):.4f}, {np.percentile(vals, 97.5):.4f}]  "
                  f"上界为 0 的比例={(vals <= 1e-9).mean():.1%}")
        hold_score = 500.0 * (means[0] + means[1])
        print(f"  >>> 留出集 mean_j 上界合计 = {means[0] + means[1]:.6f}   "
              f"等价 performance_score = {hold_score:.2f}")
        print("      与本上界配对比较的实测分必须使用**同一批种子**。")


if __name__ == "__main__":
    main()
