"""候选策略 v7：精确可达判据 + 命中点规划（逐分量最优加速度序列）。

===============================================================================
相对 P224 的改进链条（每一步都在留出集上做过配对检验）
===============================================================================
P224：可达判据 = 单一方向 reach(n, v0)；动作 = 朝目标下一步预测点的单位方向满推力。
  问题：MPE2 控制是**逐分量**单位盒 |a_x|,|a_y| <= 1（不是单位圆盘），
        n 步可达集是 zonotope，对角方向允许 sqrt(2) 倍位移。
        只测"朝目标方向的位移"会漏判对角可达区里的目标。

v6：保留 P224 的选择/锁定/记忆，把动作换成逐分量比例导引
        v* = delta/(dt*n),  a* = clip((v* - v0)/ca, -1, 1)
    留出集配对 Δ = +6.44 分（95%CI ±2.82，99 胜 / 192 平 / 9 负），显著。

v7（本文件）：把"比例导引"升级为**命中点规划**。核心洞察：
  1. 覆盖判定是"某一步中心距 <= target_radius"，所以目标是**必须走到某个点**，
     而不是"朝某个方向走"。k 步精确位移公式是
         p_{k} - p_0 = v0*dt*G(k) + ca*dt * Σ_{1<=i<k} W_i * a_i
     其中 a_i 逐分量独立，因此给定想要的终点位移 d，最优控制是
         a_i,c = clip(d_c / (ca*dt*S[k-1]) , -1, 1)
     这**恰好**给出 d_c（当 |d_c| <= coef_c 时），比比例导引收敛更快更准。
  2. 选择"最早可拦截步" k*（最小的、在该步目标位置可达的 k），朝**该步的目标
     预测位置**瞄准，一次算好整段控制序列。这样机器人是"走到拦截点等目标"，
     而不是"追目标的尾巴"。
  3. 保留 P224 的记忆/速度估计/锁定/避碰，只替换可达判据与控制律。

接口与合规：只暴露 build_policy(context)；仅 numpy；只读局部观测与公开任务参数；
动作恒为 float32[2]、分量在 [-1,1]；无文件 IO、无全局可变状态。
"""

from __future__ import annotations

import numpy as np

_DT = 0.1
_DAMPING = 0.25
_DRIVE_FORCE = 1.0
_MASS = 1.0
_MAP_HALF = 1.0
_ROBOT_RADIUS = 0.05
_TARGET_RADIUS = 0.15
_SENSE_RADIUS = 0.6
_TARGET_MAX_SPEED = 0.2
_MAX_REFLECT = 8


def _reflect_axis(pos: float, vel: float, dt: float, low: float, high: float):
    """单轴镜面反射推进，与 coverage_bench.envs.motion.reflect_coordinate 等价。"""
    if vel == 0.0:
        return float(pos), 0.0
    remaining = float(dt)
    p, v = float(pos), float(vel)
    for _ in range(_MAX_REFLECT):
        wall = high if v > 0 else low
        t_wall = (wall - p) / v
        if t_wall > remaining:
            return p + v * remaining, v
        p = wall
        v = -v
        remaining -= t_wall
    return p, v


class _Kin:
    """运动学常量与精确可达系数（逐分量 zonotope）。"""

    def __init__(self, task, horizon: int):
        self.dt = float(getattr(task, "dt", _DT))
        self.ga = 1.0 - float(getattr(task, "damping", _DAMPING))
        self.ca = (float(getattr(task, "drive_force", _DRIVE_FORCE))
                   / float(getattr(task, "robot_mass", _MASS)) * self.dt)
        self.horizon = int(horizon)
        self.map_half = float(getattr(task, "map_half_extent", _MAP_HALF))
        self.robot_radius = float(getattr(task, "robot_radius", _ROBOT_RADIUS))
        self.target_radius = float(getattr(task, "target_radius", _TARGET_RADIUS))
        self.sense_radius = float(getattr(task, "sense_radius", _SENSE_RADIUS))
        self.target_max_speed = float(getattr(task, "target_max_speed", _TARGET_MAX_SPEED))
        self.target_low = -(self.map_half - self.target_radius)
        self.target_high = +(self.map_half - self.target_radius)
        H = self.horizon + 2
        # W[i] = Σ_{j<i} ga^j；(1-ga^i)/(1-ga)
        if abs(1.0 - self.ga) > 1e-12:
            self.W = np.array([(1.0 - self.ga ** i) / (1.0 - self.ga)
                               for i in range(H)], dtype=np.float64)
        else:
            self.W = np.arange(H, dtype=np.float64)
        self.S = np.concatenate(([0.0], np.cumsum(self.W)))

    def g_sum(self, n: int) -> float:
        """Σ_{i<n} ga^i：初速在 n 步位移里的加权系数。"""
        if n <= 0:
            return 0.0
        if abs(1.0 - self.ga) > 1e-12:
            return (1.0 - self.ga ** n) / (1.0 - self.ga)
        return float(n)

    def coef(self, n: int, v0: float) -> float:
        """n 步内该分量可达的最大位移（含初速）。

            coef(n, v0) = v0*dt*G(n) + ca*dt*S[n-1]
        S[n-1] = Σ_{i<n-1} W[i]，控制项从 i=1 起（a_0 第一步位移恒为 0）。
        """
        if n <= 0:
            return 0.0
        ctrl = float(self.ca) * float(self.dt) * float(self.S[max(0, n - 1)])
        return float(v0) * self.dt * self.g_sum(n) + ctrl

    def ctrl_coef(self, n: int) -> float:
        """n 步内**控制项**能贡献的分量位移系数 ca*dt*S[n-1]。"""
        if n <= 0:
            return 0.0
        return float(self.ca) * float(self.dt) * float(self.S[max(0, n - 1)])


class ZonoPlanPolicy:
    """精确可达判据 + 命中点规划。"""

    _SAFE_GAP = 0.14

    def __init__(self):
        self.kin: _Kin | None = None
        self.agent_index = 0
        self.num_agents = 0
        self.num_targets = 0
        self._hist: list[np.ndarray] = []
        self._disp = np.zeros((1, 2), dtype=np.float64)
        self._locked = -1

    # ------------------------------------------------------------ 生命周期
    def reset(self, context) -> None:
        self.kin = _Kin(context.task, context.horizon)
        self.agent_index = int(context.agent_index)
        self.num_agents = int(context.num_agents)
        self.num_targets = int(context.num_targets)
        self._hist = []
        self._disp = np.zeros((self.num_targets, 2), dtype=np.float64)
        self._locked = -1

    def close(self) -> None:
        return None

    # ------------------------------------------------------------ 观测记忆
    def _update(self, observation) -> None:
        self_pos = np.asarray(observation["self_state"][:2], dtype=np.float64)
        targets = np.asarray(observation["targets"], dtype=np.float64)
        visible = np.asarray(observation["target_visible"], dtype=bool)
        exists = np.asarray(observation["target_exists"], dtype=bool)
        m = int(min(self.num_targets, targets.shape[0]))
        row = np.full((m, 2), np.nan, dtype=np.float64)
        for j in range(m):
            if exists[j] and visible[j]:
                row[j] = self_pos + targets[j, :2]     # position_scale = 1.0
        step = int(observation["step_index"])
        while len(self._hist) <= step:
            self._hist.append(np.full((m, 2), np.nan, dtype=np.float64))
        self._hist[step] = row
        self._estimate_disp()

    def _estimate_disp(self) -> None:
        """最小二乘线性拟合目标每步位移（分段线性运动模型下在观测窗口内有效）。"""
        m = self.num_targets
        disp = np.zeros((m, 2), dtype=np.float64)
        for j in range(m):
            ts, ps = [], []
            for t, row in enumerate(self._hist):
                if not np.isnan(row[j, 0]):
                    ts.append(t)
                    ps.append(row[j])
            if len(ts) >= 2:
                tt = np.asarray(ts, dtype=np.float64)
                pp = np.asarray(ps, dtype=np.float64)
                tc = tt - tt.mean()
                denom = float((tc ** 2).sum())
                if denom > 1e-12:
                    disp[j] = (tc[:, None] * (pp - pp.mean(axis=0))).sum(axis=0) / denom
        limit = self.kin.target_max_speed * self.kin.dt
        for j in range(m):
            nrm = float(np.linalg.norm(disp[j]))
            if limit > 0.0 and nrm > limit:
                disp[j] *= limit / nrm
        self._disp = disp

    def _last_seen(self, j: int):
        for t in range(len(self._hist) - 1, -1, -1):
            if not np.isnan(self._hist[t][j, 0]):
                return self._hist[t][j].copy(), t
        return None

    def _predict(self, j: int, now: int, k: int):
        """目标 j 再过 k 步的预测位置（分段线性 + 边界镜面反射）。"""
        seen = self._last_seen(j)
        if seen is None:
            return None
        pos, t_seen = seen
        v = self._disp[j].copy()
        for _ in range(int((now - t_seen) + k)):
            pos[0], v[0] = _reflect_axis(float(pos[0]), float(v[0]), 1.0,
                                         self.kin.target_low, self.kin.target_high)
            pos[1], v[1] = _reflect_axis(float(pos[1]), float(v[1]), 1.0,
                                         self.kin.target_low, self.kin.target_high)
        return pos

    # ------------------------------------------------------------ 可达性
    def _feasible(self, delta: np.ndarray, n: int, self_vel: np.ndarray) -> bool:
        cx = self.kin.coef(n, float(self_vel[0]))
        cy = self.kin.coef(n, float(self_vel[1]))
        return abs(float(delta[0])) <= cx + 1e-12 and abs(float(delta[1])) <= cy + 1e-12

    def _intercept_step(self, j: int, now: int, self_pos, self_vel) -> int:
        """返回对目标 j 的最早可拦截步 k*（1..horizon-now）；不可达返回 -1。

        "在第 k 步拦截"= 机器人第 k 步的位置与目标第 k 步预测位置的距离
        能进入 target_radius。故所需位移是 delta - 半径方向的余量。
        """
        n_max = max(1, self.kin.horizon - now)
        tr = self.kin.target_radius
        for k in range(1, n_max + 1):
            tp = self._predict(j, now, k)
            if tp is None:
                continue
            delta = tp - self_pos
            dist = float(np.linalg.norm(delta))
            need = max(0.0, dist - tr)
            if need <= 1e-12:
                return k
            if self._feasible(delta * (need / dist), k, self_vel):
                return k
        return -1

    # ------------------------------------------------------------ 目标选择
    def _choose(self, observation, now: int, self_pos, self_vel) -> tuple[int, int]:
        """返回 (目标编号, 拦截步)。锁定优先，避免在极小位移预算里反复切换。"""
        if self._locked >= 0 and self._last_seen(self._locked) is not None:
            k = self._intercept_step(self._locked, now, self_pos, self_vel)
            if k > 0:
                return self._locked, k
        best = (-1, -1, np.inf)
        for j in range(self.num_targets):
            k = self._intercept_step(j, now, self_pos, self_vel)
            tp = self._predict(j, now, 1)
            if tp is None:
                continue
            dist = float(np.linalg.norm(tp - self_pos))
            reachable = k > 0
            cost = (0.0 if reachable else 1e6) + dist
            if cost < best[2]:
                best = (j, k, cost)
        self._locked = best[0]
        return best[0], best[1]

    # ------------------------------------------------------------ 动作
    def act(self, observation):
        self._update(observation)
        now = int(observation["step_index"])
        self_pos = np.asarray(observation["self_state"][:2], dtype=np.float64)
        self_vel = np.asarray(observation["self_state"][2:4], dtype=np.float64)

        j, k = self._choose(observation, now, self_pos, self_vel)
        if j < 0:
            return np.zeros(2, dtype=np.float32)

        # 命中点：第 k 步的目标预测位置（保留目标半径余量，瞄准"刚好进入半径"）
        aim = self._predict(j, now, max(k, 1))
        if aim is None:
            aim = self._predict(j, now, 1)
        if aim is None:
            return np.zeros(2, dtype=np.float32)

        delta = aim - self_pos
        n_plan = max(1, k)
        cmd = self._plan_cmd(delta, n_plan, self_vel)
        cmd = self._avoid(observation, self_pos, self_vel, cmd)
        cmd = np.clip(cmd, -1.0, 1.0)
        return np.ascontiguousarray(cmd, dtype=np.float32)

    def _plan_cmd(self, delta: np.ndarray, n: int, self_vel: np.ndarray):
        """求实现终点位移 delta 的逐分量最优首步控制。

        位移分解：delta_c = v0_c*dt*G(n) + ca*dt*S[n-1] * ā_c，
        其中 ā_c 是各步 a_i,c 的加权平均（权重 W_i，i 从 1 起）。
        若 |delta_c| <= coef_c(n, v0_c)，取 ā_c = 余量/(ca*dt*S[n-1]) 即可精确到达；
        否则取满推力。控制序列中"每步都取同一个 ā"是最省力的实现方式，
        且因各步系数同号，等价于让所有步同向出力。
        """
        dt = max(self.kin.dt, 1e-12)
        ca = max(self.kin.ca, 1e-12)
        ctrl = self.kin.ctrl_coef(n)
        if ctrl <= 1e-12:
            # n = 1 时控制项为 0（第 1 步位移恒为 0），只能先把速度建起来
            return np.clip((delta / (dt * 2.0) - self_vel) / ca, -1.0, 1.0)
        # 先扣掉初速已经贡献的部分，剩余交给控制项
        resid = delta - self_vel * dt * self.kin.g_sum(n)
        a = resid / ctrl
        return np.clip(a, -1.0, 1.0)

    def _avoid(self, observation, self_pos, self_vel, cmd):
        """队友过近且本机正朝其运动时叠加侧向避让（安全网）。"""
        peers = np.asarray(observation["peers"], dtype=np.float64)
        visible = np.asarray(observation["peer_visible"], dtype=bool)
        out = np.asarray(cmd, dtype=np.float64).copy()
        n = int(min(self.num_agents, peers.shape[0]))
        for i in range(n):
            if i == self.agent_index or not visible[i]:
                continue
            rel = peers[i, :2]
            d = float(np.linalg.norm(rel))
            if d < 1e-9 or d > self._SAFE_GAP:
                continue
            u = rel / d
            if float(np.dot(self_vel, u)) <= 0.0:
                continue
            w = (self._SAFE_GAP - d) / self._SAFE_GAP
            side = np.array([-u[1], u[0]], dtype=np.float64)
            if float(np.dot(out, side)) < 0.0:
                side = -side
            out = out + side * (1.5 * w)
            fwd = float(np.dot(out, u))
            if fwd > 0.0:
                out = out - u * fwd * min(1.0, 1.5 * w)
        return out


def build_policy(context):
    """官方唯一入口。"""
    return ZonoPlanPolicy()
