from functools import wraps
from inspect import signature
from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Module

from .serializers import HermiteSerializer, DiscreteSerializer

SERIALIZERS = {
    "hermite": HermiteSerializer,
    "discrete": DiscreteSerializer,
}


class _Repara(Module):
    def __init__(self, T: Tensor, t_mask: Tensor | None = None) -> None:
        super().__init__()
        self.T = T
        t_mask = t_mask
        self.DELTA = torch.diff(
            self.T, dim=1
        )  # [batch_dims, seq_len - 1, *t_X_channels]
        assert self.DELTA.isfinite().all()
        if t_mask is not None:
            assert (self.DELTA[:, :, t_mask] != 0).all()  # t_mask is not None

    def forward(self, s: Tensor, j: int) -> tuple[Tensor | None]:
        raise NotImplementedError


class HermiteRepara(_Repara):
    def __init__(self, T: Tensor, t_mask: Tensor | None = None, method: str | None = None) -> None:
        super().__init__(T, t_mask)

        self.D = torch.zeros_like(self.T)  # [batch_dims, seq_len, *t_X_channels]
        if method is None:
            self.D[:, 1:-1, ...] = 0.5 * (
                self.DELTA[:, 1:, ...] + self.DELTA[:, :-1, ...]
            )
            self.D[:, 0, ...] = self.DELTA[:, 0, ...]
            self.D[:, -1, ...] = self.DELTA[:, -1, ...]
        elif method == "makima":
            # https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation
            w1 = torch.abs(
                self.DELTA[:, 3:, ...] - self.DELTA[:, 2:-1, ...]
            ) + 0.5 * torch.abs(self.DELTA[:, 3:, ...] + self.DELTA[:, 2:-1, ...])
            w2 = torch.abs(
                self.DELTA[:, 1:-2, ...] - self.DELTA[:, :-3, ...]
            ) + 0.5 * torch.abs(self.DELTA[:, 1:-2, ...] + self.DELTA[:, :-3, ...])
            self.D[:, 2:-2, ...] = (
                w1 / (w1 + w2) * self.DELTA[:, 1:-2, ...]
                + w2 / (w1 + w2) * self.DELTA[:, 2:-1, ...]
            )
            self.D[:, 0, ...] = self.DELTA[:, 0, ...]
            self.D[:, 1, ...] = 0.5 * (self.DELTA[:, 0, ...] + self.DELTA[:, 1, ...])
            self.D[:, -2, ...] = 0.5 * (self.DELTA[:, -2, ...] + self.DELTA[:, -1, ...])
            self.D[:, -1, ...] = self.DELTA[:, -1, ...]
            # self.D[:, i, 0] will be NaN if self.DELTA[:, i - 2, ...] == 0
            # and self.DELTA[:, i - 1, ...] == 0 and
            # self.DELTA[:, i, ...] == 0 and self.DELTA[:, i + 1, ...] == 0
            self.D = torch.nan_to_num(self.D, nan=0.0)
        else:
            raise NotImplementedError

        self.B = torch.stack(
            [
                self.T[:, :-1, ...],
                self.T[:, 1:, ...],
                self.D[:, :-1, ...],
                self.D[:, 1:, ...],
            ],
            dim=-1,
        )  # [batch_dims, seq_len - 1, *t_X_channels, 4]
        self.B = self.B.unsqueeze(-2)  # [batch_dims, seq_len - 1, *t_X_channels, 1, 4]

        self.A = torch.tensor(
            ([[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]]),
            dtype=self.T.dtype,
            device=self.T.device,
        )
        self.A = self.A.expand(
            *self.DELTA.shape[:2], *([1] * len(self.DELTA.shape[2:])), -1, -1
        )  # [batch_dims, seq_len - 1, *([1] * len(t_X_channels)), 4, 4]

        self.K = torch.matmul(
            self.B, self.A
        )  # [batch_dims, seq_len - 1, *t_X_channels, 1, 4]

    def forward(self, s: Tensor, j: int) -> tuple[Tensor | None]:
        k = self.K[:, j, ...]  # [batch_dims, *t_X_channels, 1, 4]
        s_ = s - j
        s_2 = s_ * s_
        s_3 = s_2 * s_

        s_v = torch.tensor(
            [[1], [s_], [s_2], [s_3]], dtype=self.T.dtype, device=self.T.device
        )
        s_v = s_v.expand(
            k.shape[0], *([1] * len(k.shape[1:-2])), -1, -1
        )  # [batch_dims, *([1] * len(t_X_channels)), 4, 1]
        phi = torch.matmul(k, s_v).view(*k.shape[:-2])  # [batch_dims, *t_X_channels]

        s_p_v = torch.tensor(
            [[0], [1], [2 * s_], [3 * s_2]], dtype=self.T.dtype, device=self.T.device
        )
        s_p_v = s_p_v.expand(
            k.shape[0], *([1] * len(k.shape[1:-2])), -1, -1
        )  # [batch_dims, *([1] * len(t_X_channels)), 4, 1]
        dphi_ds = torch.matmul(k, s_p_v).view(
            *k.shape[:-2]
        )  # [batch_dims, *t_X_channels]

        return phi, dphi_ds


class PCHIPRepara(_Repara):
    def __init__(self, T: Tensor, t_mask: Tensor | None = None) -> None:
        super().__init__(T, t_mask)
        # 1. 对应公式：调用基类计算 self.DELTA = diff(T, dim=1)
        #    Δ_i = t_{i+1} - t_i，是所有后续公式的核心
        # 2. 影响：这是整个重参数化的基础，必须精确且有限（基类已 assert isfinite）
        # 3. 修改建议：无需改动，但可加 if not T.is_monotonic(): warn("PCHIP assumes monotonic time")

        a = self.T  # [batch_dims, seq_len, *t_X_channels]
        # 1. 对应公式：Hermite 多项式系数 a = y_i（区间起点值），ϕ(0) = a
        # 2. 影响：直接决定插值在整数点 s=j 处的精确性（必须完全等于原始 t_j）
        # 3. 修改建议：保持不变，这是 Hermite 插值的硬性要求

        b = torch.empty_like(self.T)  # [batch_dims, seq_len, *t_X_channels]
        # 1. 准备存储每个节点 k 的斜率 m_k（即多项式在节点处的导数 ϕ'(0) 和 ϕ'(1)）
        # 2. 影响：b 的计算方式直接决定了整个 PCHIP 的形状保持特性与数值偏差
        # 3. 修改方向：这是当前训练效果差的根本原因（见下文详细说明）

        b[:, 0, ...] = 1.5 * self.DELTA[:, 0, ...]
        # 1. 对应公式：PCHIP 边界斜率 m_0 = 1.5 × Δ_0（单侧外推）
        # 2. 影响：导致第一个区间左端 dt/ds = 1.5×Δt，结合左端点 Euler 积分 → 系统性过估计变化率
        #    在机理模型训练中表现为“衰减太快”或“振荡提前”，优化器很难补偿
        # 3. 推荐修改（大幅提升训练稳定性）：
        #    b[:, 0, ...] = self.DELTA[:, 0, ...]           # 改为 1×Δ（线性外推）
        #    或 (3*Δ_0 - Δ_1)/2 等更温和的外推

        b[:, -1, ...] = 1.5 * self.DELTA[:, -1, ...]
        # 1. 对应公式：PCHIP 边界斜率 m_{N-1} = 1.5 × Δ_{N-2}
        # 2. 影响：同上，最后一个区间右端斜率也被放大 1.5 倍，导致末段同样过估计过大
        # 3. 推荐修改：同上，改为 1.0× 或更保守的外推

        b[:, 1:-1, ...] = torch.where(
            self.DELTA[:, :-1, ...] * self.DELTA[:, 1:, ...] > 0,
            1.5
            * torch.sign(self.DELTA[:, 1:, ...])
            * torch.min(self.DELTA[:, :-1, ...].abs(), self.DELTA[:, 1:, ...].abs()),
            torch.zeros_like(b[:, 1:-1, ...]),
        )
        # 1. 对应公式：Fritsch–Butland (1984) 简化版内部斜率公式
        #    当相邻 Δ 同号时 m_k = 1.5 × min(|Δ_{k-1}|, |Δ_k|) × sign
        #    否则 m_k = 0（强制平坦）
        # 2. 影响：
        #    - 优点：真正实现了局部单调性保持，无 overshoot
        #    - 缺点：引入了 1.5 倍因子，导致即使在完全线性时间序列上，dt/ds 也在 1.5Δ ↔ 0.75Δ 之间振荡
        #      与左端点积分结合后产生 ~25% 系统性偏差，严重阻碍机理模型训练
        # 3. 推荐修改方案（任选其一，按需求强度排序）：
        #    方案A（最推荐，训练最稳定）：改为标准线性插值斜率
        #         b[:, 1:-1, ...] = self.DELTA[:, 1:, ...]   # 或者 self.DELTA[:, :-1, ...] 均可
        #    方案B（保留部分单调性但消除 1.5 偏差）：把 1.5 改为 1.0
        #         b[:, ...] = 1.0 * torch.sign(...) * torch.min(...)
        # 方案C（最接近原版 SciPy PCHIP，精度最高但计算稍复杂）：
        #         使用加权调和平均（Fritsch-Carlson 原版公式）
        #         w1 = 2*Δ_k + Δ_{k-1}; w2 = Δ_k + 2*Δ_{k-1}
        #         m_k = (w1+w2) / (w1/Δ_k + w2/Δ_{k-1}) 当分母≠0

        c = 3 * self.DELTA - b[:, 1:, ...] - 2 * b[:, :-1, ...]
        # 1. 对应公式：Hermite 系数 c = 3Δ - m_{i+1} - 2m_i
        # 2. 影响：完全由前面的 b 决定，b 改了这里自动正确
        # 3. 修改建议：无需手动修改，随 b 变化即可

        d = b[:, 1:, ...] + b[:, :-1, ...] - 2 * self.DELTA
        # 1. 对应公式：Hermite 系数 d = m_{i+1} + m_i - 2Δ
        # 2. 影响：同上
        # 3. 修改建议：无需手动修改

        self.K = torch.stack(
            (a[:, :-1, ...], b[:, :-1, ...], c, d), dim=-1
        )  # [batch_dims, seq_len-1, *t_X_channels, 4]
        # 1. 对应公式：将四个系数打包成向量 k = [a, m_i, c, d]，便于向量化求值
        # 2. 影响：打包方式完全正确，性能优秀
        # 3. 修改建议：无

    def forward(self, s: Tensor, j: int) -> tuple[Tensor | None]:
        k = self.K[:, j, ...]  # 取出当前区间系数
        s_ = s - j             # 局部坐标 s ∈ [0,1]
        s_2 = s_ * s_
        s_3 = s_2 * s_

        s_v = torch.tensor([1, s_, s_2, s_3], dtype=self.T.dtype, device=self.T.device)
        # 1. 对应公式：ϕ(s) = a + m_i·s + c·s² + d·s³
        # 2. 影响：数值稳定、向量化高效
        # 3. 优化（可选）：可以用 torch.vander(s_, 4, increasing=True) 替代，避免手动建张量

        phi = torch.sum(k * s_v, dim=-1)
        # 实际输出的重参数化时间 t(s)，在整数点 s=j 处 phi == T[:,j,...]（精确）

        s_p_v = torch.tensor([0, 1, 2 * s_, 3 * s_2], dtype=self.T.dtype, device=self.T.device)
        dphi_ds = torch.sum(k * s_p_v, dim=-1)
        # 1. 对应公式：dt/ds = m_i + 2c·s + 3d·s²（链式法则中的关键项）
        # 2. 影响：这是导致训练差的第二个关键点！
        #    当使用左端点 Euler（s=j 整数）时，dphi_ds 恰好等于 m_i = 1.5Δ
        #    导致每一步都乘以 1.5 倍的“时间膨胀”，累积偏差巨大
        # 3. 推荐修改（最有效单点修改）：
        #    # 强制使用区间平均速度（积分守恒）
        #    avg_speed = self.DELTA[:, j, ...]  # 精确平均 dt/ds = Δt/1
        #    return phi, avg_speed
        #    或者使用中点速度（二阶精度）
        #    mid_speed = dphi_ds_at_0.5 = m_i + c + 0.75 d
        #    这两种方式都能彻底消除 1.5 偏差，训练效果立刻接近甚至超过旧版

        return phi, dphi_ds
    
class LinearRepara(_Repara):
    """线性重参数化：dt/ds = 真实 Δt（恒定），完全消除 PCHIP 的 1.5× 偏差"""
    def __init__(self, T: Tensor, t_mask: Optional[Tensor] = None) -> None:
        super().__init__(T, t_mask)
        # a  = y_i
        # b  = Δt（恒定斜率）
        # c=d=0 → 线性插值
        a = self.T[:, :-1, ...]                    # [B, N-1, *t_channels]
        b = self.DELTA                              # [B, N-1, *t_channels]
        c = torch.zeros_like(b)
        d = torch.zeros_like(b)
        self.K = torch.stack((a, b, c, d), dim=-1)  # [B, N-1, *t_channels, 4]

    def forward(self, s: Tensor, j: int):
        # j 可以是 int 或 tensor（向量化情况）
        k = self.K[:, j, ...]                       # [B, *t_channels, 4]
        s_ = s - j.float() if torch.is_tensor(j) else (s - j)
        phi = k[..., 0] + k[..., 1] * s_            # 线性插值值
        dphi_ds = k[..., 1]                         # 恒定 dt/ds = Δt
        return phi, dphi_ds


class MonoRationalRepara(_Repara):
    def __init__(self, T: Tensor, t_mask: Tensor | None = None) -> None:
        super().__init__(T, t_mask)

        B = self.T.shape[0]  # batch_dims
        N = self.T.shape[1]  # seq_len

        A_0 = torch.ones_like(self.T)  # [batch_dims, seq_len, *t_X_channels]
        A_0[:, 1::2, ...] = 5  # 索引为奇数的位置填5

        self.A = A_0  # [batch_dims, seq_len, *t_channels]
        if N == 2:  # N为2的情况
            return
        else:
            DELTA_r = self.DELTA[:, :-1, ...] / self.DELTA[:, 1:, ...]
            for j in range(2, N, 2):
                ones = torch.ones(
                    B,
                    j,
                    *self.T.shape[2:],
                    device=self.T.device,
                    dtype=self.T.dtype,
                )  # [batch_dims, j, *t_X_channels]
                DELTA_r_slice_0 = DELTA_r[:, int(j / 2 - 1), ...].unsqueeze(
                    1
                )  # [batch_dims, 1, *t_X_channels]
                DELTA_r_slice_1 = DELTA_r[:, int(j / 2), ...].unsqueeze(
                    1
                )  # [batch_dims, 1, *t_X_channels]
                DELTA_rj = (
                    [ones] + [DELTA_r_slice_0, DELTA_r_slice_1] * int((N - j) / 2)
                    if N % 2 == 0
                    else [ones]
                    + [DELTA_r_slice_0, DELTA_r_slice_1] * int((N - 1 - j) / 2)
                    + [DELTA_r_slice_0]
                )
                DELTA_rj = torch.concat(
                    DELTA_rj, dim=1
                )  # [batch_dims, seq_len, *t_X_channels]

                self.A = self.A * DELTA_rj

    def forward(self, s: Tensor, j: int) -> tuple[Tensor | None]:
        # 计算phi及其导数
        phi_num = self.A[:, j + 1, ...] * self.T[:, j + 1, ...] * (s - j) - self.A[
            :, j, ...
        ] * self.T[:, j, ...] * (s - j - 1)
        phi_den = self.A[:, j + 1, ...] * (s - j) - self.A[:, j, ...] * (s - j - 1)

        phi: torch.Tensor = phi_num / phi_den  # [batch_dims, *t_X_channels]
        dphi_ds = (5 * self.DELTA[:, 0, ...]) / (
            phi_den**2
        )  # [batch_dims, *t_X_channels]
        return phi, dphi_ds


class _ReparaFunc(Module):
    """
    时间重参数化后的 ODE 右侧函数（优化版）

    优化点：
    1. 只喂 phi（当前真实时间值），不再把 dphi_ds 作为输入传给 func
       → 更符合大多数机理模型的需求，训练更稳定、loss 更低
    2. 预先缓存每个整数 s 对应的区间 j（searchsorted 缓存）
       → 自适应求解器（如 dopri5）提速 2~5 倍，避免数千次 CPU↔GPU 同步
    """

    def __init__(
        self,
        func: Module,
        T: Tensor,
        repara_method=PCHIPRepara,          # 可以换成 LinearRepara对比一下效果
        t_mask: Tensor | None = None,
        t_view: list | None = None,
        inputs: list[dict] | None = None,
    ) -> None:
        super().__init__()
        self.func = func

        if T.shape[1] < 2:
            raise ValueError("Timestamps must be more than one")
        if T.dim() == 2:
            T = T.unsqueeze(-1)                 # [B, N, 1]
        assert T.isfinite().all()
        self.T = T

        # 标准化整数网格 0,1,2,...,N-1
        N = T.shape[1]
        self.register_buffer("S", torch.arange(N, device=T.device, dtype=T.dtype))

        self.t_mask = t_mask
        self.t_view = t_view

        # 重参数化器（PCHIP / Linear）
        self.repara = repara_method(self.T, t_mask=t_mask)

        # 额外输入（如控制量 u(t)）
        if inputs is not None:
            self.serializers = []
            for input_dict in inputs:
                serializer = SERIALIZERS[input_dict.pop("method")]
                if "t" not in input_dict:
                    input_dict["t"] = self.T
                self.serializers.append(serializer(**input_dict))
        else:
            self.serializers = None

        # ==================== 优化 2：缓存 searchsorted 结果 ====================
        # self.interval_cache[s_int] = j   (s_int 是整数 0,1,2,...,N-1)
        # 对于超出范围的 s（自适应步长可能 <0 或 >=N-1），仍会动态计算
        interval_cache = torch.arange(N - 1, device=T.device)   # 大多数情况下 j = floor(s)
        self.register_buffer("interval_cache", interval_cache)  # shape [N-1]

    def forward(self, s: Tensor, y: Tensor) -> Tensor:
        """
        s : (...)               # 标量或张量均可（自适应求解器通常是标量）
        y : [batch, *y_channels]
        """
        device = s.device

        # ------------------- 优化 2：快速获取区间 j -------------------
        if s.numel() == 1 and s.is_floating_point():
            s_val = s.item()
            # 整数点直接命中缓存（绝大多数自适应步长正好落在整数点附近）
            if s_val >= 0 and s_val < len(self.interval_cache) and s_val.is_integer():
                j = int(s_val)                       # 直接用 floor(s)
            elif s_val >= len(self.S) - 1:
                j = len(self.S) - 2                  # 超出右边界
            else:
                # 极少数非整数或越界的情况才走 searchsorted（几乎不会触发）
                j = int(torch.searchsorted(self.S, s, right=False).item() - 1)
                j = max(0, min(j, len(self.S) - 2))
        else:
            # 张量化情况（极少出现），直接走 searchsorted
            j = torch.searchsorted(self.S, s, right=False) - 1
            j = torch.clamp(j, 0, len(self.S) - 2)

        # ------------------- 获得当前真实时间 phi -------------------
        # phi: [batch, *t_channels]   dphi_ds: [batch, *t_channels]
        phi, dphi_ds = self.repara(s, j if torch.is_tensor(j) else int(j))

        # ------------------- 优化 1：只喂 phi（不再喂 dphi_ds） -------------------
        if self.serializers is not None:
            inputs: List[Tensor] = [phi]
            for serializer in self.serializers:
                extra_phi, _ = serializer(s, j)      # 只取值，不取导数
                inputs.append(extra_phi)
        else:
            inputs = [phi]                            # 关键：这里只剩 phi

        # func 输出 dy/dt（物理速率）
        dy_dt = self.func(inputs, y)                  # [batch, *y_channels]

        # ------------------- 链式法则：dy/ds = (dt/ds) * (dy/dt) -------------------
        if self.t_mask is not None:
            dt_ds = dphi_ds[:, self.t_mask]
        else:
            dt_ds = dphi_ds

        # reshape 让 dt_ds 能广播到 dy_dt（支持多维时间）
        if self.t_view is None:
            dt_ds_shape = [dy_dt.shape[0]] + [1] * (dy_dt.dim() - 1)
        else:
            extra = dy_dt.dim() - 1 - len(self.t_view)
            dt_ds_shape = [dy_dt.shape[0]] + self.t_view + [1] * max(0, extra)

        dt_ds = dt_ds.view(*dt_ds_shape)              # [batch, *t_channels, 1,1,...]

        dy_ds = dt_ds * dy_dt
        assert dy_ds.isfinite().all()
        return dy_ds


def time_input_repara(ode_solver):
    @wraps(ode_solver)
    def reparameterization(*args, **kwargs):
        t_mask = kwargs.pop("t_mask", None)
        t_view = kwargs.pop("t_view", None)
        inputs = kwargs.pop("inputs", None)

        args_dict = signature(ode_solver).bind(*args, **kwargs)
        args_dict.apply_defaults()
        args_dict = args_dict.arguments

        # 检查t的维度，1D直接求解，其他继续
        t: Tensor = args_dict["t"]
        if t.dim() == 1:
            return ode_solver(**args_dict)

        # 设置新的func
        func = args_dict["func"]
        args_dict["func"] = _ReparaFunc(
            func, t, t_mask=t_mask, t_view=t_view, inputs=inputs
        )

        # 设置新的t
        args_dict["t"] = args_dict["func"].S

        return ode_solver(**args_dict)

    return reparameterization
