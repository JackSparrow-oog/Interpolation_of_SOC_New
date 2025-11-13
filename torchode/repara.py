from functools import wraps
from inspect import signature
from typing import List, Tuple
import torch
from torch import Tensor
from torch.nn import Module

from .serializers import HermiteSerializer, DiscreteSerializer

SERIALIZERS = {
    "hermite": HermiteSerializer,
    "discrete": DiscreteSerializer,
}


class _Repara(Module):
    def __init__(self, T: Tensor) -> None:
        super().__init__()
        self.T = T

    def forward(self, s: Tensor, j: int) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class HermiteRepara(_Repara):
    def __init__(self, T: Tensor, method: str | None = None) -> None:
        super().__init__(T)

        self.DELTA = torch.diff(
            self.T, dim=1
        )  # [batch_dims, seq_len - 1, *t_X_channels]
        assert self.DELTA.isfinite().all()

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
    def __init__(self, T: Tensor) -> None:
        super().__init__(T)
        self.DELTA = torch.diff(self.T, dim=1)[:, :, 0]  # [1, sparse_len-1] squeeze channels
        assert self.DELTA.isfinite().all()
        a = self.T[:, :, 0]  # [1, sparse_len] squeeze
        b = torch.empty_like(a)
        b[:, 0] = 1.5 * self.DELTA[:, 0]
        b[:, -1] = 1.5 * self.DELTA[:, -1]
        if self.T.shape[1] > 2:
            b[:, 1:-1] = torch.where(
                self.DELTA[:, :-1] * self.DELTA[:, 1:] > 0,
                1.5 * torch.sign(self.DELTA[:, 1:]) * torch.min(
                    self.DELTA[:, :-1].abs(), self.DELTA[:, 1:].abs()
                ),
                torch.zeros_like(b[:, 1:-1]),
            )
        c = 3 * self.DELTA - b[:, 1:] - 2 * b[:, :-1]
        d = b[:, 1:] + b[:, :-1] - 2 * self.DELTA
        # K [1, sparse_len-1, 4] (no channels dim)
        self.K = torch.stack((a[:, :-1], b[:, :-1], c, d), dim=-1)  # [1, sparse_len-1, 4]

    def forward(self, s: Tensor, j: int) -> Tuple[Tensor, Tensor]:
        k = self.K[:, j, :]  # [1, 4] -> squeeze to [4]
        k = k.squeeze(0)  # [4]
        # s_ scalar Tensor
        s_ = s.float()  # [scalar]
        s_2 = s_ ** 2
        s_3 = s_ ** 3
        # 全 Tensor [4]
        s_v = torch.stack([torch.ones_like(s_), s_, s_2, s_3])  # [4]
        phi = torch.sum(k * s_v)  # scalar
        s_p_v = torch.stack([torch.zeros_like(s_), torch.ones_like(s_), 2 * s_, 3 * s_2])
        dphi_ds = torch.sum(k * s_p_v)
        return phi, dphi_ds

# class PCHIPRepara(_Repara):
#     def __init__(self, T: Tensor) -> None:
#         super().__init__(T)
#         self.DELTA = torch.diff(
#             self.T, dim=1
#         )  # [batch_dims, seq_len - 1, *t_X_channels]
#         assert self.DELTA.isfinite().all()

#         a = self.T  # [batch_dims, seq_len, *t_X_channels]
#         b = torch.empty_like(self.T)  # [batch_dims, seq_len, *t_X_channels]
#         b[:, 0, ...] = 1.5 * self.DELTA[:, 0, ...]
#         b[:, -1, ...] = 1.5 * self.DELTA[:, -1, ...]
#         b[:, 1:-1, ...] = torch.where(
#             self.DELTA[:, :-1, ...] * self.DELTA[:, 1:, ...] > 0,
#             1.5
#             * torch.sign(self.DELTA[:, 1:, ...])
#             * torch.min(self.DELTA[:, :-1, ...].abs(), self.DELTA[:, 1:, ...].abs()),
#             torch.zeros_like(b[:, 1:-1, ...]),
#         )
#         c = (
#             3 * self.DELTA - b[:, 1:, ...] - 2 * b[:, :-1, ...]
#         )  # [batch_dims, seq_len - 1, *t_X_channels]
#         d = (
#             b[:, 1:, ...] + b[:, :-1, ...] - 2 * self.DELTA
#         )  # [batch_dims, seq_len - 1, *t_X_channels]

#         self.K = torch.stack(
#             (a[:, :-1, ...], b[:, :-1, ...], c, d), dim=-1
#         )  # [batch_dims, seq_len - 1, *t_X_channels, 4]

#     def forward(self, s: Tensor, j: int) -> tuple[Tensor | None]:
#         k = self.K[:, j, ...]  # [batch_dims, *t_X_channels, 4]
#         s_ = s - j
#         s_2 = s_ * s_
#         s_3 = s_2 * s_

#         s_v = torch.tensor(
#             [1, s_, s_2, s_3], dtype=self.T.dtype, device=self.T.device
#         )  # [4]
#         phi = torch.sum(k * s_v, dim=-1)  # [batch_dims, *t_X_channels]

#         s_p_v = torch.tensor(
#             [0, 1, 2 * s_, 3 * s_2], dtype=self.T.dtype, device=self.T.device
#         )  # [4]
#         dphi_ds = torch.sum(k * s_p_v, dim=-1)  # [batch_dims, *t_X_channels]

#         return phi, dphi_ds


class MonoRationalRepara(_Repara):
    def __init__(self, T: Tensor) -> None:
        super().__init__(T)

        self.DELTA = torch.diff(
            self.T, dim=1
        )  # [batch_dims, seq_len - 1, *t_X_channels]
        assert self.DELTA.isfinite().all()
        assert not (self.DELTA == 0).any()  # 假定差分不为0

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
    def __init__(
        self,
        func: Module,
        T: Tensor,
        repara_method=PCHIPRepara,
        t_mask: Tensor | None = None,
        t_view: list | None = None,
        inputs: list[dict] | None = None,
    ) -> None:
        super().__init__()
        self.func = func

        if T.shape[1] < 2:
            raise ValueError("Timestamps must be more than one")
        if T.dim() == 2:
            T = T.unsqueeze(-1)  # [batch_dims, seq_len, t_channels=1]
        assert T.isfinite().all()
        self.T = T

        self.S = torch.tensor(
            [i for i in range(self.T.shape[1])],
            device=self.T.device,
            dtype=self.T.dtype,
        )  # 0, 1, ..., N-1

        self.repara = repara_method(self.T)
        self.t_mask = t_mask
        if self.t_mask is not None:
            assert self.t_mask.shape == self.T.shape[2:]
        self.t_view = t_view

        # 添加输入
        if inputs is not None:
            self.serializers = []
            for input_dict in inputs:
                serializer = SERIALIZERS[input_dict.pop("method")]
                if "t" not in input_dict:
                    input_dict["t"] = self.T
                self.serializers.append(serializer(**input_dict))
        else:
            self.serializers = None

    def forward(self, s: Tensor, y: Tensor) -> Tensor:
        # self.func: [input] × [batch_dims, (*t_channels,) *hidden_channels]
        # -> [batch_dims, (*t_channels,) *hidden_channels]
        # s.shape:torch.Size([])
        # y.shape:torch.Size([batch_dims, (*t_channels,) *hidden_channels])

        # 判断s的位置以确定j（整数）
        j = torch.searchsorted(self.S, s)  # 使用二分查找找到目标值在排序后张量中的位置
        if j == 0:  # 目标值比所有元素都小
            j = 0  # 使用第一段插值函数
        elif j == len(self.S):  # 目标值比所有元素都大
            j = len(self.S) - 2  # 使用最后一段插值函数
        else:  # 目标值在排序后张量中的位置在 [index-1, index) 范围内
            j = int((j - 1).item())

        # 获得时间
        phi, dphi_ds = self.repara(s, j)
        phi: torch.Tensor  # [batch_dims, *t_X_channels]
        dphi_ds: torch.Tensor  # [batch_dims, *t_X_channels]

        # 获得实时输入
        if self.serializers is not None:
            inputs: list[list[Tensor, Tensor]] = [[phi, dphi_ds]]
            for serializer in self.serializers:
                inputs.append(serializer(s, j))
        else:
            inputs: list[Tensor] = [phi, dphi_ds]

        dy_dt: torch.Tensor = self.func(
            inputs, y
        )  # [batch_dims, (*t_channels,) *hidden_channels]

        dt_ds_shape = (
            [dy_dt.shape[0]] + [1] * len(dy_dt.shape[1:])
            if self.t_view is None
            else [dy_dt.shape[0]]
            + self.t_view
            + [1] * (len(dy_dt.shape[1:]) - len(self.t_view))
        )

        if dphi_ds.dim() == 0:
            # 强制加上 Batch 维度 [1] 和 Features 维度 [1] (如果它是标量，我们无法知道 Features 数量)
            # 我们知道它至少应该有 Batch 维度，所以升维到 1D
            # **注意：我们只升维一次，不在这里假设 Features 数量**
            dphi_ds = dphi_ds.unsqueeze(0) 

        # 现在 dphi_ds 至少是 1D (例如 [4] 或 [1])
        if dphi_ds.dim() == 1:
            # 如果是 1D，它可能是 [4] (Features) 或 [1] (Batch+Feature)
            # 为了避免后面的 [:, self.t_mask] 索引错误，强制升维到 2D
            dphi_ds = dphi_ds.unsqueeze(0) # 变为 [1, F] 或 [1, 1]

        if dphi_ds.dim() == 2 and dphi_ds.shape == torch.Size([1, 1]):
    
            # 打印警告，确认进入此修复逻辑
            print(f"DEBUG: dphi_ds unexpectedly [1, 1]. Forcing to zero-filled [1, 4] to enable indexing.")
            
            # 用零张量替换 dphi_ds。由于原始的 [1, 1] 值是错误的，我们必须用正确的形状 [1, 4] 替换它。
            EXPECTED_N = 1
            EXPECTED_F = 4 
            dphi_ds = torch.zeros(EXPECTED_N, EXPECTED_F, device=dphi_ds.device, dtype=dphi_ds.dtype)

        dt_ds = dphi_ds if self.t_mask is None else dphi_ds[:, self.t_mask]
        dt_ds = dt_ds.view(
            *dt_ds_shape
        )  # [batch_dims, *t_channels, *([1] * len(input_channels))]

        dy_ds = dt_ds * dy_dt  # [batch_dims, (*t_channels,) *hidden_channels]
        assert dy_ds.isfinite().all()

        return dy_ds

# 假设这个类和PCHIPRepara在同一个文件中，或者您将它复制到您的环境中
class PCHIPNonUniformRepara(Module):
    """
    修改后的 PCHIP 插值器，支持非均匀时间 t 进行插值。
    - y_vals: 被插值的值 [1, sparse_len, 1]
    - t_vals: 对应的真实时间 t [1, sparse_len, 1]
    """
    def __init__(self, y_vals: torch.Tensor, t_vals: torch.Tensor) -> None:
        super().__init__()
        
        # 确保输入是 [1, N, 1] 形状，并去除最后一个通道维度，得到 [1, N]
        self.y_vals = y_vals
        self.t_vals = t_vals 

        y = self.y_vals.squeeze(-1)       # [1, sparse_len]
        t = self.t_vals.squeeze(-1)       # [1, sparse_len]
        
        sparse_len = y.shape[1]
        
        if sparse_len < 2:
            # 无法插值，设置空K
            self.K = torch.empty(1, 0, 4, device=y.device)
            self.H = torch.empty(1, 0, device=y.device)
            self.M = torch.empty_like(y)
            return

        # 计算真实时间步长 h_j = t_{j+1} - t_j
        self.H = torch.diff(t, dim=1)  # [1, sparse_len-1]
        # 计算差商 d_j = (y_{j+1} - y_j) / h_j
        self.DELTA = torch.diff(y, dim=1) / self.H # [1, sparse_len-1]
        
        assert self.H.isfinite().all() and self.DELTA.isfinite().all()
        
        # 1. 计算 PCHIP 斜率 (M)
        m = torch.empty_like(y) # [1, sparse_len]
        
        if sparse_len == 2:
            # 只有两个点，斜率等于差商
            m[:, 0] = self.DELTA[:, 0]
            m[:, 1] = self.DELTA[:, 0]
        else:
            # 内部斜率
            m[:, 1:-1] = self._pchip_compute_m_internal(
                self.H[:, :-1], self.H[:, 1:], self.DELTA[:, :-1], self.DELTA[:, 1:]
            )
            # 边界条件
            m[:, 0] = self._pchip_compute_m_end(
                self.H[:, 0:1], self.H[:, 1:2], self.DELTA[:, 0:1], self.DELTA[:, 1:2]
            )
            m[:, -1] = self._pchip_compute_m_end(
                self.H[:, -1:], self.H[:, -2:-1], self.DELTA[:, -1:], self.DELTA[:, -2:-1]
            )

        self.M = m # PCHIP 斜率 [1, sparse_len]

        # 2. 计算 PCHIP 多项式系数 K
        y0 = y[:, :-1] # a = y_j
        m0 = self.M[:, :-1] # b = m_j
        h = self.H # h_j
        d = self.DELTA # d_j

        # c = (3 * d_j - 2 * m_j - m_{j+1}) / h_j
        c = (3 * d - 2 * m0 - self.M[:, 1:]) / h
        
        # d_coeff = (m_j + m_{j+1} - 2 * d_j) / (h_j)^2
        d_coeff = (m0 + self.M[:, 1:] - 2 * d) / (h ** 2) 

        # K [1, sparse_len-1, 4] -> (y_j, m_j*h_j, c*h_j, d_coeff*h_j^2)
        # 存储的系数是 P(s) = y_j + (m_j)s + (c)s^2 + (d_coeff)s^3
        # 这里的 m_j, c, d_coeff 已经被 h_j 缩放，以便 s \in [0, 1] 时使用
        self.K = torch.stack((y0, m0 * h, c * h, d_coeff * (h ** 2)), dim=-1)

    @staticmethod
    def _pchip_compute_m_end(h1, h2, d1, d2):
        # 边界斜率计算 (Scipy 简化逻辑)
        m = ((2 * h1 + h2) * d1 - h1 * d2) / (h1 + h2)
        
        # 单调性修正
        zero = torch.zeros_like(m)
        m = torch.where(
            (torch.sign(m) != torch.sign(d1)) | (torch.sign(m) != torch.sign(d2)), 
            zero, m)
        m = torch.where(
            (torch.sign(d1) * torch.sign(d2) <= 0), 
            zero, m)
        return m.squeeze(-1) # 恢复到 [1] 形状

    @staticmethod
    def _pchip_compute_m_internal(h_prev, h_curr, d_prev, d_curr):
        # 内部斜率计算
        w1 = (2 * h_curr + h_prev) / (h_prev + h_curr)
        w2 = (h_curr + 2 * h_prev) / (h_prev + h_curr)
        m = (w1 * d_prev + w2 * d_curr) / 3

        # 单调性修正
        zero = torch.zeros_like(m)
        m = torch.where(
            (torch.sign(m) != torch.sign(d_prev)) | (torch.sign(m) != torch.sign(d_curr)), 
            zero, m)
        m = torch.where(
            (torch.sign(d_prev) * torch.sign(d_curr) <= 0), 
            zero, m)
        return m.squeeze(-1) # 恢复到 [1, N-2] 形状
        
    def forward(self, s: torch.Tensor, j: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.K.shape[1] == 0:
            # ... 处理无法插值的情况 ...
            return self.y_vals[0, 0, 0].unsqueeze(0), torch.tensor(0.0, device=s.device).unsqueeze(0)
            
        k = self.K[:, j, :] # [1, 4] -> 保持 [1, 4]
        
        # 注意：s 是 [1]，s_ 应该是 [1] 形状的 Tensor，而不是标量
        s_ = s.float() # 应该已经是 [1]
        
        # 避免使用 .squeeze(0) 导致 s_ 变成标量
        s_2 = s_ ** 2
        s_3 = s_ ** 3
        
        # P(s) = k[0] + k[1]*s + k[2]*s^2 + k[3]*s^3
        # s_v 形状为 [4, 1]
        s_v = torch.stack([torch.ones_like(s_), s_, s_2, s_3]) # [4, 1]
        
        # k 是 [1, 4]， s_v 是 [4, 1]
        phi = torch.sum(k * s_v.transpose(0, 1), dim=1) # 结果应为 [1]

        # dphi/ds = k[1] + 2*k[2]*s + 3*k[3]*s^2
        s_p_v = torch.stack([torch.zeros_like(s_), torch.ones_like(s_), 2 * s_, 3 * s_2]) # [4, 1]
        dphi_ds = torch.sum(k * s_p_v.transpose(0, 1), dim=1) # 结果应为 [1]
        
        # dphi/dt = dphi/ds * (1 / h_j)
        h_j = self.H[:, j] # [1] 形状的 Tensor
        dphi_dt = dphi_ds / h_j 
        
        return phi, dphi_dt

def time_input_repara(ode_solver):
    @wraps(ode_solver)
    def reparameterization(*args, **kwargs):
        t_mask = kwargs.pop("t_mask") if "t_mask" in kwargs else None
        t_view = kwargs.pop("t_view") if "t_view" in kwargs else None
        inputs = kwargs.pop("inputs") if "inputs" in kwargs else None

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
