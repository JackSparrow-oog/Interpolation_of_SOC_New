from functools import wraps
from inspect import signature

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

    def forward(self, s: Tensor, j: int) -> tuple[Tensor | None]:
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
        self.DELTA = torch.diff(
            self.T, dim=1
        )  # [batch_dims, seq_len - 1, *t_X_channels]
        assert self.DELTA.isfinite().all()

        a = self.T  # [batch_dims, seq_len, *t_X_channels]
        b = torch.empty_like(self.T)  # [batch_dims, seq_len, *t_X_channels]
        b[:, 0, ...] = 1.5 * self.DELTA[:, 0, ...]
        b[:, -1, ...] = 1.5 * self.DELTA[:, -1, ...]
        b[:, 1:-1, ...] = torch.where(
            self.DELTA[:, :-1, ...] * self.DELTA[:, 1:, ...] > 0,
            1.5
            * torch.sign(self.DELTA[:, 1:, ...])
            * torch.min(self.DELTA[:, :-1, ...].abs(), self.DELTA[:, 1:, ...].abs()),
            torch.zeros_like(b[:, 1:-1, ...]),
        )
        c = (
            3 * self.DELTA - b[:, 1:, ...] - 2 * b[:, :-1, ...]
        )  # [batch_dims, seq_len - 1, *t_X_channels]
        d = (
            b[:, 1:, ...] + b[:, :-1, ...] - 2 * self.DELTA
        )  # [batch_dims, seq_len - 1, *t_X_channels]

        self.K = torch.stack(
            (a[:, :-1, ...], b[:, :-1, ...], c, d), dim=-1
        )  # [batch_dims, seq_len - 1, *t_X_channels, 4]

    def forward(self, s: Tensor, j: int) -> tuple[Tensor | None]:
        k = self.K[:, j, ...]  # [batch_dims, *t_X_channels, 4]
        s_ = s - j
        s_2 = s_ * s_
        s_3 = s_2 * s_

        s_v = torch.tensor(
            [1, s_, s_2, s_3], dtype=self.T.dtype, device=self.T.device
        )  # [4]
        phi = torch.sum(k * s_v, dim=-1)  # [batch_dims, *t_X_channels]

        s_p_v = torch.tensor(
            [0, 1, 2 * s_, 3 * s_2], dtype=self.T.dtype, device=self.T.device
        )  # [4]
        dphi_ds = torch.sum(k * s_p_v, dim=-1)  # [batch_dims, *t_X_channels]

        return phi, dphi_ds


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
        dt_ds = dphi_ds if self.t_mask is None else dphi_ds[:, self.t_mask]
        dt_ds = dt_ds.view(
            *dt_ds_shape
        )  # [batch_dims, *t_channels, *([1] * len(input_channels))]

        dy_ds = dt_ds * dy_dt  # [batch_dims, (*t_channels,) *hidden_channels]
        assert dy_ds.isfinite().all()

        return dy_ds


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
