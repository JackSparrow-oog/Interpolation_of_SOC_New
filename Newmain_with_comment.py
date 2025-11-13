import math
import random
from pathlib import Path
from datetime import date
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple, Optional
from unittest.mock import Mock
from warnings import simplefilter
import time  # Added for potential timing logs

# 假设utils.metrics (这些是自定义的评估指标函数，用于计算模型性能)
def mean_absolute_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    计算平均绝对误差 (MAE)。
    
    参数:
    - y_pred: 模型预测值，形状 [batch_size * seq_len, 1] 或类似。
    - y_true: 真实值，形状与 y_pred 相同。
    
    返回:
    - MAE 标量张量。
    """
    return torch.mean(torch.abs(y_pred - y_true))

def mean_absolute_percentage_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    计算平均绝对百分比误差 (MAPE)，添加小 epsilon 避免除零。
    
    参数:
    - y_pred: 模型预测值，形状 [batch_size * seq_len, 1] 或类似。
    - y_true: 真实值，形状与 y_pred 相同。
    
    返回:
    - MAPE 标量张量（百分比形式）。
    """
    return torch.mean(torch.abs((y_pred - y_true) / (y_true + 1e-8))) * 100

def root_mean_squared_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    计算根均方误差 (RMSE)。
    
    参数:
    - y_pred: 模型预测值，形状 [batch_size * seq_len, 1] 或类似。
    - y_true: 真实值，形状与 y_pred 相同。
    
    返回:
    - RMSE 标量张量。
    """
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

from torchdiffeq import odeint  # 需要pip install torchdiffeq (用于 ODE 求解，但本代码中未直接使用，转而用手动 Euler 积分)

# 设置 PyTorch 异常检测以便调试梯度问题
torch.autograd.set_detect_anomaly(True)

# 设置随机种子以确保实验可复现
random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)

# 选择设备：优先使用 GPU (cuda:0)，否则 CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 忽略 FutureWarning 以保持输出干净
simplefilter(action="ignore", category=FutureWarning)


def init_network_weights(m: nn.Module) -> None:
    """
    使用 Xavier 初始化线性层权重，并将偏置初始化为零。
    这有助于网络训练的稳定性和收敛。
    
    参数:
    - m: nn.Module 的子模块，通常是 nn.Linear。
    """
    # 只对线性层应用初始化
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier 均匀初始化权重
        nn.init.zeros_(m.bias)  # 偏置初始化为零


class LogCoshLoss(nn.Module):
    """
    LogCosh 损失函数：类似于 Huber 损失，对异常值鲁棒。
    公式：mean(delta + softplus(-2*delta) - log(2))，其中 delta = y_pred - y_true。
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 LogCosh 损失。
        
        参数:
        - y_pred: 预测 SOC，形状 [batch_size, seq_len, 1]。
        - y_true: 真实 SOC，形状与 y_pred 相同。
        
        返回:
        - 标量损失张量（批次平均）。
        """
        delta = y_pred - y_true  # [batch_size, seq_len, 1]，误差 delta
        return torch.mean(delta + torch.nn.functional.softplus(-2.0 * delta) - math.log(2.0))


class SOCODEFunc(nn.Module):
    """
    SOC ODE 函数：定义 dSOC/dt = (eta_0 * (1 + delta_eta)) * I / (3600 * Q)
    这是电池 SOC 动态的物理模型，用于积分求解 SOC 轨迹。
    """
    def __init__(self) -> None:
        super().__init__()
        # SC 参数：Q (容量), eta_0 (初始效率)，在 set_sc_params 中设置
        self.Q: Optional[torch.Tensor] = None
        self.eta_0: Optional[torch.Tensor] = None

    def set_sc_params(self, Q: torch.Tensor, eta_0: torch.Tensor):
        """
        设置超级电容 (SC) 参数，用于 ODE 函数。
        
        参数:
        - Q: 电池容量，形状 [batch_size, 1]。
        - eta_0: 初始库仑效率，形状 [batch_size, 1]。
        """
        self.Q = Q  # [batch_size, 1]
        self.eta_0 = eta_0  # [batch_size, 1]

    def forward(self, x: list[torch.Tensor], SOC: torch.Tensor) -> torch.Tensor:
        """
        计算 dSOC/dt。
        
        参数:
        - x: 输入列表，x[0] 为观测 [batch_size, 5] (t, I, T, U, delta_eta)，但实际只用 I 和 delta_eta。
        - SOC: 当前 SOC，形状 [batch_size, 1]（标量 per batch）。
        
        返回:
        - dSOC/dt，形状 [batch_size, 1]。
        """
        if self.Q is None or self.eta_0 is None:
            raise ValueError("SC parameters must be set before forward.")
        I: torch.Tensor = x[0][..., 1:2]  # [batch_size, 1]，电流 I (从 x[0] 的第 2 列提取)
        delta_eta: torch.Tensor = x[0][..., 4:5]  # [batch_size, 1]，效率修正 delta_eta (从 x[0] 的第 5 列提取)
        dSOC_dt: torch.Tensor = (self.eta_0 * (1 + delta_eta)) * I / (3600 * self.Q)  # [batch_size, 1]，物理公式计算 dSOC/dt
        return dSOC_dt  # 数据流：标量速率，用于 Euler 积分


class SOCNet(nn.Module):
    """
    SOCNet 模型：使用神经网络估计初始 SOC 和效率修正，然后通过手动 Euler 积分求解 SOC 轨迹。
    整合物理 ODE 模型与数据驱动组件。
    """
    def __init__(self, X: torch.Tensor, SC: torch.Tensor, *args, **kwargs) -> None:
        """
        初始化 SOCNet。
        
        参数:
        - X: 输入数据示例，用于形状推断，形状 [batch_size, seq_len, 4] (t, I, T, U)。
        - SC: SC 参数示例，形状 [batch_size, 4] (Q, eta_0, R, SOC_init_ref)。
        """
        super().__init__()
        self.func = SOCODEFunc()  # ODE 函数模块
        # 初始 SOC 网络：输入 4 维初始特征，输出 delta_SOC_init (Softplus 确保正值)
        self.SOC_init_net = nn.Sequential(nn.Linear(4, 1), nn.Softplus(), nn.Linear(1, 1))
        init_network_weights(self.SOC_init_net)  # 初始化权重
        # 效率修正网络：输入 I 和 T，输出 delta_eta (Softplus 确保正值)
        self.eta_net = nn.Sequential(nn.Linear(2, 1), nn.Softplus(), nn.Linear(1, 1))
        init_network_weights(self.eta_net)  # 初始化权重

    def set_sc_params(self, SC: torch.Tensor):
        """
        设置 SC 参数到 ODE 函数。
        
        参数:
        - SC: SC 参数，形状 [batch_size, 4]。
        """
        Q = SC[..., 0:1]  # [batch_size, 1]，容量 Q
        eta_0 = SC[..., 1:2]  # [batch_size, 1]，初始效率 eta_0
        self.func.set_sc_params(Q, eta_0)  # 更新 func 的参数

    def forward(self, X: torch.Tensor, SC: torch.Tensor) -> torch.Tensor:
        """
        前向传播：估计初始 SOC，计算 delta_eta，然后 Euler 积分求解 SOC 序列。
        
        参数:
        - X: 输入序列，形状 [batch_size, seq_len, 4] (t, I, T, U)。
        - SC: SC 参数，形状 [batch_size, 4] (Q, eta_0, R, SOC_init_ref)。
        
        返回:
        - SOC_estim: 估计 SOC 序列，形状 [batch_size, seq_len, 1]。
        """
        # 步骤 1: 提取输入特征
        batch_size, seq_len = X.shape[:2]  # batch_size, seq_len
        I = X[..., 1:2]  # [batch_size, seq_len, 1]，电流 I
        T = X[..., 2:3]  # [batch_size, seq_len, 1]，温度 T
        U = X[..., 3:4]  # [batch_size, seq_len, 1]，电压 U
        R = SC[..., 2:3]  # [batch_size, 1]，欧姆内阻 R (广播到序列)

        # 步骤 2: 计算初始 SOC
        init_feats = torch.cat([I[:, 0:1], T[:, 0:1], U[:, 0:1], R.unsqueeze(1)], dim=-1)  # [batch_size, 1, 4]，初始特征: I[0], T[0], U[0], R
        SOC_init_raw = self.SOC_init_net(init_feats).squeeze(-1)  # [batch_size, 1]，网络输出 delta_SOC_init (Softplus 确保正值)
        SOC_init = (SC[..., 3:4] * (1 + SOC_init_raw)).unsqueeze(1)  # [batch_size, 1, 1]，初始 SOC = SOC_init_ref * (1 + delta)，扩展时间维

        # 步骤 3: 计算效率修正 delta_eta
        delta_eta = self.eta_net(torch.cat([I, T], dim=-1))  # [batch_size, seq_len, 1]，输入 [I, T]，输出 delta_eta

        # 步骤 4: 设置 ODE 参数
        self.set_sc_params(SC)  # 设置 func 的 Q [batch_size, 1], eta_0 [batch_size, 1]

        # 步骤 5: 构建扩展观测 (添加 delta_eta)
        X_obs = torch.cat([X, delta_eta], dim=-1)  # [batch_size, seq_len, 5] (t, I, T, U, delta_eta)

        # 步骤 6: 手动 Euler 积分求解 SOC 轨迹
        times = X_obs[:, :, 0]  # [batch_size, seq_len]，时间序列 t
        SOC = SOC_init  # [batch_size, 1, 1]，初始 SOC (已扩展时间维)
        for k in range(1, seq_len):
            # 提取前一步观测
            x_step = X_obs[:, k-1:k, :].squeeze(1)  # [batch_size, 5]，前一步: t[k-1], I[k-1], T[k-1], U[k-1], delta_eta[k-1]
            x_list = [x_step]  # list of [batch_size, 5]，供 func 使用
            current_SOC = SOC[:, -1]  # [batch_size, 1]，当前 SOC (最后一步)
            dSOC_dt = self.func(x_list, current_SOC)  # [batch_size, 1]，在 t[k-1], SOC[k-1] 处的 dSOC/dt
            dt = times[:, k:k+1] - times[:, k-1:k]  # [batch_size, 1]，时间步长 Δt
            delta_SOC = dSOC_dt * dt  # [batch_size, 1]，SOC 增量
            new_SOC = current_SOC.unsqueeze(-1) + delta_SOC.unsqueeze(-1)  # [batch_size, 1, 1]，新 SOC = old + delta
            SOC = torch.cat([SOC, new_SOC], dim=1)  # [batch_size, k+1, 1]，累积 SOC 序列
        SOC_estim = SOC  # [batch_size, seq_len, 1]，最终 SOC 估计序列
        return SOC_estim  # 数据流：从初始估计 + 积分得到完整轨迹


def load_data(filename: Optional[str] = None, dataset: str = "ordered", model_type=type, device: torch.device = device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从 pickle 文件加载原始数据 (X, SC, Y)。
    
    参数:
    - filename: 数据文件路径 (默认: datasets/{dataset}/resample_SOC_50.pkl)。
    - dataset: 数据集名称 (e.g., "ordered")。
    - model_type: 模型类型 (未使用，但保留兼容性)。
    - device: 张量设备 (GPU/CPU)。
    
    返回:
    - X: 输入序列，形状 [n_samples, seq_len, 4] (t, I, T, U)。
    - SC: SC 参数，形状 [n_samples, >=4] (Q, eta_0, R, SOC_init_ref, ...)，如果 <4 则填充零。
    - Y: 真实 SOC，形状 [n_samples, seq_len, 1]。
    """
    if filename is None:
        filename = f"datasets/{dataset}/resample_SOC_50.pkl"  # 默认文件路径
    print(f"Loading data from {filename}...")
    with open(filename, "rb") as f:
        data_dict = pickle.load(f)  # 加载字典 {'X': np.array, 'SC': np.array, 'Y': np.array}
    X = torch.from_numpy(data_dict["X"]).to(device)  # [n_samples, seq_len, 4]
    SC = torch.from_numpy(data_dict["SC"]).to(device)  # [n_samples, 4+]
    Y = torch.from_numpy(data_dict["Y"]).to(device)  # [n_samples, seq_len, 1]
    # 如果 SC 维度不足 4，填充零 (e.g., 缺少 R 或 SOC_init_ref)
    if SC.shape[1] < 4:
        padding = torch.zeros(SC.shape[0], 4 - SC.shape[1], device=device)  # [n_samples, missing_dims]
        SC = torch.cat([SC, padding], dim=-1)  # [n_samples, 4]
    print(f"Data loaded: X{X.shape}, SC{SC.shape}, Y{Y.shape}")
    return X, SC, Y  # 数据流：原始均匀采样数据


class PCHIPNonUniformRepara(nn.Module):
    """
    Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) 用于非均匀时间点的插值。
    确保单调性和形状保持，适用于稀疏到密集的重采样。
    """
    def __init__(self, y_vals: torch.Tensor, t_vals: torch.Tensor) -> None:
        """
        初始化 PCHIP 插值器。
        
        参数:
        - y_vals: y 值 (e.g., 通道数据)，形状 [1, sparse_len, 1] -> squeeze to [1, sparse_len]。
        - t_vals: 时间点，形状 [1, sparse_len, 1] -> squeeze to [1, sparse_len]。
        """
        super().__init__()
        self.y_vals = y_vals.squeeze(-1)  # [1, sparse_len]，y 值序列
        self.t_vals = t_vals.squeeze(-1)  # [1, sparse_len]，时间序列
        sparse_len = self.y_vals.shape[1]  # 稀疏长度
        if sparse_len < 2:
            # 如果点数 <2，无法插值，设置空系数
            self.K = torch.empty(1, 0, 4, device=y_vals.device)  # [1, 0, 4]，空系数矩阵
            self.H = torch.empty(1, 0, device=y_vals.device)  # [1, 0]，空间隔
            return
        self.H = torch.diff(self.t_vals, dim=1)  # [1, sparse_len-1]，时间间隔 h_i = t_{i+1} - t_i
        self.DELTA = torch.diff(self.y_vals, dim=1) / self.H  # [1, sparse_len-1]，平均斜率 delta_i = (y_{i+1} - y_i)/h_i

        # 步骤: 计算 PCHIP 斜率 m (确保单调性)
        m = torch.zeros_like(self.y_vals)  # [1, sparse_len]，初始化斜率 m
        if sparse_len > 2:
            # 内部点：使用加权平均斜率，并检查单调性
            for i in range(1, sparse_len - 1):
                d0, d1 = self.DELTA[0, i-1], self.DELTA[0, i]  # 前后斜率
                h0, h1 = self.H[0, i-1], self.H[0, i]  # 前后间隔
                # 加权三点公式
                m[0, i] = ((2*h1 + h0)*d0 + (h1 + 2*h0)*d1) / (h0 + h1) / 3
                # 单调性检查：如果斜率符号不一致或违反单调，则设为 0
                if (d0 * d1 <= 0) or (torch.sign(m[0, i]) != torch.sign(d0)) or (torch.sign(m[0, i]) != torch.sign(d1)):
                    m[0, i] = 0.0
        # 边界：简单使用相邻斜率
        m[0, 0] = self.DELTA[0, 0]  # 左边界
        m[0, -1] = self.DELTA[0, -1]  # 右边界
        self.M = m  # [1, sparse_len]，最终斜率

        # 步骤: 计算分段三次多项式系数 K = [a, b, c, d] for phi(s) = a + b s + c s^2 + d s^3, s in [0,1]
        y0 = self.y_vals[:, :-1]  # [1, sparse_len-1]，段起始 y_i
        m0 = self.M[:, :-1]  # [1, sparse_len-1]，起始斜率 m_i
        c = (3 * self.DELTA - 2 * m0 - self.M[:, 1:]) / self.H  # [1, sparse_len-1]，二次系数
        d_coeff = (m0 + self.M[:, 1:] - 2 * self.DELTA) / (self.H ** 2)  # [1, sparse_len-1]，三次系数
        self.K = torch.stack([y0, m0 * self.H, c * self.H, d_coeff * (self.H ** 2)], dim=-1)  # [1, sparse_len-1, 4]，系数矩阵

    def forward(self, s: torch.Tensor, j: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        在第 j 段的局部坐标 s (0~1) 处计算插值 phi(s) 和导数 dphi/dt。
        
        参数:
        - s: 局部参数，形状 [] (标量)。
        - j: 段索引 (0 ~ sparse_len-2)。
        
        返回:
        - phi: 插值值，标量 []。
        - dphi_dt: 时间导数，标量 [] (dphi/ds / h_j)。
        """
        if self.K.shape[1] == 0:
            # 退化情况：返回第一个 y 值和零导数
            return self.y_vals[0, 0], torch.tensor(0.0, device=s.device, dtype=s.dtype)
        k = self.K[0, j]  # [4]，第 j 段系数 [a, b, c, d]
        s_ = s.float().item()  # 标量 float
        s2 = s_ ** 2
        s3 = s_ ** 3
        basis = torch.tensor([1.0, s_, s2, s3], device=s.device, dtype=s.dtype)  # [4]，基函数 [1, s, s^2, s^3]
        phi = torch.sum(k * basis)  # []，phi(s) = k @ basis
        basis_p = torch.tensor([0.0, 1.0, 2 * s_, 3 * s2], device=s.device, dtype=s.dtype)  # [4]，导数基 [0, 1, 2s, 3s^2]
        dphi_ds = torch.sum(k * basis_p)  # []，dphi/ds
        h_j = self.H[0, j].item()  # 标量，h_j
        dphi_dt = dphi_ds / h_j if h_j > 0 else torch.tensor(0.0, device=s.device, dtype=s.dtype)  # []，dphi/dt = (dphi/ds) / h_j
        return phi, dphi_dt  # 数据流：标量插值和导数，用于重采样


def resample_with_pchip(sparse_list_X: List[torch.Tensor], sparse_list_SC: List[torch.Tensor], sparse_list_Y: List[torch.Tensor],
                        target_len: int = 50, device: torch.device = device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    使用 PCHIP 对稀疏序列进行重采样到固定长度 target_len。
    对于每个样本：线性插值 t 到 [t_min, t_max]，然后逐通道插值 X 和 Y。
    
    参数:
    - sparse_list_X: 稀疏 X 列表，每个 [sparse_len_i, 4] (t, I, T, U)。
    - sparse_list_SC: 稀疏 SC 列表，每个 [4+] (Q, eta_0, R, SOC_init_ref)。
    - sparse_list_Y: 稀疏 Y 列表，每个 [sparse_len_i, 1] (SOC)。
    - target_len: 目标序列长度 (默认 50)。
    - device: 张量设备。
    
    返回:
    - full_X: 重采样 X，形状 [batch_size, target_len, 4]。
    - full_SC: SC，形状 [batch_size, 4+] (不变)。
    - full_Y: 重采样 Y，形状 [batch_size, target_len, 1]。
    """
    print(f"Starting PCHIP resampling for {len(sparse_list_X)} samples to target length {target_len}...")
    start_time = time.time()  # 计时开始
    batch_size = len(sparse_list_X)  # 样本数
    x_feats, sc_feats, y_feats = sparse_list_X[0].shape[-1], sparse_list_SC[0].shape[-1], sparse_list_Y[0].shape[-1]  # 特征维度
    # 初始化输出张量
    full_X = torch.zeros(batch_size, target_len, x_feats, device=device)  # [batch_size, target_len, 4]
    full_SC = torch.zeros(batch_size, sc_feats, device=device)  # [batch_size, sc_feats]
    full_Y = torch.zeros(batch_size, target_len, y_feats, device=device)  # [batch_size, target_len, 1]
    
    # 步骤: 逐样本重采样
    for i in range(batch_size):
        if (i + 1) % 100 == 0 or i == 0:  # 进度报告
            print(f"Resampling progress: {i+1}/{batch_size} samples processed...")
        sparse_X_i = sparse_list_X[i]  # [sparse_len_i, 4]
        sparse_Y_i = sparse_list_Y[i]  # [sparse_len_i, 1]
        sparse_len = sparse_X_i.shape[0]  # 当前稀疏长度
        if sparse_len < 2:
            # 退化：重复第一个点
            full_X[i] = sparse_X_i[0].repeat(target_len, 1)  # [target_len, 4]
            full_Y[i] = sparse_Y_i[0].repeat(target_len, 1)  # [target_len, 1]
            full_SC[i] = sparse_list_SC[i]  # [sc_feats]
            continue
        t_sparse = sparse_X_i[:, 0].contiguous()  # [sparse_len]，时间 t (确保连续以避免警告)
        t_min, t_max = t_sparse.min(), t_sparse.max()  # 范围 [t_min, t_max]
        t_query = torch.linspace(t_min, t_max, target_len, device=device)  # [target_len]，均匀查询时间点
        full_SC[i] = sparse_list_SC[i]  # [sc_feats]，SC 不变
        full_X[i, :, 0] = t_query  # [target_len]，设置 X 的 t 通道

        # 步骤: 插值 X 的每个通道 (ch=1: I, ch=2: T, ch=3: U)
        for ch in range(1, x_feats):
            sparse_ch = sparse_X_i[:, ch].unsqueeze(-1)  # [sparse_len, 1]，当前通道
            repara = PCHIPNonUniformRepara(sparse_ch.unsqueeze(0), t_sparse.unsqueeze(0).unsqueeze(-1))  # 初始化 PCHIP [1, sparse_len, 1], [1, sparse_len, 1]
            for k in range(target_len):
                t_q = t_query[k]  # 标量，查询时间
                if t_q <= t_min:
                    full_X[i, k, ch] = sparse_X_i[0, ch]  # 外推：使用 t_min 值
                    continue
                if t_q >= t_max:
                    full_X[i, k, ch] = sparse_X_i[-1, ch]  # 外推：使用 t_max 值
                    continue
                # 找到段 j：使用二分搜索
                j = torch.searchsorted(t_sparse, t_q, right=False).item() - 1
                if j < 0: j = 0
                if j >= sparse_len - 1: j = sparse_len - 2
                # 局部 s = (t_q - t_j) / (t_{j+1} - t_j)
                s_local = (t_q - t_sparse[j]) / (t_sparse[j+1] - t_sparse[j])
                s_local = torch.clamp(s_local, 0.0, 1.0)  # 夹紧 [0,1]
                phi, _ = repara(s_local.unsqueeze(0), j)  # []，插值 phi(s)
                full_X[i, k, ch] = phi.item()  # 填充标量值

        # 步骤: 插值 Y (SOC)，类似 X 但只一通道
        sparse_Y_flat = sparse_Y_i[:, 0].unsqueeze(-1)  # [sparse_len, 1]
        repara_y = PCHIPNonUniformRepara(sparse_Y_flat.unsqueeze(0), t_sparse.unsqueeze(0).unsqueeze(-1))  # [1, sparse_len, 1]
        for k in range(target_len):
            t_q = t_query[k]
            if t_q <= t_min or t_q >= t_max:
                # 外推：最近邻
                full_Y[i, k, 0] = sparse_Y_i[torch.argmin(torch.abs(t_sparse - t_q)), 0]
                continue
            j = torch.searchsorted(t_sparse, t_q, right=False).item() - 1
            if j < 0: j = 0
            if j >= sparse_len - 1: j = sparse_len - 2
            s_local = (t_q - t_sparse[j]) / (t_sparse[j+1] - t_sparse[j])
            s_local = torch.clamp(s_local, 0.0, 1.0)
            phi, _ = repara_y(s_local.unsqueeze(0), j)  # []，SOC 插值
            full_Y[i, k, 0] = phi.item()

    # 步骤: 后处理 - 夹紧值到物理范围
    full_Y = torch.clamp(full_Y, 0.0, 1.0)  # SOC [0,1]
    full_X[:, :, [0,2,3]] = torch.clamp(full_X[:, :, [0,2,3]], min=0.0)  # t, T, U >=0 (I 可负)

    elapsed_time = time.time() - start_time  # 计时结束
    print(f"Resampling completed in {elapsed_time:.2f}s. Final shapes: X{full_X.shape}, SC{full_SC.shape}, Y{full_Y.shape}")
    return full_X, full_SC, full_Y  # 数据流：从稀疏列表到密集均匀张量


def drop_random_in_range_points(X: torch.Tensor, SC: torch.Tensor, Y: torch.Tensor, drop_range: Tuple[float, float] = (0.10, 0.14)) -> tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    在指定比例范围内随机丢弃点，生成稀疏序列 (模拟不均匀采样)。
    
    参数:
    - X: 输入，形状 [batch_size, seq_len, 4]。
    - SC: SC，形状 [batch_size, 4+]。
    - Y: 输出，形状 [batch_size, seq_len, 1]。
    - drop_range: 丢弃比例范围 (low, high)，e.g., (0.10, 0.14) 表示 10%-14%。
    
    返回:
    - list_X: 稀疏 X 列表，每个 [kept_len_i, 4]。
    - list_SC: SC 列表，每个 [4+] (不变)。
    - list_Y: 稀疏 Y 列表，每个 [kept_len_i, 1]。
    """
    print(f"Starting random point dropping in range {drop_range} for {X.shape[0]} samples (original seq_len: {X.shape[1]})...")
    start_time = time.time()  # 计时
    batch_size, seq_len = X.shape[:2]  # batch_size, seq_len
    device = X.device
    list_X, list_SC, list_Y = [], [], []  # 初始化列表
    total_dropped = 0  # 总丢弃计数
    
    # 步骤: 逐样本随机丢弃
    for i in range(batch_size):
        # 随机选择丢弃比例
        drop_ratio = torch.rand(1, device=device).item() * (drop_range[1] - drop_range[0]) + drop_range[0]
        num_drop = max(1, int(drop_ratio * seq_len))  # 至少丢 1 个
        drop_indices = torch.randperm(seq_len, device=device)[:num_drop]  # 随机索引
        keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)  # [seq_len]，保持掩码
        keep_mask[drop_indices] = False  # 标记丢弃
        kept_len = keep_mask.sum().item()  # 保持长度
        
        # 提取保持部分
        list_X.append(X[i][keep_mask])  # [kept_len, 4]
        list_SC.append(SC[i])  # [4+]
        list_Y.append(Y[i][keep_mask])  # [kept_len, 1]
        total_dropped += num_drop  # 更新计数
        
        if (i + 1) % 100 == 0 or i == 0:  # 进度报告
            print(f"Dropping progress: {i+1}/{batch_size} samples processed (avg dropped so far: {total_dropped / (i+1):.2f}/sample)")
    
    # 统计
    avg_drop_ratio = total_dropped / (batch_size * seq_len)
    elapsed_time = time.time() - start_time
    print(f"Dropping completed in {elapsed_time:.2f}s. Avg drop ratio: {avg_drop_ratio:.4f}, resulting sparse lengths: {[len(x) for x in list_X[:5]]}... (first 5)")
    return list_X, list_SC, list_Y  # 数据流：从密集张量到稀疏列表


def load_preprocessed_data(dataset: str, model_type: type, device: torch.device) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    尝试加载预处理数据 (丢弃 + 重采样后)。
    
    参数:
    - dataset: 数据集名称。
    - model_type: 模型类型 (用于文件名)。
    - device: 设备。
    
    返回:
    - (X, SC, Y) 如果存在，否则 None。形状 [batch_size, target_len, 4/1], [batch_size, 4+]。
    """
    preprocessed_file = Path(f"datasets/preprocessed/{dataset}/{model_type.__name__}_preprocessed.pkl")  # 文件路径
    if preprocessed_file.exists():
        print(f"Loading preprocessed data from {preprocessed_file}...")
        with open(preprocessed_file, "rb") as f:
            data_dict = pickle.load(f)  # {'X': np, 'SC': np, 'Y': np}
        X, SC, Y = (torch.from_numpy(data_dict["X"]).to(device),
                    torch.from_numpy(data_dict["SC"]).to(device),
                    torch.from_numpy(data_dict["Y"]).to(device))
        print(f"Preprocessed data loaded: X{X.shape}, SC{SC.shape}, Y{Y.shape}")
        return X, SC, Y
    print(f"No preprocessed file found at {preprocessed_file}. Will preprocess from raw data.")
    return None  # 未找到，返回 None


def save_preprocessed_data(X: torch.Tensor, SC: torch.Tensor, Y: torch.Tensor, dataset: str, model_type: type):
    """
    保存预处理数据到 pickle。
    
    参数:
    - X, SC, Y: 预处理张量。
    - dataset: 数据集名称。
    - model_type: 模型类型 (用于文件名)。
    """
    print(f"Saving preprocessed data: X{X.shape}, SC{SC.shape}, Y{Y.shape}...")
    start_time = time.time()
    preprocessed_dir = Path(f"datasets/preprocessed/{dataset}")  # 目录
    preprocessed_dir.mkdir(parents=True, exist_ok=True)  # 创建目录
    preprocessed_file = preprocessed_dir / f"{model_type.__name__}_preprocessed.pkl"  # 文件
    data_dict = {"X": X.cpu().numpy(), "SC": SC.cpu().numpy(), "Y": Y.cpu().numpy()}  # CPU numpy
    with open(preprocessed_file, "wb") as f:
        pickle.dump(data_dict, f)  # 保存
    elapsed_time = time.time() - start_time
    print(f"Preprocessed data saved to {preprocessed_file} in {elapsed_time:.2f}s.")


class SaveAndEarlyStop:
    """
    早停和模型保存器：基于验证损失监控，保存最佳模型。
    """
    def __init__(self, dataset: str, patience: int = 30, delta: float = 0.0, save: Optional[str] = None):
        """
        初始化早停器。
        
        参数:
        - dataset: 数据集名称 (用于日志)。
        - patience: 耐心值 (连续无改善 epoch 数)。
        - delta: 最小改善阈值。
        - save: 保存标志 (如果 None，则不保存)。
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0  # 无改善计数
        self.best_score = None  # 最佳分数
        self.early_stop = False  # 早停标志
        self.save = save
        self.dataset = dataset

    def stop(self, val_loss: float, model: nn.Module, optimizer: Adam) -> bool:
        """
        检查是否早停，并保存模型如果改善。
        
        参数:
        - val_loss: 当前验证损失。
        - model: 模型。
        - optimizer: 优化器。
        
        返回:
        - bool: 是否早停。
        """
        if self.patience is None:
            # 无早停：总是保存
            if self.save:
                self._save_model(model, optimizer)
            return False
        score = -val_loss  # 更高分数更好 (负损失)
        if self.best_score is None:
            self.best_score = score
            if self.save:
                self._save_model(model, optimizer)  # 首次保存
        elif score < self.best_score + self.delta:
            # 无改善
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # 触发早停
        else:
            # 改善
            self.best_score = score
            self.counter = 0
            if self.save:
                self._save_model(model, optimizer)  # 保存最佳
        return self.early_stop

    def _save_model(self, model: nn.Module, optimizer: Adam):
        """
        保存模型和优化器状态到日期目录。
        """
        dirpath = Path(f"save/{date.today()}/")  # 今日目录
        dirpath.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), dirpath / f"{type(model).__name__}.pt")  # 模型权重
        torch.save(optimizer.state_dict(), dirpath / f"{type(model).__name__}-Optimizer.pt")  # 优化器状态
        print(f"Model and optimizer saved to {dirpath}.")


def main(model_type: type = SOCNet, dataset: str = "ordered", save: Optional[str] = None, max_epochs: int = 800):
    """
    主训练函数：加载/预处理数据，训练模型，评估并日志。
    
    参数:
    - model_type: 模型类 (默认 SOCNet)。
    - dataset: 数据集 (默认 "ordered")。
    - save: 保存/日志标志 (启用 TensorBoard 和模型保存)。
    - max_epochs: 最大 epoch 数。
    """
    print(f"Starting training: {model_type.__name__} on {dataset}")
    print(f"Using device: {device}")
    
    # 步骤 1: 加载或预处理数据
    preprocessed = load_preprocessed_data(dataset, model_type, device)
    if preprocessed is None:
        print("Preprocessing from raw data...")
        X_raw, SC_raw, Y_raw = load_data(dataset=dataset, model_type=model_type)  # 原始数据
        print(f"Raw data shapes: X{X_raw.shape}, SC{SC_raw.shape}, Y{Y_raw.shape}")
        X_list, SC_list, Y_list = drop_random_in_range_points(X_raw, SC_raw, Y_raw)  # 随机丢弃 -> 稀疏列表
        X, SC, Y = resample_with_pchip(X_list, SC_list, Y_list, target_len=X_raw.shape[1])  # PCHIP 重采样 -> 密集
        save_preprocessed_data(X, SC, Y, dataset, model_type)  # 保存预处理
        print(f"Preprocessed shapes: X{X.shape}, SC{SC.shape}, Y{Y.shape}")
    else:
        X, SC, Y = preprocessed  # 直接加载

    # 步骤 2: 数据划分 (80/20)
    choice = np.random.permutation(X.shape[0]).tolist()  # 随机排列索引
    train_choice, test_choice = choice[:int(0.8 * len(choice))], choice[int(0.8 * len(choice)):]  # 训练/测试索引
    n_train, n_test = len(train_choice), len(test_choice)
    print(f"Dataset split: Train {n_train} samples, Test {n_test} samples")
    
    # 步骤 3: 数据加载器
    batch_size = 1024  # 批次大小
    train_dataset = TensorDataset(X[train_choice], SC[train_choice], Y[train_choice])  # 训练数据集
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 打乱加载
    print(f"Train dataloader: {len(train_dataloader)} batches of size {batch_size}")

    # 步骤 4: 初始化模型、损失、优化器
    model: nn.Module = model_type(X, SC).to(device)  # 创建模型 (使用示例形状)
    print(f"Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    loss_fn = LogCoshLoss()  # 损失函数
    optimizer = Adam(model.parameters(), lr=0.001)  # Adam 优化器
    writer = SummaryWriter(log_dir=f"runs/{dataset}_{save}") if save else Mock()  # TensorBoard (Mock 如果无 save)
    saver = SaveAndEarlyStop(dataset=dataset, save=save)  # 早停器

    # 步骤 5: 训练循环
    print("Starting training loop...")
    for epoch in range(max_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        num_batches = len(train_dataloader)
        for batch_idx, (batch_x, batch_sc, batch_y) in enumerate(train_dataloader):
            # 前向 + 反向
            optimizer.zero_grad()  # 清零梯度
            y_pred = model(batch_x, batch_sc)  # [batch_size, seq_len, 1]
            loss = loss_fn(y_pred, batch_y)  # 标量损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            train_loss += loss.item()  # 累积损失
            
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:  # 批次进度
                print(f"Epoch {epoch+1} batch progress: {batch_idx+1}/{num_batches} (current batch loss: {loss.item():.6f})")
        train_loss /= len(train_dataloader)  # 平均训练损失

        # 评估阶段 (全测试集)
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X[test_choice], SC[test_choice])  # [n_test, seq_len, 1]
            test_loss = loss_fn(y_pred_test, Y[test_choice]).item()  # 测试损失
            mae = mean_absolute_error(y_pred_test.flatten(), Y[test_choice].flatten()).item()  # MAE (展平 [n_test*seq_len])
            mape = mean_absolute_percentage_error(y_pred_test.flatten(), Y[test_choice].flatten()).item()  # MAPE
            rmse = root_mean_squared_error(y_pred_test.flatten(), Y[test_choice].flatten()).item()  # RMSE

        # 日志
        print(f"Epoch {epoch+1}: Train {train_loss:.6f}, Test {test_loss:.6f}, MAE {mae:.6f}, MAPE {mape:.4f}%, RMSE {rmse:.6f}")
        writer.add_scalar("Loss/Train", train_loss, epoch+1)
        writer.add_scalar("Loss/Test", test_loss, epoch+1)
        writer.add_scalar("MAE", mae, epoch+1)
        writer.add_scalar("MAPE", mape, epoch+1)
        writer.add_scalar("RMSE", rmse, epoch+1)
        
        # 早停检查
        if saver.stop(test_loss, model, optimizer):
            print(f"Early stop at epoch {epoch+1}")
            break
    print("Training completed.")


if __name__ == "__main__":
    main(save="socnet_run")  # 启用save以保存模型