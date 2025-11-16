import torch
from torch import nn, Tensor
# from torchode import odeint


def OCV_SOC_curve_mich(SOC: torch.Tensor, OCV_SOC_params: torch.Tensor) -> torch.Tensor:
    # SOC.shape: [batch_dims, seq_len, 1]
    # OCV_SOC_params.shape: [batch_dims, seq_len, 2]
    x = OCV_SOC_params[..., 0:1]  # [batch_dims, seq_len, 1]
    y = OCV_SOC_params[..., 1:2]  # [batch_dims, seq_len, 1]
    Up = (
        4.4167  # 5.9452 - 0.5623 * e
        - 1.6518 * y
        + 1.6225 * y**2
        - 2.0843 * y**3
        + 3.5146 * y**4
        - 2.2166 * y**5
        - 4 * torch.exp(109.451 * y - 100.006)
    )
    Un = (
        0.063
        + 0.8 * torch.exp(-75 * (x + 0.001))
        - 0.0120 * torch.tanh((x - 0.127) / 0.016)
        - 0.0118 * torch.tanh((x - 0.155) / 0.016)
        - 0.0035 * torch.tanh((x - 0.220) / 0.020)
        - 0.0095 * torch.tanh((x - 0.190) / 0.013)
        - 0.0145 * torch.tanh((x - 0.490) / 0.020)
        - 0.0800 * torch.tanh((x - 1.030) / 0.055)
    )
    OCV = Up - Un
    return OCV


params = {
    "a123": {
        "R_1": (0.005, 0.02),
        "1/R_1C_1": (0.025, 0.07),
        "OCV - U": (0.1, 1.0),
        "-M_H": (0.0, 0.055),
        "K_H": (0.002, 0.025),
        "OCVSOCParams": [...],
        "OCVSOCcurve": ...,
        "Q_n": 1.1,
    },
    "mich": {
        "R_1": (0.0, 0.04),
        "1/R_1C_1": (0.0, 1e-6),
        "OCV - U": (0.05, 0.8),
        "-M_H": (0.0, 0.0),
        "K_H": (0.0, 0.0),
        "x": (0.04236, 0.84),
        "y": (0.023, 0.84804),
        "OCVSOCParams": ["x", "y"],
        "OCVSOCcurve": OCV_SOC_curve_mich,
        "Q_n": 5.0,
    },
}


#原本prednet的odeint
def odeint(
    func: nn.Module, y0: Tensor, X: Tensor, t_mask: Tensor | None = None, **kwargs
) -> Tensor:
    # y0.shape: [batch_dims, *y_channels]
    # X.shape: [batch_dims, seq_len, *t_X_channels]
    # t_mask.shape: [*t_X_channels]
    if t_mask is not None:
        ts = X[:, :, t_mask]  # [batch_dims, seq_len, *t_channels]
    else:
        ts = X  # [batch_dims, seq_len, *t_channels]
    y = y0  # [batch_dims, *y_channels]
    ys = [y]
    for i in range(ts.shape[1] - 1):
        y: torch.Tensor = y + (ts[:, i + 1, ...] - ts[:, i, ...]) * func.forward(
            [X[:, i, ...]], y
        )  # [batch_dims, *y_channels]
        # assert y.isfinite().all()
        ys.append(y)

    return torch.stack(ys, dim=0)  # [seq_len, batch_dims, *y_channels]







def init_network_weights(net: nn.Module, init_method="normal", std=0.1, bias=0.0):
    # 高斯分布初始化/xavier初始化
    if not hasattr(net, "modules"):
        return
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            if init_method == "normal":
                torch.nn.init.normal_(m.weight, mean=0, std=std)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, val=0)
            elif init_method == "xavier":
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(bias)
            else:
                raise NotImplementedError


def create_net(
    input_dims: int,
    output_dims=None,
    units=100,
    hidden_layers=0,
    activation=nn.Tanh,
    device=torch.device("cpu"),
    dropout_rate=None,
    bias=True,
    init_method="normal",
):
    # 定义模型
    if output_dims is None:
        output_dims = input_dims
    layers = [nn.Linear(input_dims, units, bias=bias)]

    for _ in range(hidden_layers):
        layers.append(activation())
        if dropout_rate is not None:
            layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.Linear(units, units, bias=bias))

    layers.append(activation())
    if dropout_rate is not None:
        layers.append(nn.Dropout(p=dropout_rate))
    layers.append(nn.Linear(units, output_dims, bias=bias))

    # 用在函数传递当中，*尝试将对象转为元组（可变参数），**尝试将对象转为字典（关键字参数），两者均可传入0个或任意个参数
    net = nn.Sequential(*layers).to(device)
    init_network_weights(net, init_method=init_method)
    return net


def create_net_from_list(
    input_dims: int,
    output_dims=None,
    hidden_list=None,
    device=torch.device("cpu"),
    dropout_rate=None,
    bias=True,
    init_method="normal",
):
    """
    Hidden list is a nested list or None, each inner list defines one hidden layer
    that has 2 elements:
    1. the hidden dim;
    2. the activation function that should be applied.
    """
    if output_dims is None:
        output_dims = input_dims
    if hidden_list is None:
        layers = [nn.Linear(input_dims, output_dims, bias=bias)]
    else:
        layers = [nn.Linear(input_dims, hidden_list[0][0], bias=bias)]
        if len(hidden_list) > 1:
            for i in range(len(hidden_list) - 1):
                layers.append(hidden_list[i][1]())
                if dropout_rate is not None:
                    layers.append(nn.Dropout(p=dropout_rate))
                layers.append(
                    nn.Linear(hidden_list[i][0], hidden_list[i + 1][0], bias=bias)
                )
        layers.append(hidden_list[-1][1]())
        if dropout_rate is not None:
            layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.Linear(hidden_list[-1][0], output_dims, bias=bias))

    # 用在函数传递当中，*尝试将对象转为元组（可变参数），**尝试将对象转为字典（关键字参数），两者均可传入0个或任意个参数
    net = nn.Sequential(*layers).to(device)
    init_network_weights(net, init_method=init_method)
    return net

class Copy_SOCODEFunc(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def set_sc_params(self, Q: Tensor, eta_0: Tensor):
        # Q.shape: [batch_dims, 1]
        # eta_0.shape: [batch_dims, 1]
        self.Q = Q
        self.eta_0 = eta_0

    def forward(self, x: list[Tensor], SOC: Tensor) -> Tensor:
        # x[0]: (t_I_T.shape: [batch_dims, 4])
        # x[1]: (dt_I_T_ds.shape: [batch_dims, 4])
        # SOC.shape: [batch_dims, 1]
        # t: Tensor = x[0][..., 0:1]  # [batch_dims, 1]
        I: Tensor = x[0][..., 1:2]  # [batch_dims, 1]
        # T: Tensor = x[0][..., 2:3]  # [batch_dims, 1]
        # U: Tensor = x[0][..., 3:4]  # [batch_dims, 1]
        delta_eta: Tensor = x[0][..., 4:5]  # [batch_dims, 1]

        dSOC_dt: Tensor = (
            (self.eta_0 * (1 + delta_eta)) * I / (3600 * self.Q)
        )  # [batch_dims, 1]
        # 注意：I < 0，这里使用放电的eta_0

        return dSOC_dt


class Copy_SOCNet(nn.Module):
    def __init__(self, X: Tensor, SC: Tensor, *args, **kwargs) -> None:
        #*args会将所有未被明确接收的额外位置参数打包成一个元组（tuple）
        #**kwargs会将所有未被明确接收的额外关键字参数打包成一个字典（dict）
        super().__init__()
        self.func = Copy_SOCODEFunc()
        self.SOC_init_net = nn.Sequential(
            nn.Linear(4, 1), nn.Softplus(), nn.Linear(1, 1)
        )
        init_network_weights(self.SOC_init_net)
        self.eta_net = nn.Sequential(nn.Linear(2, 1), nn.Softplus(), nn.Linear(1, 1))
        init_network_weights(self.eta_net)

    def set_sc_params(self, SC: Tensor):
        # SC.shape: [batch_dims, 4]
        Q = SC[..., 0:1]
        eta_0 = SC[..., 1:2]
        self.func.set_sc_params(Q, eta_0)

    def forward(self, X: Tensor, SC: Tensor) -> Tensor:
        # X.shape: [batch_dims, seq_len, 4]
        # SC.shape: [batch_dims, 4]
        # t = X[..., 0:1]  # [batch_dims, seq_len, 1]
        I = X[..., 1:2]  # [batch_dims, seq_len, 1]
        T = X[..., 2:3]  # [batch_dims, seq_len, 1]
        U = X[..., 3:4]  # [batch_dims, seq_len, 1]
        R = SC[..., 2:3]  # [batch_dims, 1]

        SOC_init = SC[..., 3:4] * (
            1
            + self.SOC_init_net(
                torch.cat([I[:, 0, :], T[:, 0, :], U[:, 0, :], R], dim=-1)
            )
        )  # [batch_dims, 1])
        delta_eta = self.eta_net(torch.cat([I, T], dim=-1))  # [batch_dims, seq_len, 1]
        self.set_sc_params(SC)
        X = torch.cat([X, delta_eta], dim=-1)
        t_mask = torch.zeros(X.shape[-1], device=X.device, dtype=torch.bool)
        t_mask[0] = True
        SOC_estim: Tensor = odeint(
            self.func,
            SOC_init,
            X,
            t_mask=t_mask,
            method="euler",
        )  # [seq_len, batch_dims, 1]
        SOC_estim = SOC_estim.transpose(0, 1)  # [batch_dims, seq_len, 1]
        return SOC_estim

class W_SOCODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.Q = None
        self.eta_0 = None

    def set_sc_params(self, Q, eta_0):
        self.Q = Q  # [N,1]
        self.eta_0 = eta_0  # [N,1]

    # 注意：在 odeint(self.func, SOC_init, X_obs, ...) 的调用中，
    # 这里的 t 实际上是插值后的 X_obs（4个特征），y 是当前的 SOC 估计 [N, 1]
    def forward(self, t, y): 
        x = t 
        N = y.shape[0] # 获取正确的 Batch Size N=1024
        
        # ------------------ 最终的容错逻辑 --------------------
        if isinstance(x, (list, tuple)):
            # 尝试从列表中提取 Tensor
            x_tensor = None
            for item in x:
                if isinstance(item, torch.Tensor):
                    x_tensor = item
                    break
            x = x_tensor if x_tensor is not None else torch.zeros_like(y)
        
        if not isinstance(x, torch.Tensor) or x.dim() < 2 or x.shape[-1] < 4:
            # 如果 x 不正确（例如 0D 或 1D），强制用零填充为 [N, 4]
            if x.dim() == 0:
                print(f"Warning: X is 0D, zero-filling to [{N}, 4].")
            else:
                print(f"Warning: X shape {x.shape} invalid, zero-filling to [{N}, 4].")
                
            x = torch.zeros(N, 4, device=y.device, dtype=y.dtype)
        
        # x 现在应该是 [1024, 4]

        I = x[..., 0:1]  # I (通道0), [N, 1]
        delta_eta = x[..., 3:4]  # delta_eta (通道3), [N, 1]
        
        # 确保 SC 参数已设置
        if self.Q is None or self.eta_0 is None:
            # 应该不会发生，但在 ODEFunc 中添加断言总是好的
            raise ValueError("SC parameters Q and eta_0 must be set.")

        # dy/dt 应该返回 [N, 1]
        dy_dt = (self.eta_0 * (1 + delta_eta)) * I / (3600 * self.Q)
        
        return dy_dt # [N, 1]

class W_SOCNet(nn.Module):
    def __init__(self, X, SC, *args, **kwargs):
        super().__init__()
        self.func = W_SOCODEFunc()
        self.SOC_init_net = nn.Sequential(nn.Linear(4, 1), nn.Softplus(), nn.Linear(1, 1))
        init_network_weights(self.SOC_init_net)
        self.eta_net = nn.Sequential(nn.Linear(2, 1), nn.Softplus(), nn.Linear(1, 1))
        init_network_weights(self.eta_net)

    def set_sc_params(self, SC):
        self.func.set_sc_params(SC[..., 0:1], SC[..., 1:2])  # Q/eta_0

    def forward(self, X, SC):
        # X: [N,L,4] (t/I/T/U), SC: [N,4]
        I = X[..., 1:2]  # [N,L,1] 电流
        T = X[..., 2:3]  # [N,L,1] 温度
        U = X[..., 3:4]  # [N,L,1] 电压
        R = SC[..., 2:3]  # [N,1] 内阻
        
        # 初始特征：使用序列第0点，squeeze 到 [N,1]
        init_feats = torch.cat([I[:, 0], T[:, 0], U[:, 0], R], dim=-1)  # [N,4]
        SOC_init = SC[..., 3:4] * (1 + self.SOC_init_net(init_feats))  # [N,1] 初始 SOC
        
        delta_eta = self.eta_net(torch.cat([I, T], dim=-1))  # [N,L,2] -> [N,L,1] delta_eta
        
        # 关键修复：排除 t，仅观测通道 I/T/U/delta_eta [N,L,4]
        X_obs = torch.cat([I, T, U, delta_eta], dim=-1)  # [N,L,4]
        
        # t_mask 全 True（所有通道均为 obs，无需 mask t）
        t_mask = torch.ones(X_obs.shape[-1], dtype=torch.bool, device=X.device)  # [4] 全 True
        
        self.set_sc_params(SC)
        # odeint 积分（时间网格隐式从位置推导）
        SOC_estim = odeint(self.func, SOC_init, X_obs, t_mask=t_mask, method="euler").transpose(0, 1)
        return SOC_estim  # [N,L,1] 预测 SOC


class SOCODEFunc(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def set_sc_params(self, Q: Tensor, eta_0: Tensor):
        # Q.shape: [batch_dims, 1]
        # eta_0.shape: [batch_dims, 1]
        self.Q = Q
        self.eta_0 = eta_0

    def forward(self, x: list[Tensor], SOC: Tensor) -> Tensor:
        # x[0]: (t_I_T.shape: [batch_dims, 4])
        # x[1]: (dt_I_T_ds.shape: [batch_dims, 4])
        # SOC.shape: [batch_dims, 1]
        # t: Tensor = x[0][..., 0:1]  # [batch_dims, 1]
        I: Tensor = x[0][..., 1:2]  # [batch_dims, 1]
        # T: Tensor = x[0][..., 2:3]  # [batch_dims, 1]
        # U: Tensor = x[0][..., 3:4]  # [batch_dims, 1]
        delta_eta: Tensor = x[0][..., 4:5]  # [batch_dims, 1]

        dSOC_dt: Tensor = (
            (self.eta_0 * (1 + delta_eta)) * I / (3600 * self.Q)
        )  # [batch_dims, 1]
        # 注意：I < 0，这里使用放电的eta_0

        return dSOC_dt


class SOCNet(nn.Module):
    def __init__(self, X: Tensor, SC: Tensor, *args, **kwargs) -> None:
        #*args会将所有未被明确接收的额外位置参数打包成一个元组（tuple）
        #**kwargs会将所有未被明确接收的额外关键字参数打包成一个字典（dict）
        super().__init__()
        self.func = SOCODEFunc()
        self.SOC_init_net = nn.Sequential(
            nn.Linear(4, 1), nn.Softplus(), nn.Linear(1, 1)
        )
        init_network_weights(self.SOC_init_net)
        self.eta_net = nn.Sequential(nn.Linear(2, 1), nn.Softplus(), nn.Linear(1, 1))
        init_network_weights(self.eta_net)

    def set_sc_params(self, SC: Tensor):
        # SC.shape: [batch_dims, 4]
        Q = SC[..., 0:1]
        eta_0 = SC[..., 1:2]
        self.func.set_sc_params(Q, eta_0)

    def forward(self, X: Tensor, SC: Tensor) -> Tensor:
        # X.shape: [batch_dims, seq_len, 4]
        # SC.shape: [batch_dims, 4]
        # t = X[..., 0:1]  # [batch_dims, seq_len, 1]
        I = X[..., 1:2]  # [batch_dims, seq_len, 1]
        T = X[..., 2:3]  # [batch_dims, seq_len, 1]
        U = X[..., 3:4]  # [batch_dims, seq_len, 1]
        R = SC[..., 2:3]  # [batch_dims, 1]

        SOC_init = SC[..., 3:4] * (
            1
            + self.SOC_init_net(
                torch.cat([I[:, 0, :], T[:, 0, :], U[:, 0, :], R], dim=-1)
            )
        )  # [batch_dims, 1])
        delta_eta = self.eta_net(torch.cat([I, T], dim=-1))  # [batch_dims, seq_len, 1]
        self.set_sc_params(SC)
        X = torch.cat([X, delta_eta], dim=-1)
        t_mask = torch.zeros(X.shape[-1], device=X.device, dtype=torch.bool)
        t_mask[0] = True
        SOC_estim: Tensor = odeint(
            self.func,
            SOC_init,
            X,
            t_mask=t_mask,
            method="euler",
        )  # [seq_len, batch_dims, 1]
        SOC_estim = SOC_estim.transpose(0, 1)  # [batch_dims, seq_len, 1]
        return SOC_estim


class IRODEFunc(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: list[Tensor], voltages: Tensor) -> Tensor:
        # x[0]: ([t, I, T, SOC, R_1, R_1C_1_r, _M_H, K_H].shape: [batch_dims, 8])
        # x[1]: (d[t, I, T, SOC, R_1, R_1C_1_r, _M_H, K_H]_ds.shape: [batch_dims, 8])
        # voltages.shape: [batch_dims, 2]
        # t: Tensor = x[0][..., 0:1]  # [batch_dims, 1]
        I: Tensor = x[0][..., 1:2]  # [batch_dims, 1]
        # T: Tensor = x[0][..., 2:3]  # [batch_dims, 1]
        # SOC: Tensor = x[0][..., 3:4]  # [batch_dims, 1]
        R_1: Tensor = x[0][..., 4:5]  # [batch_dims, 1]
        R_1C_1_r: Tensor = x[0][..., 5:6]  # [batch_dims, 1]
        _M_H: Tensor = x[0][..., 6:7]  # [batch_dims, 1]
        K_H: Tensor = x[0][..., 7:8]  # [batch_dims, 1]

        U_H: Tensor = voltages[..., 0:1]  # [batch_dims, 1]
        U_1: Tensor = voltages[..., 1:2]  # [batch_dims, 1]

        dU_H_dt = -K_H * I * (-_M_H - U_H)  # [batch_dims, 1]
        # 注意：I < 0，这里 K_H * I 绝对值直接取负值
        dU_1_dt = R_1C_1_r * (R_1 * I - U_1)  # [batch_dims, 1]
        dvoltages_dt = torch.cat([dU_H_dt, dU_1_dt], dim=-1)  # [batch_dims, 2]

        return dvoltages_dt


class IRParamsNet(nn.Module):
    def __init__(
        self,
        X: Tensor,
        SC: Tensor,
        dataset: str,
        U_H_init=False,
        ub: list | None = None,
        lb: list | None = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        x_channels = 2
        c_channels = SC.shape[-1]
        self.p_net = nn.Sequential(
            nn.Linear(x_channels + c_channels, 2 * (x_channels + c_channels)),
            nn.Softplus(),
            nn.Linear(2 * (x_channels + c_channels), 5), #五个输出参数是R_1, 1/(R_1 C_1), OCV - U, -M_H, K_H
        )
        init_network_weights(self.p_net)
        self.R_s_init_net = nn.Sequential(
            nn.Linear(x_channels + c_channels, 1),
            nn.Softplus(),
            nn.Linear(1, 1),
        )
        init_network_weights(self.R_s_init_net)

        self.ub = torch.tensor(
            (
                [
                    params[dataset][p][1]#索引1是上限值
                    for p in ["R_1", "1/R_1C_1", "OCV - U", "-M_H", "K_H"]
                ]
                if ub is None
                else ub
            ),
            dtype=X.dtype,#新张量的数据类型
            device=X.device,#所在的设备（如 CPU 或 GPU）
        )
        self.lb = torch.tensor(
            (
                [
                    params[dataset][p][0]#索引0是下限值
                    for p in ["R_1", "1/R_1C_1", "OCV - U", "-M_H", "K_H"]
                ]
                if lb is None
                else lb
            ),
            dtype=X.dtype,
            device=X.device,
        )
        self.range = self.ub - self.lb  # [5]

        self.dvoltages_dt_func = IRODEFunc()

        if U_H_init:
            self.U_H_init_net = nn.Sequential(
                nn.Linear(3, 1), nn.Softplus(), nn.Linear(1, 1)
            )
            init_network_weights(self.U_H_init_net)
        else:
            self.U_H_init_net = None

    def forward(
        self, X: Tensor, SC: Tensor, return_params=False
        ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        # X: 输入张量，形状为 [batch_dims, seq_len, 5]，包含时间序列数据
        #   - batch_dims: 批次中的电池样本数量
        #   - seq_len: 时间序列的长度（时间步数）
        #   - 5 个特征: [t, I, T, U, SOC] (时间, 电流, 温度, 电压, 荷电状态)
        # SC: 静态协变量张量，形状为 [batch_dims, 3]，包含固定特征
        #   - 3 个特征: [Q, eta_0, R_s_base] (容量, 初始效率, 基础电阻)
        # return_params: 布尔标志，若为 True 则返回额外参数 (theta, voltages) 用于调试
        # 返回值: 若 return_params=False，则返回张量 (IR_estim)；若 True，则返回元组 (IR_estim, theta, voltages)

        # 步骤 1: 从输入张量 X 和 SC 中提取特征
        # 从 X 中提取时间 t (第一个特征)，形状 [batch_dims, seq_len, 1]
        t = X[..., 0:1]  # 参考: 论文 Fig. 2 "Inputs" (时间序列数据)
        # 从 X 中提取电流 I (第二个特征)，形状 [batch_dims, seq_len, 1]
        I = X[..., 1:2]  # 参考: 论文 Fig. 5 (等效电路中的电流 I)
        # 从 X 中提取温度 T (第三个特征)，形状 [batch_dims, seq_len, 1]
        T = X[..., 2:3]  # 参考: 论文 III-B (温度作为 IRNet 的输入)
        # 从 X 中提取荷电状态 SOC (第五个特征)，形状 [batch_dims, seq_len, 1]
        SOC = X[..., 4:]  # 参考: 论文 Fig. 2 (SOC 作为健康指标)

        # 步骤 2: 通过拼接特征为参数网络准备输入
        # 沿着最后一个维度拼接 SOC, T 和扩展后的 SC
        # SC.unsqueeze(-2) 添加时间维度，expand 匹配 seq_len，结果为 [batch_dims, seq_len, 5]
        x: Tensor = torch.cat(
            [SOC, T, SC.unsqueeze(-2).expand(-1, X.shape[-2], -1)], dim=-1
        )  # 参考: 论文公式 (43) [SOC, T, η, R, Q] 作为 MLP 输入

        # 步骤 3: 使用 p_net 估计原始参数
        # p_net 处理 x，输出原始参数估计 p，形状 [batch_dims, seq_len, 5]
        p: Tensor = self.p_net(x)  # 参考: 论文 Fig. 2 "MLP" 用于 EC 变量

        # 步骤 4: 应用 sigmoid 变换，将参数约束在边界内
        # theta = lb + range * sigmoid(0.01 * p) 将 p 映射到 [lb, ub]，形状 [batch_dims, seq_len, 5]
        # 0.01 缩放 sigmoid，避免梯度饱和
        theta = self.lb + self.range * torch.sigmoid(0.01 * p)  # 参考: 论文公式 (44) Θ = Θ_l + (Θ_u - Θ_l) σ(λ Θ_0)

        # 步骤 5: 从 theta 中提取各个参数
        # 将 theta 分割成 5 个组件，对应 R_1, 1/(R_1 C_1), OCV-U, -M_H, K_H
        R_1 = theta[..., 0:1]  # RC 电路的电阻，形状 [batch_dims, seq_len, 1]
        R_1C_1_r = theta[..., 1:2]  # R_1 C_1 时间常数的倒数，形状 [batch_dims, seq_len, 1]
        OCV_U = theta[..., 2:3]  # 开路电压减去终端电压，形状 [batch_dims, seq_len, 1]
        _M_H = theta[..., 3:4]  # 负的滞后电压最大值，形状 [batch_dims, seq_len, 1]
        K_H = theta[..., 4:5]  # 滞后因子，形状 [batch_dims, seq_len, 1]  # 参考: 论文 Fig. 5 (EC 变量)

        # 步骤 6: 估计初始串联电阻 R_s_init
        # R_s_init = SC[..., 2:3] * (1 + R_s_init_net(x[:, 0, :]))，其中 SC[..., 2:3] 为基础电阻
        # x[:, 0, :] 取第一个时间步，形状 [batch_dims, 5]
        R_s_init = SC[..., 2:3] * (1 + self.R_s_init_net(x[:, 0, :]))  # 参考: 论文 Fig. 5 (R_0 初始估计)

        # 步骤 7: 通过拼接所有变量为 ODE 求解器准备输入
        # X_ode 组合 t, I, T, SOC 和 theta 参数，形状 [batch_dims, seq_len, 8]
        X_ode = torch.cat(
            [t, I, T, SOC, R_1, R_1C_1_r, _M_H, K_H], dim=-1
        )  # 参考: 论文 Fig. 5 (微分方程的输入)

        # 步骤 8: 为 ODE 求解器创建时间掩码
        # t_mask 是长度为 8 (X_ode.shape[-1]) 的布尔张量，初始全 True
        # device 和 dtype 与 X_ode 匹配以确保兼容性
        t_mask = torch.ones(X_ode.shape[-1], device=X_ode.device, dtype=torch.bool)
        # 将除第一元素 (时间 t) 外的所有元素设为 False，指示仅 t 为独立变量
        t_mask[1:] = False  # 参考: 论文公式 (40) (Euler 方法使用 t 作为独立变量)

        # 步骤 9: 计算 ODE 的初始电压
        # U_H_init: 初始滞后电压，形状 [batch_dims, 1]
        # 若 U_H_init_net 为 None，则使用零值；否则，从 SC 和初始 SOC 估计
        U_H_init = (
            torch.zeros(*X_ode.shape[:1], 1, device=X_ode.device, dtype=X_ode.dtype)
            if self.U_H_init_net is None
            else self.U_H_init_net(torch.cat([SC, SOC[:, 0, :]], dim=-1))
        )  # 参考: 论文 Fig. 5 (U_H 的初始条件)

        # U_1_init: 初始 RC 电路电压，根据初始条件计算
        # 形状 [batch_dims, 1]，基于等效电路方程
        U_1_init = -OCV_U[:, 0, :] - U_H_init - I[:, 0, :] * R_s_init  # 参考: 论文 Fig. 5 (U = OCV + U_H + U_1 + I * R)

        # 步骤 10: 求解 ODE 以获得电压轨迹
        # odeint 使用 Euler 方法求解微分方程
        # 输入: 初始 [U_H_init, U_1_init], X_ode, t_mask；输出: voltages [seq_len, batch_dims, 2]
        voltages: Tensor = odeint(
            self.dvoltages_dt_func,
            torch.cat([U_H_init, U_1_init], dim=-1),
            X_ode,
            t_mask=t_mask,
            method="euler",
        )  # 参考: 论文公式 (40) (用于 U_H, U_1 的 Euler 积分)
        # 转置以匹配预期形状 [batch_dims, seq_len, 2]
        voltages = voltages.transpose(0, 1)

        # 步骤 11: 提取滞后电压和 RC 电压
        # 将 voltages 分割成 U_H 和 U_1，形状均为 [batch_dims, seq_len, 1]
        U_H, U_1 = voltages[..., :1], voltages[..., 1:]  # 参考: 论文 Fig. 5 (U_H 和 U_1 动态)

        # 步骤 12: 计算内部电阻估计
        # IR_estim 是 (-OCV_U - U_H - U_1) / I 随时间平均值，形状 [batch_dims, 1]
        # nanmean 处理可能的 NaN 值（来自除法）
        IR_estim: Tensor = torch.nanmean(
            (-OCV_U - U_H - U_1) / I, dim=-2
        )  # 参考: 论文 III-B (从等效电路估计 R)

        # 步骤 13: 根据 return_params 标志返回结果
        # 若 True，则返回 IR_estim, theta 和 voltages；否则，仅返回 IR_estim
        if return_params:
            return IR_estim, theta, voltages
        else:
            return IR_estim

# def forward(
#     self, X: Tensor, SC: Tensor, return_params=False
# ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
#     # X.shape: [batch_dims, seq_len, 5]
#     # SC.shape: [batch_dims, 3]
#     # Q = SC[..., 0:1]  # [batch_dims, 1]
#     # eta_0 = SC[..., 1:2]  # [batch_dims, 1]
#     t = X[..., 0:1]  # [batch_dims, seq_len, 1]
#     I = X[..., 1:2]  # [batch_dims, seq_len, 1]
#     T = X[..., 2:3]  # [batch_dims, seq_len, 1]
#     # U = X[..., 3:4]  # [batch_dims, seq_len, 1]
#     SOC = X[..., 4:]  # [batch_dims, seq_len, 1]

#     x: Tensor = torch.cat(
#         [SOC, T, SC.unsqueeze(-2).expand(-1, X.shape[-2], -1)], dim=-1
#     )  # [batch_dims, seq_len, 5]
#     p: Tensor = self.p_net(x)  # [batch_dims, seq_len, 5]
#     theta = self.lb + self.range * torch.sigmoid(
#         0.01 * p
#     )  # [batch_dims, seq_len, 5]

#     R_1 = theta[..., 0:1]  # [batch_dims, seq_len, 1]
#     R_1C_1_r = theta[..., 1:2]  # [batch_dims, seq_len, 1]
#     OCV_U = theta[..., 2:3]  # [batch_dims, seq_len, 1]
#     _M_H = theta[..., 3:4]  # [batch_dims, seq_len, 1]
#     K_H = theta[..., 4:5]  # [batch_dims, seq_len, 1]
#     R_s_init = SC[..., 2:3] * (1 + self.R_s_init_net(x[:, 0, :]))  # [batch_dims, 1]

#     X_ode = torch.cat(
#         [t, I, T, SOC, R_1, R_1C_1_r, _M_H, K_H], dim=-1
#     )  # [batch_dims, seq_len, 8]
#     t_mask = torch.ones(X_ode.shape[-1], device=X_ode.device, dtype=torch.bool)
#     t_mask[1:] = False #除第一列外设为 False，表示仅时间维度为独立变量

#     U_H_init = (
#         torch.zeros(*X_ode.shape[:1], 1, device=X_ode.device, dtype=X_ode.dtype)
#         if self.U_H_init_net is None
#         else self.U_H_init_net(torch.cat([SC, SOC[:, 0, :]], dim=-1))
#     )  # [batch_dims, 1]

#     U_1_init = -OCV_U[:, 0, :] - U_H_init - I[:, 0, :] * R_s_init  # [batch_dims, 1]

#     voltages: Tensor = odeint(
#         self.dvoltages_dt_func,
#         torch.cat([U_H_init, U_1_init], dim=-1),
#         X_ode,
#         t_mask=t_mask,
#         method="euler",
#     )  # [seq_len, batch_dims, 2]
#     voltages = voltages.transpose(0, 1)  # [batch_dims, seq_len, 2]
#     U_H, U_1 = voltages[..., :1], voltages[..., 1:]

#     IR_estim: Tensor = torch.nanmean(
#         (-OCV_U - U_H - U_1) / I, dim=-2
#     )  # [batch_dims, 1]

#     if return_params:
#         return IR_estim, theta, voltages
#     else:
#         return IR_estim


class ChannelAttention(nn.Module):
    def __init__(self, x_channels, hc=16):
        super().__init__()
        self.max_pool_layer = nn.AdaptiveMaxPool1d(1)
        self.avg_pool_layer = nn.AdaptiveAvgPool1d(1)

        self.mlp_layer = nn.Sequential(
            nn.Linear(x_channels, hc),
            nn.Softplus(),
            nn.Linear(hc, x_channels),
        )
        self.scoring_layer = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # x.shape: [batch_dims, seq_len, x_channels]
        x = x.transpose(-2, -1)  # [batch_dims, x_channels, seq_len]
        max_out: Tensor = self.max_pool_layer(x)  # [batch_dims, x_channels, 1]
        max_out = self.mlp_layer(
            max_out.transpose(-2, -1)
        )  # [batch_dims, 1, x_channels]
        avg_out: Tensor = self.avg_pool_layer(x)  # [batch_dims, x_channels, 1]
        avg_out = self.mlp_layer(
            avg_out.transpose(-2, -1)
        )  # [batch_dims, 1, x_channels]
        out: Tensor = max_out + avg_out  # [batch_dims, 1, x_channels]
        ca = self.scoring_layer(out)  # [batch_dims, 1, x_channels]
        return ca


class TemporalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool_layer = nn.AdaptiveMaxPool1d(1)
        self.avg_pool_layer = nn.AdaptiveAvgPool1d(1)
        self.linear_layer = nn.Linear(2, 1)
        self.scoring_layer = nn.Softmax(dim=-2)

    def forward(self, x: Tensor) -> Tensor:
        # x.shape: [batch_dims, seq_len, x_channels]
        max_out = self.max_pool_layer(x)  # [batch_dims, seq_len, 1]
        avg_out = self.avg_pool_layer(x)  # [batch_dims, seq_len, 1]
        out = torch.cat([max_out, avg_out], dim=-1)  # [batch_dims, seq_len, 2]
        out: Tensor = self.linear_layer(out)  # [batch_dims, seq_len, 1]
        ta = self.scoring_layer(out)  # [batch_dims, seq_len, 1]
        return ta


class QNet(nn.Module):
    def __init__(
        self,
        X: Tensor,
        SC: Tensor,
        y_channels: int = 1,
        h_channels: int = 16,
        channel_attn=True,
        temporal_attn=False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        x_channels = X.shape[-1]
        sc_channels = SC.shape[-1]
        hc = 2 * h_channels

        if channel_attn:
            self.ca_net = ChannelAttention(x_channels=x_channels, hc=hc)
            init_network_weights(self.ca_net)
        else:
            self.ca_net = None
        if temporal_attn:
            self.ta_net = TemporalAttention()
            init_network_weights(self.ta_net)
        else:
            self.ta_net = None

        kc, sc, dc, kp, sp, dp = 3, 1, 1, 5, 1, 1
        lc, lp = (
            sc - dc * (kc - 1) - 1,
            sp - dp * (kp - 1) - 1,
        )  # 分别根据卷积层/池化层的特征而定
        q = sc * sp
        l = lc + sc * lp
        m, om = 1, 44
        self.n = om * q**m - l * sum([q**i for i in range(0, m)])

        module_list = [
            nn.Conv1d(
                in_channels=x_channels,
                out_channels=hc,
                kernel_size=kc,
                stride=sc,
                dilation=dc,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kp, stride=sp, dilation=dp),
        ]
        for _ in range(m - 1):
            module_list.extend(
                [
                    nn.Conv1d(
                        in_channels=hc,
                        out_channels=hc,
                        kernel_size=kc,
                        stride=sc,
                        dilation=dc,
                    ),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=kp, stride=sp, dilation=dp),
                ]
            )
        module_list.extend(
            [nn.Flatten(), nn.Linear(in_features=hc * om, out_features=h_channels)]
        )
        self.x_encoder = nn.Sequential(*module_list)
        init_network_weights(self.x_encoder)

        self.sc_encoder = nn.Sequential(
            nn.Linear(in_features=sc_channels, out_features=h_channels),
            nn.Softplus(),
            nn.Linear(in_features=h_channels, out_features=h_channels),
        )
        init_network_weights(self.sc_encoder)

        self.regressor = nn.Sequential(
            nn.Linear(in_features=h_channels * 2, out_features=y_channels),
            nn.Softplus(),
            nn.Linear(in_features=y_channels, out_features=y_channels),
        )
        init_network_weights(self.regressor)

    def forward(
        self, X: Tensor, SC: Tensor, return_attn=False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        # X.shape: [batch_dims, seq_len, x_channels]
        # SC.shape: [batch_dims, sc_channels]
        if self.ca_net is not None:
            channel_attn = self.ca_net(X)  # [batch_dims, 1, x_channels]
            X = X * channel_attn
        else:
            channel_attn = torch.ones(
                *X.shape[:-2], 1, X.shape[-1], dtype=X.dtype, device=X.device
            )
        if self.ta_net is not None:
            temporal_attn = self.ta_net(X)  # [batch_dims, seq_len, 1]
            X = X * temporal_attn
        else:
            temporal_attn = torch.ones(*X.shape[:-1], 1, dtype=X.dtype, device=X.device)
        X = X.transpose(-2, -1)  # [batch_dims, x_channels, seq_len]
        if X.shape[-1] != self.n:
            X: torch.Tensor = torch.nn.functional.interpolate(
                X, size=self.n, mode="linear", align_corners=True
            )  # [batch_dims, x_channels, n]
        x_fea: Tensor = self.x_encoder(X)  # [batch_dims, h_channels]

        sc_fea: Tensor = self.sc_encoder(SC)  # [batch_dims, h_channels]

        fea = torch.cat((x_fea, sc_fea), dim=-1)  # [batch_dims, 2 * h_channels]
        Q_estim = SC[..., 2:3] * (1 + self.regressor(fea))  # [batch_dims, y_channels]

        if return_attn:
            return Q_estim, channel_attn, temporal_attn
        else:
            return Q_estim


class OneNet(nn.Module):
    def __init__(self, X: Tensor, SC: Tensor, dataset: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # X.shape: [batch_dims, seq_len, [t, I, T, U, SOC, R_1, R_1C_1_r, OCV_U, _M_H, K_H, U_H, U_1]]
        # SC.shape: [batch_dims, [IR, eta_0, Q]]

        # 对 SOCNet
        # X 只取前 4 个维度 [t, I, T, U]
        # SC 需要将 [IR, eta_0, Q] 变为 [Q, eta_0, IR, SOC[0]]
        x = X[..., 0:4]  # [batch_dims, seq_len, 4]
        sc = torch.concat(
            [SC[..., [2, 1, 0]], X[..., 0, 4:5]], dim=-1
        )  # [batch_dims, 4]
        self.soc_net = SOCNet(X=x, SC=sc)

        # 对 IRNet
        # X 只取前 5 个维度 [t, I, T, U, SOC]
        # SC 需要将 [IR, eta_0, Q] 变为 [Q, eta_0, IR]
        x = X[..., 0:5]  # [batch_dims, seq_len, 5]
        sc = SC[..., [2, 1, 0]]  # [batch_dims, 3]
        self.ir_net = IRParamsNet(X=x, SC=sc, dataset=dataset)

        self.q_net = QNet(X=X, SC=SC)

    def forward(self, X: Tensor, SC: Tensor) -> Tensor:
        # X.shape: [batch_dims, seq_len, 12]
        # SC.shape: [batch_dims, 3]
        # X 只取前 4 个维度 [t, I, T, U]
        # SC 需要将 [IR, eta_0, Q] 变为 [Q, eta_0, IR, SOC[0]]
        x = X[..., 0:4]  # [batch_dims, seq_len, 4]
        sc = torch.concat(
            [SC[..., [2, 1, 0]], X[..., 0, 4:5]], dim=-1
        )  # [batch_dims, 4]
        SOC_estim = self.soc_net.forward(x, sc)  # [batch_dims, seq_len, 1]

        # X 扩充至 5 个维度 [t, I, T, U, SOC_estim]
        # SC 需要将 [IR, eta_0, Q] 变为 [Q, eta_0, IR]
        x = torch.concat([x, SOC_estim], dim=-1)  # [batch_dims, seq_len, 5]
        sc = SC[..., [2, 1, 0]]  # [batch_dims, 3]
        IR_estim, theta, voltages = self.ir_net.forward(
            x, sc, return_params=True
        )  # [batch_dims, 1]

        # X 扩充至 12 个维度 [t, I, T, U, SOC_estim, R_1, R_1C_1_r, OCV_U, _M_H, K_H, U_H, U_1]
        # SC 需要将 [Q, eta_0, IR] 变为 [IR_estim, eta_0, Q]
        x = torch.concat([x, theta, voltages], dim=-1)  # [batch_dims, seq_len, 12]
        sc = torch.concat([IR_estim, SC[..., 1:]], dim=-1)  # [batch_dims, seq_len, 3]
        Q_estim = self.q_net.forward(x, sc)  # [batch_dims, 1]
        return Q_estim


class CNNForIR(nn.Module):
    def __init__(
        self, X: torch.Tensor, SC: torch.Tensor, h_channels=16, *args, **kwargs
    ):
        super().__init__()
        x_channels = X.shape[-1]
        sc_channels = SC.shape[-1]
        y_channels = 1
        kc, sc, dc, kp, sp, dp = 3, 1, 1, 5, 1, 1
        lc, lp = (
            sc - dc * (kc - 1) - 1,
            sp - dp * (kp - 1) - 1,
        )  # 分别根据卷积层/池化层的特征而定
        q = sc * sp
        l = lc + sc * lp
        m, om = 1, 44
        self.n = om * q**m - l * sum([q**i for i in range(0, m)])

        module_list = [
            nn.Conv1d(
                in_channels=x_channels + sc_channels,
                out_channels=h_channels,
                kernel_size=kc,
                stride=sc,
                dilation=dc,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kp, stride=sp, dilation=dp),
        ]
        for _ in range(m - 1):
            module_list.extend(
                [
                    nn.Conv1d(
                        in_channels=h_channels,
                        out_channels=h_channels,
                        kernel_size=kc,
                        stride=sc,
                        dilation=dc,
                    ),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=kp, stride=sp, dilation=dp),
                ]
            )
        module_list.extend(
            [
                nn.Flatten(),
                nn.Linear(in_features=h_channels * om, out_features=y_channels),
            ]
        )
        self.net = nn.Sequential(*module_list)
        init_network_weights(self.net)

    def forward(self, X: Tensor, SC: Tensor):
        # Get the batch_dims and seq_len
        batch_dims, seq_len, _ = X.shape
        # Repeat SC along the seq_len dimension
        SC_expanded = SC.unsqueeze(-2).expand(batch_dims, seq_len, -1)
        # Concatenate X and SC along the last dimension
        X_cat = torch.cat((X, SC_expanded), dim=-1)
        # Permute X_cat to match the input shape for Conv1d (batch_dims, channels, seq_len)
        X_cat = X_cat.permute(0, 2, 1)
        y = self.net(X_cat)
        return y


CNNForQ = type("CNNForQ", (CNNForIR,), {})


class LSTMForSOC(nn.Module):
    def __init__(
        self,
        X: torch.Tensor,
        SC: torch.Tensor,
        h_channels=16,
        num_layers=1,
        *args,
        **kwargs
    ):
        super().__init__()
        x_channels = X.shape[-1]
        sc_channels = SC.shape[-1]
        y_channels = 1
        self.h_channels = h_channels
        self.num_layers = num_layers
        self.lstm_input_size = x_channels + sc_channels

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=h_channels,
            num_layers=num_layers,
            batch_first=True,
        )
        init_network_weights(self.lstm)

        # Define the output fully connected layer
        self.net = nn.Linear(h_channels, y_channels)
        init_network_weights(self.net)

    def forward(self, X: Tensor, SC: Tensor):
        batch_dims, seq_len, _ = X.shape
        # Repeat SC for each time step and concatenate with X
        sc_expanded = SC.unsqueeze(-2).expand(
            -1, seq_len, -1
        )  # shape: [batch_dims, seq_len, sc_channels]
        lstm_input = torch.cat(
            (X, sc_expanded), dim=-1
        )  # shape: [batch_dims, seq_len, x_channels + sc_channels]
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_dims, self.h_channels).to(X.device)
        c0 = torch.zeros(self.num_layers, batch_dims, self.h_channels).to(X.device)
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(
            lstm_input, (h0, c0)
        )  # lstm_out shape: [batch_dims, seq_len, h_channels]
        # Map the LSTM outputs to the desired output dimension
        y = self.net(lstm_out)  # shape: [batch_dims, seq_len, y_channels]
        return y


def fore2regr_forward(self: nn.Module, *args, **kwargs) -> Tensor:
    original_output: Tensor = super(self.__class__, self).forward(
        *args, **kwargs
    )  # [batch_dims, seq_len, y_channels]
    return torch.nanmean(original_output, dim=-2)  # [batch_dims, y_channels]


LSTMForIR = type("LSTMForIR", (LSTMForSOC,), {"forward": fore2regr_forward})
LSTMForQ = type("LSTMForQ", (LSTMForSOC,), {"forward": fore2regr_forward})


class TFTForSOC(nn.Module):
    # 定义Temporal Fusion Transformer模型
    def __init__(
        self,
        X: torch.Tensor,
        SC: torch.Tensor,
        nhead=4,
        x_encoder_layers=1,
        x_hidden_channels=2048,
        sc_proj_channels=4,
        load_SOCNet: str | None = None,
        load_IRNet: str | None = None,
        *args,
        **kwargs
    ):
        super().__init__()
        x_channels = X.shape[-1]
        sc_channels = SC.shape[-1]
        y_channels = 1
        # 定义模型的各层
        x_proj_channels = 3 * nhead
        self.projection_layer = nn.Sequential(
            nn.Linear(x_channels, x_proj_channels),
            nn.Tanh(),
            nn.Linear(x_proj_channels, x_proj_channels),
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=x_proj_channels,
            nhead=nhead,
            dim_feedforward=x_hidden_channels,
            batch_first=True,
        )
        self.x_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=x_encoder_layers
        )
        init_network_weights(self.x_encoder)
        # 静态特征处理层
        self.sc_encoder = self.sc_encoder = nn.Sequential(
            nn.Linear(in_features=sc_channels, out_features=sc_proj_channels),
            nn.Softplus(),
            nn.Linear(in_features=sc_proj_channels, out_features=sc_proj_channels),
        )
        init_network_weights(self.sc_encoder)
        # 输出层
        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=x_proj_channels + sc_proj_channels, out_features=y_channels
            ),
            nn.Softplus(),
            nn.Linear(in_features=y_channels, out_features=y_channels),
        )
        init_network_weights(self.decoder)
        if load_SOCNet is not None:
            self.soc_net = SOCNet(X=X, SC=SC)
            self.soc_net.load_state_dict(torch.load(load_SOCNet, weights_only=True))
        else:
            self.soc_net = None
        if load_IRNet is not None:
            self.ir_net = IRParamsNet(X=X, SC=SC, dataset=kwargs["dataset"])
            self.ir_net.load_state_dict(torch.load(load_IRNet, weights_only=True))
        else:
            self.ir_net = None

    def forward(self, X: Tensor, SC: Tensor):
        # 将输入x传入Transformer Encoder
        H_x = self.projection_layer(X)
        H_x: torch.Tensor = self.x_encoder(H_x)
        # 将静态特征通过全连接层处理
        H_sc: torch.Tensor = self.sc_encoder(SC)
        # 将Transformer输出和静态特征进行拼接
        H = torch.cat([H_x, H_sc.unsqueeze(-2).repeat(1, H_x.shape[-2], 1)], dim=-1)
        # 最终输出层
        y = self.decoder(H)
        if self.soc_net is not None:
            y = self.soc_net.forward(X=X, SC=SC).detach() * y
        if self.ir_net is not None:
            y = self.ir_net.forward(X=X, SC=SC).unsqueeze(-2).detach() * y
        return y


TFTForIR = type("TFTForIR", (TFTForSOC,), {"forward": fore2regr_forward})
TFTForQ = type("TFTForQ", (TFTForSOC,), {"forward": fore2regr_forward})


class ODEFunc(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_dim = hidden_channels
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        init_network_weights(self.net)

    def forward(self, t, h):
        return self.net(h)


class ODERNNCell(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.ode_func = ODEFunc(hidden_channels)
        self.rnn_cell = nn.GRUCell(input_channels, hidden_channels)

    def forward(self, h, x, delta_t):
        h = odeint(self.ode_func, h, delta_t)[-1]
        h = self.rnn_cell(x, h)
        return h


class ODERNNForSOC(nn.Module):
    def __init__(
        self, X: torch.Tensor, SC: torch.Tensor, h_channels=16, *args, **kwargs
    ):
        super().__init__()
        x_channels = X.shape[-1]
        sc_channels = SC.shape[-1]
        y_channels = 1
        self.h_channels = h_channels
        self.ode_rnn_cell = ODERNNCell(x_channels + sc_channels, h_channels)
        self.output_layer = nn.Linear(h_channels, y_channels)

    def forward(self, X: Tensor, SC: Tensor):
        batch_dims, seq_len, _ = X.shape
        t = X[..., 0:1]  # [batch_dims, seq_len, 1]
        h = torch.zeros(batch_dims, self.h_channels).to(X.device)

        # Repeat SC for each time step and concatenate with X
        SC_repeated = SC.unsqueeze(-2).repeat(1, seq_len, 1)
        X_concat = torch.cat([X, SC_repeated], dim=-1)
        hs = []
        for i in range(-1, seq_len - 1):
            if i < 0:
                delta_t = torch.stack([t[..., 0, :], t[..., 1, :]], dim=-2)
            else:
                delta_t = torch.stack([t[..., i, :], t[..., i + 1, :]], dim=-2)
            h = self.ode_rnn_cell(h, X_concat[:, i, :], delta_t)
            hs.append(h)
        hs = torch.stack(hs, dim=-2)
        y = self.output_layer(hs)
        return y


ODERNNForIR = type("ODERNNForIR", (ODERNNForSOC,), {"forward": fore2regr_forward})
ODERNNForQ = type("ODERNNForQ", (ODERNNForSOC,), {"forward": fore2regr_forward})


class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(CDEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels * input_channels),
        )
        init_network_weights(self.net)
        self.hidden_channels = hidden_channels
        self.input_channels = input_channels
        self.time = None
        self.data = None

    def store_continuous_state(self, time: Tensor, data: Tensor):
        # time.shape: [seq_len]
        # data.shape: [batch_dims, seq_len, input_channels]
        self.time = time
        self.data = data

    def get_continuous_state(self, t: Tensor) -> Tensor:
        # t.shape: [1, 1]
        t = t[0].squeeze()
        # 判断s的位置以确定j（整数）
        j: Tensor = torch.searchsorted(self.time, t)
        if j == 0:  # 目标值比所有元素都小
            j = 0  # 使用第一段插值函数
        elif j == len(self.time):  # 目标值比所有元素都大
            j = len(self.time) - 2  # 使用最后一段插值函数
        else:  # 目标值在排序后张量中的位置在 [index-1, index) 范围内
            j = int((j - 1).item())

        dx = self.data[:, j + 1, ...] - self.data[:, j, ...]  # 暂时用线性估计
        return dx  # [batch_dims, input_channels]

    def forward(self, t, z: Tensor):
        # z.shape [batch_dims, hidden_channels]
        vf: Tensor = self.net(z)  # [batch_dims, hidden_channels * input_channels]
        vf = vf.view(
            -1, self.hidden_channels, self.input_channels
        )  # [batch_dims, hidden_channels, input_channels]
        dx = self.get_continuous_state(t).unsqueeze(
            -2
        )  # [batch_dims, 1, input_channels]
        out = torch.sum(vf * dx, dim=-1)  # [batch_dims, hidden_channels]
        return out


class CDEForSOC(torch.nn.Module):
    def __init__(
        self, X: torch.Tensor, SC: torch.Tensor, h_channels=16, *args, **kwargs
    ):
        super().__init__()
        x_channels = X.shape[-1]
        sc_channels = SC.shape[-1]
        y_channels = 1
        self.input_channels = x_channels + sc_channels
        self.h_channels = h_channels

        self.initial = torch.nn.Linear(self.input_channels, h_channels)
        init_network_weights(self.initial)
        self.func = CDEFunc(self.input_channels, h_channels)
        self.net = torch.nn.Linear(h_channels, y_channels)
        init_network_weights(self.net)

    def forward(self, X: Tensor, SC: Tensor):
        # Repeat sc along the seq_len dimension
        SC = SC.unsqueeze(-2).repeat(1, X.shape[-2], 1)
        # Concatenate x and sc
        X = torch.cat([X, SC], dim=-1)
        z0 = self.initial(X[:, 0, :])  # [batch_dims, h_channels]
        t = torch.arange(X.shape[-2], dtype=torch.int, device=X.device)
        self.func.store_continuous_state(t, X)
        z = odeint(
            self.func, z0, t.unsqueeze(0).unsqueeze(-1)
        )  # [seq_len, batch_dims, h_channels]
        z = z.transpose(0, 1)  # [batch_dims, seq_len, h_channels]
        y = self.net(z)
        return y


CDEForIR = type("CDEForIR", (CDEForSOC,), {"forward": fore2regr_forward})
CDEForQ = type("CDEForQ", (CDEForSOC,), {"forward": fore2regr_forward})


class VoltageODEFunc(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x, voltages: Tensor) -> Tensor:
        # x[0]: ([I, R_1, R_1C_1_r, _M_H, K_H].shape: [batch_dims, 5])
        # x[1]: (d[I, R_1, R_1C_1_r, _M_H, K_H]_ds.shape: [batch_dims, 5])
        # voltages.shape: [batch_dims, 2]
        I: Tensor = x[0][:, 0:1]  # [batch_dims, 1]
        R_1: Tensor = x[0][:, 1:2]  # [batch_dims, 1]
        R_1C_1_r: Tensor = x[0][:, 2:3]  # [batch_dims, 1]
        _M_H: Tensor = x[0][:, 3:4]  # [batch_dims, 1]
        K_H: Tensor = x[0][:, 4:5]  # [batch_dims, 1]

        U_H: Tensor = voltages[:, :1]  # [batch_dims, 1]
        U_1: Tensor = voltages[:, 1:]  # [batch_dims, 1]

        dU_H_dt = -K_H * I * (-_M_H - U_H)  # [batch_dims, 1]
        # 注意：I < 0，这里 K_H * I 绝对值直接取负值
        dU_1_dt = R_1C_1_r * (R_1 * I - U_1)  # [batch_dims, 1]
        dvoltages_dt = torch.concat([dU_H_dt, dU_1_dt], dim=-1)  # [batch_dims, 2]

        return dvoltages_dt


class VoltageNet(nn.Module):
    def __init__(
        self,
        X: Tensor,
        SC: Tensor,
        dataset: str,
        use_U_H_init: bool = False,
        ub: list | None = None,
        lb: list | None = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        x_channels = 3

        self.Qn = params[dataset]["Q_n"]
        self.ub = torch.tensor(
            (
                [
                    params[dataset][p][1]
                    for p in ["R_1", "1/R_1C_1", "OCV - U", "-M_H", "K_H"]
                    + params[dataset]["OCVSOCParams"]
                ]
                if ub is None
                else ub
            ),
            dtype=X.dtype,
            device=X.device,
        )
        self.lb = torch.tensor(
            (
                [
                    params[dataset][p][0]
                    for p in ["R_1", "1/R_1C_1", "OCV - U", "-M_H", "K_H"]
                    + params[dataset]["OCVSOCParams"]
                ]
                if lb is None
                else lb
            ),
            dtype=X.dtype,
            device=X.device,
        )
        self.range = self.ub - self.lb  # [5 + p]
        self.p_net = nn.Sequential(
            nn.Linear(x_channels, 2 * x_channels),
            nn.Softplus(),
            nn.Linear(2 * x_channels, self.range.shape[-1]),
        )

        self.OCV_SOC_curve = params[dataset]["OCVSOCcurve"]

        self.dvoltages_dt_func = VoltageODEFunc()

        if use_U_H_init:
            self.U_H_init_module = nn.Sequential(
                nn.Linear(3, 1), nn.Softplus(), nn.Linear(1, 1)
            )
        else:
            self.U_H_init_module = None

    def forward(
        self, X: Tensor, SC: Tensor, return_params=False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        # X.shape: [batch_dims, seq_len, 3]
        # SC.shape: [batch_dims, 2]
        t = X[..., 0:1]  # [batch_dims, seq_len, 1]
        I = X[..., 1:2]  # [batch_dims, seq_len, 1]
        T = X[..., 2:3]  # [batch_dims, seq_len, 1]
        Q = SC[..., 0:1]  # [batch_dims, 1]
        R_0 = SC[..., 1:2]  # [batch_dims, 1]
        SOC = torch.concat(
            [
                torch.zeros_like(I[..., :1, :]),  # [batch_dims, 1, 1]
                (I[..., 1:, :] + I[..., :-1, :])
                * (t[..., 1:, :] - t[..., :-1, :]),  # [batch_dims, seq_len - 1, 1]
            ],
            dim=-2,
        ) / (
            2 * 3600 * self.Qn
        )  # [batch_dims, seq_len, 1]
        SOC: torch.Tensor = torch.cumsum(SOC, dim=-2)  # [batch_dims, seq_len, 1]
        SOC += Q.unsqueeze(-2).expand(-1, 1, -1) / self.Qn  # [batch_dims, seq_len, 1]

        x = torch.concat(
            [
                SOC,
                R_0.unsqueeze(-2).expand(-1, SOC.shape[-2], -1),
                T.nanmean(dim=-2, keepdim=True).repeat_interleave(
                    SOC.shape[-2], dim=-2
                ),
            ],
            dim=-1,
        )  # [batch_dims, seq_len, 3]
        p = self.p_net(x)  # [batch_dims, seq_len, 5 + p]
        theta = self.lb + self.range * torch.sigmoid(
            0.01 * p
        )  # [batch_dims, seq_len, 5 + p]

        R_1 = theta[..., 0:1]  # [batch_dims, seq_len, 1]
        R_1C_1_r = theta[..., 1:2]  # [batch_dims, seq_len, 1]
        OCV_U = theta[..., 2:3]  # [batch_dims, seq_len, 1]
        _M_H = theta[..., 3:4]  # [batch_dims, seq_len, 1]
        K_H = theta[..., 4:5]  # [batch_dims, seq_len, 1]
        OCV_SOC_params = theta[..., 5:]  # [batch_dims, seq_len, p]

        OCV = self.OCV_SOC_curve(SOC, OCV_SOC_params)  # [batch_dims, seq_len, 1]

        X_ode = torch.concat(
            [I, R_1, R_1C_1_r, _M_H, K_H], dim=-1
        )  # [batch_dims, seq_len, 5]
        t_mask = torch.ones(X_ode.shape[-1], device=X_ode.device, dtype=torch.bool)
        t_mask[1:] = False
        U_H_init = (
            torch.zeros(*X_ode.shape[:1], 1, device=X_ode.device, dtype=X_ode.dtype)
            if self.U_H_init_module is None
            else self.U_H_init_module(torch.concat([SC, SOC[:, 0, :]], dim=-1))
        )  # [batch_dims, 1]
        U_1_init = -OCV_U[:, 0, :] - U_H_init - I[:, 0, :] * R_0  # [batch_dims, 1]

        voltages: Tensor = odeint(
            self.dvoltages_dt_func,
            torch.concat([U_H_init, U_1_init], dim=-1),
            X_ode,
            t_mask=t_mask,
            method="euler",
        )  # [seq_len, batch_dims, 2]
        voltages = voltages.transpose(0, 1)  # [batch_dims, seq_len, 2]
        U_H, U_1 = voltages[..., :1], voltages[..., 1:]

        pred_U: Tensor = (
            OCV + U_H + U_1 + I * R_0.unsqueeze(-2)
        )  # [batch_dims, seq_len, 1]

        if return_params:
            return pred_U, theta, voltages
        else:
            return pred_U
