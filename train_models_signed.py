import math
import random
from unittest.mock import Mock
from pathlib import Path
from datetime import date
import pickle
import numpy as np
import torch
from warnings import simplefilter
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple
from torchode.repara import _Repara, PCHIPRepara

from utils import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        root_mean_squared_error,
    )

from models import (
    W_SOCNet,
    SOCNet
)

torch.autograd.set_detect_anomaly(True)  # 检测梯度异常

# 随机数种子
random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)

# 使用CPU或GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 屏蔽FutureWarning
simplefilter(action="ignore", category=FutureWarning)


class LogCoshLoss(torch.nn.Module):
    # 回归损失函数
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        delta = y_pred - y_true
        #这行代码实现了 Log-Cosh 损失的数学公式。Log-Cosh 损失的定义是                   log(cosh(x)) = log((e^x + e^-x) / 2)           ，其中 x 是误差 delta
        #为了避免数值不稳定（特别是当 x 的值很大时），代码中可以利用数学上的一个近似公式：log(cosh(x))≈softplus(x)+softplus(−x)−log(2)
        #实际代码使用了另一个等效且更简化的公式：                                        log(cosh(x))=x+softplus(−2x)−log(2)            ,这个公式在数值计算上更稳定
        return torch.nanmean(
            delta + torch.nn.functional.softplus(-2.0 * delta) - math.log(2.0)
        )



def load_data(
    filename: str | None = None,
    dataset: str = "ordered",
    model_type: type | None = None,
    device: torch.device = device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #在 Python 的类型提示中，tuple 是 typing 模块中的一个类型对象，用于描述元组的类型结构，而方括号 [] 是用来指定元组中元素的具体类型，而不是用小括号 ()

    """加载数据集并将其转换为 PyTorch 张量，分配到指定设备（CPU 或 GPU）。

    Args:
        filename (str | None): 数据文件的路径，如果为 None，则根据 model_type 和 dataset 自动选择。
        dataset (str): 数据集名称，默认为 "a123"。
        model_type (type | None): 模型类型，用于选择对应的数据集文件。
        device (torch.device): 设备类型（CPU 或 GPU），用于张量分配。

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 包含输入特征 X、条件特征 SC 和目标值 Y 的张量元组。

    Raises:
        NotImplementedError: 如果 model_type 不在支持的模型列表中。
    """
    print("Step 1: Starting data loading...")  # 日志：步骤1 - 开始数据加载过程
    if filename is None:
        if model_type in [SOCNet]:
            filename = f"datasets/{dataset}/random_SOC_50.pkl"
        # 修改
        elif model_type in [W_SOCNet]:
            filename = f"datasets/{dataset}/resample_SOC_50.pkl"
        else:
            raise NotImplementedError

    print(f"Loading data from file: {filename}")  # 日志：指定加载的文件路径
    with open(filename, "rb") as f:
        #pickle.load(f) 从打开的文件对象 f 中加载数据，并将其反序列化成 Python 对象，通常是字典（如这里的 data_dict）
        data_dict = pickle.load(f)

    X, SC, Y = (
        torch.from_numpy(data_dict["X"]),
        torch.from_numpy(data_dict["SC"]),
        torch.from_numpy(data_dict["Y"]),
    )
    X: torch.Tensor = X.to(device)
    SC: torch.Tensor = SC.to(device)
    Y: torch.Tensor = Y.to(device)

    # 新增：针对 W_SOCNet 扩展 SC 到 [N, 4]（处理实际 pkl 可能 1D/单特征）
    if model_type == W_SOCNet:
        if SC.dim() == 1:  # [N] 1D
            SC = SC.unsqueeze(-1)  # 先变 [N, 1]，假设第一个是 Q
        if SC.shape[1] < 4:  # [N, k] k<4
            # 默认值：Q=已有的（或复制），eta_0=1.0, R=0.01, SOC_init_ref=0.5
            existing_feats = SC.shape[1]
            default_feats = torch.tensor([[1.0, 0.01, 0.5]], device=device).unsqueeze(0).repeat(SC.shape[0], 1)
            SC = torch.cat([SC, default_feats[:, :4 - existing_feats]], dim=-1)  # 拼接缺失特征

        print(f"Extended SC shape for W_SOCNet: {SC.shape}")  # 日志：W_SOCNet模型下扩展SC特征的形状（临时打印，可移除）

    print(f"Data loaded successfully. Shapes - X: {X.shape}, SC: {SC.shape}, Y: {Y.shape}")  # 日志：数据加载成功，显示各张量形状
    print("Step 1: Data loading completed.\n")  # 日志：步骤1 - 数据加载完成
    return X, SC, Y


def create_model(
    model_type: type,
    device: torch.device,
    X: torch.Tensor,
    SC: torch.Tensor,
    Y: torch.Tensor,
    dataset: str = "ordered",
) -> torch.nn.Module:
    
    print(f"Step 4: Creating model {model_type.__name__}...")  # 日志：步骤4 - 开始创建模型
    model = model_type(X=X, SC=SC, dataset=dataset).to(device)
    print(f"Model created and moved to device: {device}")  # 日志：模型创建完成并移至指定设备
    print("Step 4: Model creation completed.\n")  # 日志：步骤4 - 模型创建完成
    return model


class SaveAndEarlyStop:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        dataset: str,
        patience: int | None = 30,
        delta: float = 0.0,
        trace_func=print,
        save: str | None = None,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.delta = delta
        self.trace_func = trace_func
        self.counter = 0                #记录验证损失没改善的轮次
        self.best_score = None
        self.early_stop = False
        self.save = save
        self.dataset = dataset

    def stop(
        self,
        val_loss: float | torch.Tensor,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        if self.patience is None:
            if self.save is not None:
                self.save_model_with_code(model, optimizer)
            return False #没有早停，返回 False，训练继续
        else:
            score = -val_loss
            if self.best_score is None:#首次调用
                self.best_score = score
                if self.save is not None:
                    self.save_model_with_code(model, optimizer)
            elif score < self.best_score + self.delta:#损失未改善
                self.counter += 1
                if self.counter >= self.patience:
                    self.trace_func(
                        f"EarlyStopping counter: {self.counter} out of {self.patience}, best score is {self.best_score}"
                    )
                    self.early_stop = True
            else:#损失改善，更新
                self.best_score = score
                if self.save is not None:
                    self.save_model_with_code(model, optimizer)
                self.counter = 0
            return self.early_stop

    def save_model_with_code(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer
    ):
        # 自定义保存函数
        # 创建对应文件夹
        dirpath = f"save/{str(date.today())}/"
        Path(dirpath).mkdir(parents=True, exist_ok=True)#parents=True允许创建父目录，exist_ok=True避免目录已存在时抛出错误
        Path(f"final/{self.dataset}/").mkdir(parents=True, exist_ok=True)
        # 记录模型状态
        torch.save(model.state_dict(), dirpath + type(model).__name__ + ".pt")#.state_dict()返回模型的参数字典，包含所有可训练参数（如权重和偏置）
        torch.save(
            model.state_dict(), f"final/{self.dataset}/" + type(model).__name__ + ".pt"
        )
        torch.save(
            optimizer.state_dict(), dirpath + type(model).__name__ + "-Optimizer.pt"
        )

# 插值函数
def resample_with_pchip(
    sparse_list_X: List[torch.Tensor], 
    sparse_list_SC: List[torch.Tensor], 
    sparse_list_Y: List[torch.Tensor],
    target_len: int = 50,  # 原序列长度
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    使用 PCHIPRepara 对稀疏序列上采样到 target_len，不依赖原始时间步。
    假设每个稀疏序列在自己的均匀索引 [0, sparse_len-1] 上定义。
    SC 为静态数据，不进行插值，直接复制（保持无序列维度）。
    
    Args:
        sparse_list_*: 删除后的列表，每个 [sparse_len_i, features] (X/Y) 或 [sc_features] (SC)
        target_len: 目标统一长度 (e.g., 50)
        device: 张量设备
    
    Returns:
        (full_X, full_SC, full_Y): 
        - full_X/Y: [batch_size, target_len, features]
        - full_SC: [batch_size, sc_features]（静态，无 seq_dim）
    """
    batch_size = len(sparse_list_X)
    x_features = sparse_list_X[0].shape[-1]
    sc_features = sparse_list_SC[0].shape[-1]
    y_features = sparse_list_Y[0].shape[-1]  # 通常1
    
    full_X = torch.zeros(batch_size, target_len, x_features, device=device)
    full_SC = torch.zeros(batch_size, sc_features, device=device)  # 修改：无 seq_dim，只 [B, sc_features]
    full_Y = torch.zeros(batch_size, target_len, y_features, device=device)
    
    print(f"Step 3: Starting resampling for {batch_size} samples to length {target_len}...")  # 日志：步骤3 - 开始对{batch_size}个样本进行重采样至长度{target_len}
    print(f"X features: {x_features}, Y features: {y_features}")  # 日志：显示X和Y的特征维度数
    
    for i in range(batch_size):
        if (i + 1) % 100 == 0 or i == 0:  # 每100个样本或第一个打印一次，保持日志密度适中
            print(f"Resampling progress: sample {i+1}/{batch_size} ({(i+1)/batch_size*100:.1f}%)")  # 日志：重采样进度 - 当前样本{i+1}/{batch_size}，进度百分比
        
        sparse_X_i = sparse_list_X[i]  # [sparse_len, x_features]
        sparse_len = sparse_X_i.shape[0]
        if sparse_len == 0:
            continue  # 跳过空序列
        
        # 生成查询位置：linspace(0, sparse_len-1, target_len)，上采样
        query_pos = torch.linspace(0, sparse_len - 1, target_len, device=device)
        
        # 处理 X (多通道)
        for ch in range(x_features):
            # 假设你的 PCHIPRepara 支持 T [1, sparse_len, 1]
            sparse_T = sparse_X_i[:, ch].unsqueeze(0).unsqueeze(-1)  # [1, sparse_len, 1]
            repara = PCHIPRepara(sparse_T)
            
            for k in range(target_len):
                pos = query_pos[k]
                j = int(math.floor(pos.item()))  # 区间索引
                s_local = pos - j  # 局部 s ∈ [0,1]
                if j < 0:
                    full_X[i, k, ch] = sparse_X_i[0, ch]
                elif j >= sparse_len - 1:
                    full_X[i, k, ch] = sparse_X_i[-1, ch]
                else:
                    phi, _ = repara.forward(s_local, j)
                    full_X[i, k, ch] = phi.squeeze()
        
        # 处理 SC (静态，不插值，直接复制)
        sc_i = sparse_list_SC[i]  # [sc_features]
        full_SC[i] = sc_i  # 直接赋值 [sc_features]，不 repeat、不 unsqueeze
        
        # 处理 Y (假设单通道 [sparse_len, 1])
        sparse_Y_i = sparse_list_Y[i]  # [sparse_len, 1]
        sparse_T = sparse_Y_i.unsqueeze(0)  # [1, sparse_len, 1]
        repara = PCHIPRepara(sparse_T)
        for k in range(target_len):
            pos = query_pos[k]
            j = int(math.floor(pos.item()))
            s_local = pos - j
            if j < 0:
                full_Y[i, k, 0] = sparse_Y_i[0, 0]
            elif j >= sparse_len - 1:
                full_Y[i, k, 0] = sparse_Y_i[-1, 0]
            else:
                phi, _ = repara.forward(s_local, j)
                full_Y[i, k, 0] = phi.squeeze()
    
    print("Step 3: Resampling completed. Output shapes - full_X: {}, full_SC: {}, full_Y: {}".format(
        full_X.shape, full_SC.shape, full_Y.shape))  # 日志：步骤3 - 重采样完成，显示输出张量形状
    print("\n")  # 日志：空行分隔
    return full_X, full_SC, full_Y

def drop_random_in_range_points(
    X: torch.Tensor, 
    SC: torch.Tensor, 
    Y: torch.Tensor, 
    drop_range: Tuple[float, float] = (0.10, 0.14)
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    对于每个时间序列，随机选择 drop_range 内的比例，随机删除对应数量的点。
    SOC 静态数据 (SC) 不删除，保持完整。
    不跟踪原始位置，假设删除后序列有自己的均匀时间步。
    序列个数不变，X 和 Y 的长度缩短且参差不齐（每个序列独立比例），SC 长度不变。
    
    Args:
        X: 主要时序输入特征 [batch_size, seq_len, 4]
        SC: 静态辅助特征/电池参数 [batch_size, 4]，不参与删除
        Y: 预测目标 [batch_size, seq_len, 1]
        drop_range: 删除比例范围 (min, max)，如 (0.10, 0.14)
    
    Returns:
        (list_X, list_SC, list_Y): 
        - list_X: batch_size 个 Tensor 的列表，每个 [new_seq_len, 4]
        - list_SC: batch_size 个 Tensor 的列表，每个 [4]（不变）
        - list_Y: batch_size 个 Tensor 的列表，每个 [new_seq_len, 1]
    """
    batch_size, seq_len, *_ = X.shape
    device = X.device
    
    list_X, list_SC, list_Y = [], [], []
    print(f"Step 2: Starting random point dropping for {batch_size} samples (seq_len={seq_len}, drop_range={drop_range})...")  # 日志：步骤2 - 开始对{batch_size}个样本进行随机点删除（序列长度{seq_len}，删除比例范围{drop_range}）
    
    for i in range(batch_size):
        if (i + 1) % 100 == 0 or i == 0:  # 每100个样本或第一个打印一次
            print(f"Dropping progress: sample {i+1}/{batch_size} ({(i+1)/batch_size*100:.1f}%)")  # 日志：删除进度 - 当前样本{i+1}/{batch_size}，进度百分比
        
        drop_ratio = torch.rand(1, device=device).item() * (drop_range[1] - drop_range[0]) + drop_range[0]
        num_drop = max(1, int(drop_ratio * seq_len))  # 至少删1个
        
        # 随机选择删除位置（全局随机）
        drop_indices = torch.randperm(seq_len, device=device)[:num_drop]
        keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        keep_mask[drop_indices] = False
        
        # 提取保持的部分（顺序保持）
        list_X.append(X[i][keep_mask])
        list_SC.append(SC[i])  # SC 静态，不删除
        list_Y.append(Y[i][keep_mask])
    
    print(f"Step 2: Dropping completed. Average new seq_len: {np.mean([len(x) for x in list_X]):.1f}")  # 日志：步骤2 - 删除完成，显示平均新序列长度
    print("\n")  # 日志：空行分隔
    return list_X, list_SC, list_Y

def main(
    model_type: type,
    dataset: str,
    load_state_dict: str | None = None,
    save: str | None = None,
    max_epochs=5000,
):
    print(f"=== Starting main for model: {model_type.__name__}, dataset: {dataset}, max_epochs: {max_epochs} ===")  # 日志：开始主函数执行 - 模型{model_type.__name__}，数据集{dataset}，最大轮次{max_epochs}
    print(f"Using device: {device}")  # 日志：使用设备{device}
    
    # 构建数据集和数据加载器
    X, SC, Y = load_data(dataset=dataset, model_type=model_type, device=device)

    X_list, SC_list, Y_list = drop_random_in_range_points(X, SC, Y, drop_range=(0.10, 0.14))

    # 插值回统一长度
    target_len = X.shape[1]  # e.g., 50
    X, SC, Y = resample_with_pchip(X_list, SC_list, Y_list, target_len=target_len, device=device)

    # 现在 X 等是 [batch, 50, features]，继续划分 train/test
    choice = np.random.permutation(X.shape[0]).tolist()
    

    # 训练集和测试集的划分
    # 【重要】训练集和测试集分开加载
    choice = np.random.permutation(X.shape[0]).tolist()#.permutation（样本数）用来生成0到样本数-1的随机排列索引，再变成列表
    train_choice = choice[: int(X.shape[0] * 0.8)]#前百分之八十的索引是训练集索引
    test_choice = choice[int(X.shape[0] * 0.8) :]
    batch_size = 1024
    print(f"Step 5: Splitting data - Train samples: {len(train_choice)}, Test samples: {len(test_choice)}")  # 日志：步骤5 - 数据划分 - 训练样本数{len(train_choice)}，测试样本数{len(test_choice)}
    train_dataset = TensorDataset(X[train_choice], SC[train_choice], Y[train_choice])#TensorDataset用来将多个张量组合为一个数据集
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#数据加载器，传入数据和批次，设置打乱模式
    num_batches = len(train_dataloader)
    print(f"Train dataloader ready: {num_batches} batches of size {batch_size}")  # 日志：训练数据加载器准备完成 - {num_batches}个批次，每个大小{batch_size}

    # 定义网络模型
    model = create_model(model_type, device, X, SC, Y, dataset=dataset)
    if load_state_dict is not None:
        filepath = Path(f"save/{load_state_dict}/{type(model).__name__}.pt")#.__name__用来获取字符串名称
        if filepath.exists():
            print(f"Loading model state dict from {filepath}")  # 日志：从{filepath}加载模型状态字典
            model.load_state_dict(torch.load(filepath, weights_only=True))#torch.load加载参数，load_state_dict将加载的参数应用到模型
        else:
            print(f"{filepath} doesn't exist.")  # 日志：文件{filepath}不存在

    # 定义损失函数、评价指标和优化器
    loss_function = LogCoshLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    if load_state_dict is not None:#加载先前的优化器状态，支持断点续训
        filepath = Path(f"save/{load_state_dict}/{type(model).__name__}-Optimizer.pt")
        if filepath.exists():
            print(f"Loading optimizer state dict from {filepath}")  # 日志：从{filepath}加载优化器状态字典
            optimizer.load_state_dict(torch.load(filepath, weights_only=True))
        else:
            print(f"{filepath} doesn't exist.")  # 日志：文件{filepath}不存在

    # 记录器
    if save is not None:
        writer = SummaryWriter(log_dir=f"runs/{dataset + save}")#用于记录训练指标到TensorBoard中
        print(f"TensorBoard writer initialized for {save}")  # 日志：为{save}初始化TensorBoard记录器
    else:
        writer = Mock()
    saver = SaveAndEarlyStop(dataset=dataset, save=save)

    print("Step 6: Starting training loop...")  # 日志：步骤6 - 开始训练循环
    for epoch in range(max_epochs):
        print(f"\n--- Epoch {epoch + 1}/{max_epochs} ---")  # 日志：当前轮次{epoch + 1}/{max_epochs}开始
        # 训练模型
        model.train()
        train_loss = 0
        for batch_idx, (batch_x, batch_sc, batch_y) in enumerate(train_dataloader):
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:  # 每50个batch或第一个打印一次，密度适中（假设num_batches~1000/1024~1，会根据实际调整）
                print(f"  Training progress: batch {batch_idx + 1}/{num_batches} ({(batch_idx + 1)/num_batches*100:.1f}%)")  # 日志：训练进度 - 当前批次{batch_idx + 1}/{num_batches}，进度百分比
            
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            y_output = model(batch_x, batch_sc)
            # 计算损失
            loss: torch.Tensor = loss_function.forward(y_output, batch_y)
            # 反向传播
            loss.backward()
            # 更新权重
            optimizer.step()

            train_loss += loss

        train_loss: torch.Tensor = train_loss / len(train_dataloader)

        # 测试模型
        print("  Running evaluation...")  # 日志：运行评估
        model.eval()
        with torch.no_grad():
            y_output: torch.Tensor = model(X[test_choice], SC[test_choice])
            test_loss: float = loss_function.forward(y_output, Y[test_choice]).item() #.item()用来提取标量值
            #平均绝对误差
            mae: float = mean_absolute_error(
                y_output.flatten(), Y[test_choice].flatten()#.flatten()用来将张量降维为一维
            ).item()
            #平均绝对百分比误差
            mape: float = mean_absolute_percentage_error(
                y_output.flatten(), Y[test_choice].flatten()
            ).item()
            #均方根误差
            rmse: float = root_mean_squared_error(
                y_output.flatten(), Y[test_choice].flatten()
            ).item()

        # 保存信息和使用早停法
        # 5d是五位宽的整数格式，8.5e是指数形式的浮点数格式，8位宽，5位小数，8.5f也是8位5位小数的浮点数格式，8.4%是百分数格式，8位宽，4位小数
        print(
            f"  Epoch {epoch + 1:5d} | Train Loss: {train_loss:8.5e}, Test Loss: {test_loss:8.5e}, MAE: {mae:8.5f}, MAPE: {mape:8.4%}, RMSE: {rmse:8.5f}"
        )  # 日志：轮次{epoch + 1}结果 - 训练损失、测试损失、MAE、MAPE、RMSE
        # 记录训练和测试指标到 TensorBoard
        writer.add_scalar("Train Loss", train_loss.item(), epoch + 1)
        writer.add_scalar("Test Loss", test_loss, epoch + 1)
        writer.add_scalar("MAE", mae, epoch + 1)
        writer.add_scalar("MAPE", mape, epoch + 1)
        writer.add_scalar("RMSE", rmse, epoch + 1)
        if saver.stop(test_loss, model, optimizer):
            print(f"Early stopping triggered at epoch {epoch + 1}")  # 日志：早停机制在轮次{epoch + 1}触发
            break
    
    print(f"=== Training completed for {model_type.__name__} on {dataset} ===")  # 日志：{model_type.__name__}模型在{dataset}数据集上的训练完成


if __name__ == "__main__":
    for model_sel in [W_SOCNet]:
        for data_sel in ["ordered"]:
            main(
                model_sel,
                dataset=data_sel,
                # save=f"{model_sel.__name__}",  # 反注释这行后可以保存训练后的模型
                max_epochs=10,  # 按需要调大训练的 epoch 数目，实现训练效果
            )