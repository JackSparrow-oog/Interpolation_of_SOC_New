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

from utils import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        root_mean_squared_error,
    )

from models import (
    W_SOCNet,
    SOCNet,
    Copy_SOCNet,
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
    if filename is None:
        if model_type in [SOCNet]:
            filename = f"datasets/{dataset}/random_SOC_50.pkl"
        # 修改
        elif model_type in [Copy_SOCNet]:
            filename = f"datasets/{dataset}/resample_SOC_50.pkl"
        else:
            raise NotImplementedError

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

    return X, SC, Y


def create_model(
    model_type: type,
    device: torch.device,
    X: torch.Tensor,
    SC: torch.Tensor,
    Y: torch.Tensor,
    dataset: str = "ordered",
) -> torch.nn.Module:
    
    model = model_type(X=X, SC=SC, dataset=dataset).to(device)

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


def drop_random_in_range_points(
    X: torch.Tensor, 
    SC: torch.Tensor, 
    Y: torch.Tensor, 
    drop_ratio: float = 0.12
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对于每个时间序列，随机删除 drop_ratio 比例的点。
    SOC 静态数据 (SC) 不删除，保持完整。
    不跟踪原始位置，假设删除后序列有自己的均匀时间步。
    序列个数不变，X 和 Y 的长度缩短（固定比例，所有序列相同新长度），SC 长度不变。
    
    Args:
        X: 主要时序输入特征 [batch_size, seq_len, 4]
        SC: 静态辅助特征/电池参数 [batch_size, 4]，不参与删除
        Y: 预测目标 [batch_size, seq_len, 1]
        drop_ratio: 删除比例，如 0.12 (12%)
    
    Returns:
        (X_padded: torch.Tensor, SC: torch.Tensor, Y_padded: torch.Tensor): 
        - X_padded: [batch_size, new_seq_len, 4]
        - SC: [batch_size, 4]（不变）
        - Y_padded: [batch_size, new_seq_len, 1]
        其中 new_seq_len = seq_len - num_drop，所有序列长度相同。
    """
    batch_size, seq_len, *_ = X.shape
    device = X.device
    
    num_drop = max(1, int(drop_ratio * seq_len))  # 至少删1个
    new_seq_len = seq_len - num_drop
    
    list_X, list_Y = [], []
    for i in range(batch_size):
        # 随机选择删除位置（全局随机）
        drop_indices = torch.randperm(seq_len, device=device)[:num_drop]
        keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        keep_mask[drop_indices] = False
        
        # 提取保持的部分（顺序保持）
        list_X.append(X[i][keep_mask])
        list_Y.append(Y[i][keep_mask])
    
    X_out = torch.stack(list_X, dim=0)
    Y_out = torch.stack(list_Y, dim=0)
    
    return X_out, SC, Y_out

def main(
    model_type: type,
    dataset: str,
    load_state_dict: str | None = None,
    save: str | None = None,
    max_epochs=5000,
):
    # 构建数据集和数据加载器
    X, SC, Y = load_data(dataset=dataset, model_type=model_type, device=device)

    X, SC, Y = drop_random_in_range_points(X, SC, Y, drop_ratio=0.15)

    # 训练集和测试集的划分
    # 【重要】训练集和测试集分开加载
    choice = np.random.permutation(X.shape[0]).tolist()#.permutation（样本数）用来生成0到样本数-1的随机排列索引，再变成列表
    train_choice = choice[: int(X.shape[0] * 0.8)]#前百分之八十的索引是训练集索引
    test_choice = choice[int(X.shape[0] * 0.8) :]
    batch_size = 1024
    train_dataset = TensorDataset(X[train_choice], SC[train_choice], Y[train_choice])#TensorDataset用来将多个张量组合为一个数据集
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#数据加载器，传入数据和批次，设置打乱模式

    # 定义网络模型
    model = create_model(model_type, device, X, SC, Y, dataset=dataset)
    if load_state_dict is not None:
        filepath = Path(f"save/{load_state_dict}/{type(model).__name__}.pt")#.__name__用来获取字符串名称
        if filepath.exists():
            model.load_state_dict(torch.load(filepath, weights_only=True))#torch.load加载参数，load_state_dict将加载的参数应用到模型
        else:
            print(f"{filepath} doesn't exist.")

    # 定义损失函数、评价指标和优化器
    loss_function = LogCoshLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    if load_state_dict is not None:#加载先前的优化器状态，支持断点续训
        filepath = Path(f"save/{load_state_dict}/{type(model).__name__}-Optimizer.pt")
        if filepath.exists():
            optimizer.load_state_dict(torch.load(filepath, weights_only=True))
        else:
            print(f"{filepath} doesn't exist.")

    # 记录器
    if save is not None:
        writer = SummaryWriter(log_dir=f"runs/{dataset + save}")#用于记录训练指标到TensorBoard中
    else:
        writer = Mock()
    saver = SaveAndEarlyStop(dataset=dataset, save=save)

    for epoch in range(max_epochs):
        # 训练模型
        model.train()
        train_loss = 0
        for batch_x, batch_sc, batch_y in train_dataloader:
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
            f"Epoch: {epoch + 1:5d} | Train Loss: {train_loss:8.5e}, Test Loss: {test_loss:8.5e}, MAE: {mae:8.5f}, MAPE: {mape:8.4%}, RMSE: {rmse:8.5f}"
        )
        # 记录训练和测试指标到 TensorBoard
        writer.add_scalar("Train Loss", train_loss.item(), epoch + 1)
        writer.add_scalar("Test Loss", test_loss, epoch + 1)
        writer.add_scalar("MAE", mae, epoch + 1)
        writer.add_scalar("MAPE", mape, epoch + 1)
        writer.add_scalar("RMSE", rmse, epoch + 1)
        if saver.stop(test_loss, model, optimizer):
            break



if __name__ == "__main__":
    for model_sel in [W_SOCNet]:
        for data_sel in ["ordered"]:
            main(
                model_sel,
                dataset=data_sel,
                # save=f"{model_sel.__name__}",  # 反注释这行后可以保存训练后的模型
                max_epochs=10,  # 按需要调大训练的 epoch 数目，实现训练效果
            )
