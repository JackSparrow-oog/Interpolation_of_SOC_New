import torch
import pickle
from train_models import load_data, Copy_SOCNet, device

def drop_random_in_range_points(
    X: torch.Tensor, 
    SC: torch.Tensor, 
    Y: torch.Tensor, 
    drop_ratio: float = 0.12
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


X, SC, Y = load_data(filename="datasets/ordered/resample_SOC_50.pkl", device=device)

for drop_ratio in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    X_dropped, SC_dropped, Y_dropped = drop_random_in_range_points(X, SC, Y, drop_ratio=drop_ratio)
    data_dict = {"X": X_dropped.cpu().numpy(), "SC": SC_dropped.cpu().numpy(), "Y": Y_dropped.cpu().numpy()}
    with open(f"datasets/dropped/dropped_SOC_50_{drop_ratio}.pkl", "wb") as f:
        pickle.dump(data_dict, f)