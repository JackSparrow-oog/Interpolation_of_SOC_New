import pickle
import numpy as np
import pandas as pd

# data['X']: 主要时序输入特征 (X.shape: [样本数, 序列长度, 4])
# 维度含义:
# X[..., 0:1] -> 时间 (t)
# X[..., 1:2] -> 电流 (I)
# X[..., 2:3] -> 温度 (T)
# X[..., 3:4] -> 电压 (U)


# data['SC']: 静态辅助特征/电池参数 (SC.shape: [样本数, 4])
# 维度含义:
# SC[..., 0:1] -> 额定容量 (Q)
# SC[..., 1:2] -> 初始库仑效率 (eta_0)
# SC[..., 2:3] -> 欧姆内阻 (R)
# SC[..., 3:4] -> 初始 SOC 基准值 (SOC_init_ref)


# data['Y']: 预测目标 (Y.shape: [样本数, 序列长度, 1])
# 维度含义:
# Y[..., 0:1] -> 真实荷电状态 (True State of Charge, SOC)


# 可以调用的查阅函数 (display_data_sample)
# display_data_sample('X', X_data, slice_count=3) 
# display_data_sample('SC', SC_data, slice_count=5)
# display_data_sample('Y', Y_data, slice_count=5)

# 文件路径，请替换为你的实际文件路径
#file_path = './datasets/ordered/resample_SOC_50.pkl'
#file_path = './datasets/preprocessed/ordered/W_SOCNet_preprocessed.pkl'
file_path = './datasets/preprocessed/ordered/SOCNet_preprocessed.pkl'

def display_data_sample(name, data_object, slice_count=5):
    """
    一个通用的函数，用于打印数据对象的类型、形状/长度和前 n 个元素。
    """
    print("-" * 60)
    print(f"【 键 '{name}'：数据概览 】")
    print(f"类型: {type(data_object)}")

    sample_to_print = None
    shape_info = ""

    if isinstance(data_object, np.ndarray):
        # 1. 如果是 NumPy 数组
        shape_info = f"形状: {data_object.shape}"
        if data_object.ndim >= 1 and data_object.shape[0] > 0:
            # 打印第一维的前 n 个元素
            sample_to_print = data_object[:slice_count]
        else:
            sample_to_print = "（数组为空或维度异常）"

    elif isinstance(data_object, (list, tuple)):
        # 2. 如果是列表或元组
        shape_info = f"总长度: {len(data_object)}"
        if len(data_object) > 0:
            sample_to_print = data_object[:slice_count]
        else:
            sample_to_print = "（列表/元组为空）"

    elif isinstance(data_object, pd.DataFrame):
        # 3. 如果是 Pandas DataFrame
        shape_info = f"形状: {data_object.shape}"
        sample_to_print = data_object.head(slice_count)
        
    elif isinstance(data_object, dict):
        # 4. 如果字典的值本身又是字典（递归检查）
        shape_info = f"包含键: {list(data_object.keys())}"
        sample_to_print = "（请手动检查嵌套字典的内容）"

    else:
        # 5. 对于其他标量类型，直接打印
        shape_info = "（标量或其他非集合类型）"
        sample_to_print = data_object


    print(shape_info)
    print(f"\n--- 示例 (前 {slice_count} 个元素/样本/行): ---")
    print(sample_to_print)


# ==========================================================
# 主程序
# ==========================================================
try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        print(f"⚠️ 警告：数据集不是字典类型 ({type(data)})，将直接打印前 5 个元素。")
        display_data_sample("数据集根对象", data)
    
    elif all(key in data for key in ['X', 'SC', 'Y']):
        print("=" * 60)
        print("✅ 数据集加载成功，正在逐一检查 'X', 'SC', 'Y' 键...")
        
        # 统一调用函数来查阅三个键的内容
        display_data_sample('X', data['X'], slice_count=3)  # X数据量大，只取前 3 个样本
        display_data_sample('SC', data['SC'], slice_count=3)
        display_data_sample('Y', data['Y'], slice_count=3)
        
        print("-" * 60)
        
    else:
        print("❌ 错误：字典中缺少 'X', 'SC', 或 'Y' 键。")
        print(f"实际键: {list(data.keys())}")


except FileNotFoundError:
    print(f"❌ 错误：文件未找到。请检查路径：{file_path} 和当前工作目录。")
except Exception as e:
    print(f"❌ 发生未知错误: {e}")