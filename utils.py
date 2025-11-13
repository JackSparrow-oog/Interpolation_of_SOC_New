from pathlib import Path
import numpy as np
import torch
import pandas as pd
# from PIL import Image


color_list = [
    "#9e2e23",
    "#e2af42",
    "#407d53",
    "#418ab3",
    "#8e2961",
    "#827100",
    "#6b331a",
    "#161823",
    "#ad807a",
    "#003152",
    "#f86b1c",
    "#b0a4e3",
    "#4c000a",
    "#bddd22",
    "#ff3300",
    "#253d25",
    "#4f383e",
    "#b8ce8e",
    "#696969",
    "#000000",
]

marker_list = [
    "o",
    "v",
    "<",
    "^",
    ">",
    "s",
    "p",
    "*",
    "+",
    "x",
    "|",
    "_",
    "8",
    "h",
    "d",
    "1",
    "2",
    "3",
    "4",
    ".",
]

ls_list = ["solid", (0, (3, 1, 1, 1, 1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5))]
fsize = 20
width = 20
column_width = 3.5
page_width = 7.16


# def pdf2image(pdf_path: str, width_inches: float, dpi: int = 300):
#     from subprocess import run

#     pdf_path: Path = Path(pdf_path)
#     png_path = pdf_path.with_suffix(".png")
#     width_px = int(width_inches * dpi)

#     cmd = [
#         "magick",
#         "-quiet",
#         "-density",
#         str(dpi),
#         str(pdf_path),
#         "-resize",
#         str(width_px) + "x",
#         str(png_path),
#     ]
#     return run(cmd, capture_output=True)


def resize_image(image_path: str, width_inches: float, dpi: int = 300):
    from subprocess import run

    width_px = int(width_inches * dpi)
    cmd = [
        "magick",
        "-quiet",
        "-density",
        str(dpi),
        str(image_path),
        "-resize",
        str(width_px) + "x",
        str(image_path),
    ]
    return run(cmd, capture_output=True)


def mae_function(y, y_output):
    return np.mean(np.abs(y - y_output))


def mean_absolute_error(
    y_pred: torch.Tensor, y: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    """
    Compute the Mean Absolute Error (MAE) along a given dimension.

    Args:
        y_pred (torch.Tensor): Predicted values.
        y (torch.Tensor): Ground truth values.
        dim (int): Dimension along which to compute the error. Default is 0.

    Returns:
        torch.Tensor: MAE computed along the specified dimension.
    """
    return torch.mean(torch.abs(y_pred - y), dim=dim)


def mean_absolute_percentage_error(
    y_pred: torch.Tensor, y: torch.Tensor, dim: int = 0, eps: float = 1.17e-06
) -> torch.Tensor:
    """
    Compute the Mean Absolute Percentage Error (MAPE) along a given dimension.
    Adds a small epsilon to denominator for numerical stability.

    Args:
        y_pred (torch.Tensor): Predicted values.
        y (torch.Tensor): Ground truth values.
        dim (int): Dimension along which to compute the error. Default is 0.
        eps (float): Small constant to avoid division by zero. Default is 1e-8.

    Returns:
        torch.Tensor: MAPE (in float) computed along the specified dimension.
    """
    return torch.mean(
        torch.abs((y_pred - y)) / torch.clamp(torch.abs(y), min=eps), dim=dim
    )


def root_mean_squared_error(
    y_pred: torch.Tensor, y: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    """
    Compute the Root Mean Squared Error (RMSE) along a given dimension.

    Args:
        y_pred (torch.Tensor): Predicted values.
        y (torch.Tensor): Ground truth values.
        dim (int): Dimension along which to compute the error. Default is 0.

    Returns:
        torch.Tensor: RMSE computed along the specified dimension.
    """
    return torch.sqrt(torch.mean((y_pred - y) ** 2, dim=dim))


def rmse_function(y, y_output):
    return np.sqrt(np.mean((y - y_output) ** 2))


def mape_function(y, y_output):
    return np.mean(np.abs((y - y_output) / y))


def r2_function(y, y_output):
    y_mean = np.mean(y)
    return 1 - np.sum((y - y_output) ** 2) / np.sum((y - y_mean) ** 2)


def get_params_num(model: torch.nn.Module, return_num=True) -> int | str:
    # 度量模型
    # filter接收两个参数，第一个为函数，第二个为序列
    # 序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中——计算参数量
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params_num = sum((np.prod(p.shape) for p in model_parameters))
    if return_num:
        return params_num
    else:
        kilo_num = params_num // 1000
        mod_num = params_num % 1000
        return str(kilo_num) + " k " + str(mod_num)


def get_macs(model: torch.nn.Module, X: torch.Tensor, SC: torch.Tensor) -> int:
    if hasattr(model, "get_macs") and callable(getattr(model, "get_macs")):
        return model.get_macs(X, SC)
    else:
        from torchinfo import summary

        macs = summary(model, input_data=(X, SC)).total_mult_adds
        return macs


def get_flops(model: torch.nn.Module, X: torch.Tensor, SC: torch.Tensor) -> int:
    if hasattr(model, "get_flops") and callable(getattr(model, "get_flops")):
        return model.get_flops(X, SC)
    else:
        return 2 * get_macs(model, X, SC)


def get_model(filepath: str, device: torch.device) -> torch.nn.Module | None:
    filepath: Path = Path(filepath)
    if filepath.exists():
        model: torch.nn.Module = torch.load(filepath, weights_only=True)
        model = model.to(device)
        return model
    else:
        print(f"{filepath} doesn't exist. Return None.")
        return None


def get_model_with_state_dict(
    dirpath: str,
    X: torch.Tensor,
    SC: torch.Tensor,
    model_type: type,
    device: torch.device,
    *args,
    **kwargs,
) -> torch.nn.Module | None:
    filepath = Path(dirpath + f"/{model_type.__name__}.pt")
    if filepath.exists():
        model: torch.nn.Module = model_type(X, SC, *args, **kwargs)
        model = model.to(device)
        model.load_state_dict(torch.load(filepath, weights_only=True))
        # print(f"Params number: {get_params_num(model)}")
        # for p in model.named_parameters():
        #     print(f"{p[0]}: {p[1].data.tolist()}")
        return model
    else:
        print(f"{filepath} doesn't exist. Return None.")
        return None


def rolling_df(dataframe: pd.DataFrame, window: int, stride: int = 1) -> list:
    return [
        dataframe.iloc[i : i + window]
        for i in range(0, dataframe.shape[0] - window, stride)
    ]


def resample_df(
    data: pd.DataFrame,
    times: str = None,
    seq_len: int = None,
):
    # 检查每一列是否为数值型数据，如果有非数值型数据则报错
    if not all(np.issubdtype(data[col].dtype, np.number) for col in data.columns):
        raise ValueError("All columns in the data must be numeric.")

    if times is None:
        x = data.index.to_numpy()
    else:
        x = data[times].to_numpy()

    if seq_len is None:
        seq_len = len(data)
    x_rsp = np.linspace(x[0], x[-1], seq_len)

    # 对每一列数据进行插值，并生成新的数据
    data_interp = np.stack(
        [np.interp(x_rsp, x, data[col].to_numpy()) for col in data.columns], axis=1
    )

    # 将插值后的数据构造成一个新的DataFrame
    df_rsp = pd.DataFrame(
        data_interp, index=[i for i in range(len(x_rsp))], columns=data.columns
    )

    return df_rsp


def add_noise_with_snr(signal: np.ndarray, snr: float):
    # 添加信噪比指定的噪声
    PS = np.sum(signal**2) / len(signal)
    PN = PS / (10 ** (snr / 10))
    noise = np.random.randn(len(signal)) * np.sqrt(PN)
    signal_with_noise = signal + noise
    return signal_with_noise


def compute_snr(signal: np.ndarray, signal_with_noise: np.ndarray):
    # 计算序列的信噪比
    length = min(len(signal), len(signal_with_noise))
    noise = signal_with_noise[:length] - signal[:length]
    PS = np.sum(signal**2)
    PN = np.sum(noise**2)
    snr = 10 * np.log10((PS / PN))
    return snr


def generate_noised_series(
    signal: np.ndarray,  # 对应的DataFrame
    snr=70,  # 信噪比
    mu=None,  # 漂移距离平均，None表示不漂移
    sigma=1e-8,  # 漂移方差平均
):
    # 生成添加噪声和随机漂移的序列
    signal = add_noise_with_snr(signal, snr)
    if mu is not None:
        signal = signal + np.random.normal(mu, sigma, len(signal))
    return signal


def compute_sampen(signal: np.ndarray, m=2, r=None):
    if r is None:
        r = 0.2 * signal.std()
    N = len(signal)
    # Split time series and save all templates of length m
    xmi = np.array([signal[i : i + m] for i in range(N - m)])
    xmj = np.array([signal[i : i + m] for i in range(N - m + 1)])
    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])
    # Similar for computing A
    m += 1
    xm = np.array([signal[i : i + m] for i in range(N - m + 1)])
    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
    # Return SampEn
    return -np.log(A / B)


def df_to_tensor(
    df: pd.DataFrame,
    features: list,
    device: torch.device = torch.device("cuda:0")
    if torch.cuda.is_available()
    else torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    detect_diff: bool = False,
) -> torch.Tensor:
    df_np = df[features].values
    df_tensor = torch.tensor(df_np).to(device=device, dtype=dtype)
    if detect_diff:
        if (df[features[0]].diff() == 0).any():
            print(f"Pandas dataframe diffrentiation for {features[0]} is zero")
        elif (np.diff(df_np[:, 0]) == 0).any():
            print(f"Numpy array diffrentiation for {features[0]} is zero")
        elif (torch.diff(df_tensor[:, 0]) == 0).any():
            print(f"Torch tensor diffrentiation for {features[0]} is zero")
    return df_tensor


def tensor_to_np(data: torch.Tensor, squeeze: bool = True) -> np.ndarray:
    # 转换为ndarray
    if data.requires_grad:
        data = data.detach()
    if data.is_cuda:
        data = data.cpu()
    if squeeze:
        data = data.squeeze()

    return data.numpy()


def tensor_to_df(
    data: torch.Tensor, columns: list[str], squeeze: bool = True
) -> pd.DataFrame:
    data = tensor_to_np(data, squeeze)
    return pd.DataFrame(data, columns=columns)


# def resize_after_save(fname: str, new_width_inch: float, dpi=300):
#     # 打开图片
#     with Image.open(fname) as img:
#         # 获取图片的原始尺寸（像素）
#         original_width_px, original_height_px = img.size

#         # 计算当前宽度（像素）
#         new_width_px = new_width_inch * dpi

#         # 计算缩放比例
#         scale_factor = new_width_px / original_width_px

#         # 根据缩放比例计算新的高度（保持纵横比）
#         new_height_px = original_height_px * scale_factor

#         # 重新调整图像大小
#         resized_img = img.resize((int(new_width_px), int(new_height_px)), Image.LANCZOS)

#         # 保存图像，并设置 DPI
#         resized_img.save(fname, dpi=(dpi, dpi))
