
#### 1. **LogCosh Loss**（训练/测试损失）
- **数学公式**：
  \[
  \mathcal{L}_{\text{LogCosh}} = \frac{1}{N} \sum_{i=1}^{N} \left( \delta_i + \log(\cosh(2\delta_i)) - \log(2) \right)
  \]
  其中 \( \delta_i = \hat{y}_i - y_i \)，\( \cosh(x) = \frac{e^x + e^{-x}}{2} \)（双曲余弦函数）。
- **参数含义**：
  - \( \hat{y}_i \): 第 i 个预测 SOC 值（范围 [0,1]）。
  - \( y_i \): 第 i 个真实 SOC 值（范围 [0,1]）。
  - \( N \): 总样本点数（e.g., batch_size × seq_len）。
- **解释**：LogCosh 是平滑的 L1/L2 混合损失，对小误差线性、对大误差二次惩罚
#### 2. **MAE**（平均绝对误差）
- **数学公式**：
  \[
  \text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_i - y_i|
  \]
- **参数含义**：
  - \( \hat{y}_i \): 第 i 个预测 SOC 值。
  - \( y_i \): 第 i 个真实 SOC 值。
  - \( N \): 总样本点数。
- **解释**：简单绝对偏差平均，直观反映 SOC 预测精度（e.g., 0.03 = 平均 3% 误差）

#### 3. **MAPE (%)**（平均绝对百分比误差）
- **数学公式**：
  \[
  \text{MAPE} = \frac{100}{N} \sum_{i=1}^{N} \left| \frac{\hat{y}_i - y_i}{y_i + \epsilon} \right|
  \]
  其中 \( \epsilon = 10^{-8} \)（小常数，避免除零）。
- **参数含义**：
  - \( \hat{y}_i \): 第 i 个预测 SOC 值。
  - \( y_i \): 第 i 个真实 SOC 值。
  - \( N \): 总样本点数。
  - \( \epsilon \): 数值稳定性项（防 y_i=0 爆炸）。
- **解释**：相对误差百分比，强调比例准确。但对低 SOC（y_i ≈0）敏感，导致大值

#### 4. **RMSE**（均方根误差）
- **数学公式**：
  \[
  \text{RMSE} = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 }
  \]
- **参数含义**：
  - \( \hat{y}_i \): 第 i 个预测 SOC 值。
  - \( y_i \): 第 i 个真实 SOC 值。
  - \( N \): 总样本点数。
- **解释**：误差的“标准差”式度量，惩罚大偏差更重。

