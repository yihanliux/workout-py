import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt  # 导入 ruptures 库
import json

def load_json_data(filename):
    """读取 JSON 文件并解析数据"""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"错误: 无法读取 {filename}，请检查文件路径或格式。")
        return []
    
    frames = data.get("frames", [])  # 读取帧数据列表
    body_height = [frame.get("body_height") for frame in frames] 
    head_y = [frame.get("head_y") for frame in frames]

    return body_height, head_y

# ✅ 读取数据
filename = "output_data8.json"
body_height, head_y = load_json_data(filename)

# ✅ 数据转换为 NumPy 数组
signal = np.array(head_y, dtype=float)

# ✅ 处理 NaN/Inf 值，防止计算错误
signal = np.nan_to_num(signal, nan=np.nanmean(signal), posinf=np.max(signal), neginf=np.min(signal))

# ✅ 2. 选择一个变化点检测算法 (Pelt 方法)
algo = rpt.Pelt(model="l2").fit(signal)

# ✅ 3. 设定惩罚参数（越大越不敏感）
breakpoints = algo.predict(pen=1)

# ✅ 4. 绘制结果
plt.figure(figsize=(10, 5))
plt.plot(signal, label="Signal")
for bp in breakpoints[:-1]:  # 突变点（去掉最后一个终点）
    plt.axvline(bp, color="red", linestyle="--", label="Change Point")
plt.legend()
plt.title("Change Point Detection using Ruptures")
plt.show()
