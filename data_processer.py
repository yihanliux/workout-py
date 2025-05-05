import json
import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import statsmodels.api as sm
from PIL import Image
import os
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.colors import to_rgb, to_hex
from collections import defaultdict
from matplotlib.patches import FancyBboxPatch



def load_json_data(filename):
    """
    读取 JSON 文件并解析数据。
    
    参数:
        filename (str): JSON 文件的路径。
    
    返回:
        tuple: 包含以下数据的元组：
            - fps (int): 视频的帧率，默认为 30。
            - people_counts (list): 每一帧检测到的人数列表。
            - body_height (list): 每一帧检测到的身体高度列表。
            - orientation (list): 每一帧检测到的方向数据列表。
            - head_y (list): 每一帧检测到的头部 Y 坐标列表。
    """
    try:
        # 尝试打开 JSON 文件并解析内容
        with open(filename, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # 处理文件不存在或 JSON 解析失败的情况
        print(f"错误: 无法读取 {filename}，请检查文件路径或格式。")
        return 0, [], [], [], [], []
    
    # 解析 JSON 数据
    fps = data.get("fps", 30)  # 读取 fps，默认为 30
    frames = data.get("frames", [])  # 读取帧数据列表，如果不存在则默认为空列表
    
    # 提取每一帧的数据
    people_counts = [frame.get("people_count") for frame in frames]
    body_height = [frame.get("body_height") for frame in frames]  
    orientation = [frame.get("orientation") for frame in frames] 
    head_y = [frame.get("head_y") for frame in frames]
    
    # 打印视频帧信息
    print("视频总共有", len(people_counts), "帧，帧率:", fps, "FPS")
    
    # 返回解析后的数据
    return fps, people_counts, body_height, orientation, head_y

def smooth_stable_data(people_counts, orientation, window_size=10, consensus_ratio=0.8):
    """
    平滑数据，移除噪音，使 `people_counts` 和 `orientation` 更稳定。
    
    该方法使用滑动窗口计算最常见值，并在比例达到 `consensus_ratio` 时替换当前值，
    以减少噪声的影响，使数据更平滑。
    
    参数:
        people_counts (list[int]): 每一帧检测到的人数数据。
        orientation (list[str]): 每一帧的面部朝向信息。
        motion_states (list[str]): 每一帧的运动状态 ('static' 或 'dynamic')。
        window_size (int): 滑动窗口的大小，决定平滑时考虑的帧数（默认 10）。
        consensus_ratio (float): 认定最常见值的比例，若达到该比例则采用最常见值（默认 0.8）。

    返回:
        tuple: 包含平滑后的数据：
            - filtered_people_counts (list[int]): 平滑后的人数数据。
            - filtered_orientation (list[str]): 平滑后的面部朝向数据。
            - filtered_motion_states (list[str]): 平滑后的运动状态数据。
    """
    
    # 复制原始数据，避免修改输入列表
    filtered_people_counts = people_counts[:]
    filtered_orientation = orientation[:]

    # 遍历所有帧数据
    for i in range(len(people_counts)):
        # 定义滑动窗口的范围
        start, end = max(0, i - window_size), min(len(people_counts), i + window_size)

        # 计算滑动窗口内的最常见值
        most_common_people = max(set(people_counts[start:end]), key=people_counts[start:end].count)
        most_common_orientation = max(set(orientation[start:end]), key=orientation[start:end].count)

        # 计算最常见值的占比
        people_consensus = people_counts[start:end].count(most_common_people) / (end - start)
        orientation_consensus = orientation[start:end].count(most_common_orientation) / (end - start)

        # 如果最常见值的比例超过 `consensus_ratio`，则采用它，否则保持原值
        filtered_people_counts[i] = most_common_people if people_consensus >= consensus_ratio else people_counts[i]
        filtered_orientation[i] = most_common_orientation if orientation_consensus >= consensus_ratio else orientation[i]

    return filtered_people_counts, filtered_orientation

def first_orientation_segments(orientation, body_height, head_y, fps):

    orient_segments = []
    current_orient, start_frame = None, None

    # 预处理，将 None 变成 'Invalid'
    orientation = ['Invalid' if orientation is None else orient for orient in orientation]

    for i in range(len(body_height)):
        if body_height[i] is None or head_y[i] is None:
            orientation[i] = "Invalid"  # 修改 orientation

    # 遍历每一帧的姿态方向，分割不同的片段
    for i, orient in enumerate(orientation):
        if current_orient is None:
            current_orient, start_frame = orient, i
        elif orient != current_orient:
            end_frame = i - 1
            duration = end_frame - start_frame + 1
            orient_segments.append({
                "orient": current_orient,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_sec": duration / fps,
                "duration_frames": duration
            })
            current_orient, start_frame = orient, i

    # 处理最后一个片段
    if current_orient is not None:
        end_frame = len(orientation) - 1
        duration = end_frame - start_frame + 1
        orient_segments.append({
            "orient": current_orient,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_sec": duration / fps,
            "duration_frames": duration
        })

    return orient_segments

def filter_invalid_orientation_segments(orient_segments, orientation, body_height, head_y, fps=30, min_duration_sec=3, max_duration_sec=90):
    
    min_duration_frames = fps * min_duration_sec
    max_duration_frames = fps * max_duration_sec
    total_frames = len(orientation)
    
    # 计算前 10% 和后 10% 的帧范围
    first_10_percent = min(int(0.1 * total_frames), max_duration_sec)
    last_10_percent = max(int(0.9 * total_frames), total_frames - max_duration_frames)
    
    # 找出所有 "Invalid" 片段
    long_invalid_segments = []
    first_invalid_in_10_percent = None
    last_invalid_in_90_percent = None

    for segment in orient_segments:
        if segment["orient"] == "Invalid":
            # 检查是否超过 min_duration_frames
            if segment["duration_frames"] > min_duration_frames:
                long_invalid_segments.append(segment)

            # 检查是否在前10%
            if segment["end_frame"] < first_10_percent:
                first_invalid_in_10_percent = segment  # 持续更新，找到最后一个
            # 检查是否在后10%
            elif segment["start_frame"] >= last_10_percent and last_invalid_in_90_percent is None:
                last_invalid_in_90_percent = segment  # 仅找到第一个就停

    # 更新 orientation_segments 使得前后片段变为 "Invalid"
    new_orient_segments = []
    invalid_mode = False  # 这个标志决定是否将后续片段设为 "Invalid"

    for segment in orient_segments:
        if first_invalid_in_10_percent and segment["end_frame"] <= first_invalid_in_10_percent["end_frame"]:
            invalid_mode = True  # 触发 Invalid 模式

        if last_invalid_in_90_percent and segment["start_frame"] >= last_invalid_in_90_percent["start_frame"]:
            invalid_mode = True  # 触发 Invalid 模式
        
        if invalid_mode:
            # 将当前片段变为 Invalid
            new_segment = segment.copy()
            new_segment["orient"] = "Invalid"
            new_orient_segments.append(new_segment)
        else:
            new_orient_segments.append(segment)


    # 4️⃣ 删除超过 1 秒的 "Invalid" 片段
    frames_to_remove = set()
    for segment in orient_segments:
        if segment["orient"] == "Invalid":
            frames_to_remove.update(range(segment["start_frame"], segment["end_frame"] + 1))
    
    updated_orient_segments = []
    for segment in orient_segments:
        if segment["orient"] == "Invalid" :
            print(f"Deleted Invalid segment: Start {segment['start_frame']}, End {segment['end_frame']}, Duration {segment['duration_frames']} frames ({segment['duration_sec']} sec)")
        else:
            updated_orient_segments.append(segment)  # 只保留未被删除的片段

    frames_to_keep = set(range(total_frames)) - frames_to_remove
    updated_orientation = [orientation[i] for i in frames_to_keep]
    updated_body_height = [body_height[i] for i in frames_to_keep]
    updated_head_y = [head_y[i] for i in frames_to_keep]
    
    if updated_orient_segments:
        new_segments = []
        prev_end_frame = 0
        
        for seg in updated_orient_segments:
            duration = seg["end_frame"] - seg["start_frame"]
            seg["start_frame"] = prev_end_frame
            seg["end_frame"] = prev_end_frame + duration
            prev_end_frame = seg["end_frame"] + 1
            new_segments.append(seg)
        
        updated_orient_segments = new_segments

    return updated_orient_segments, updated_orientation, updated_body_height, updated_head_y
    
def compute_adaptive_threshold(data, method="std", k=2):
    """
    计算数据的自适应阈值，用于检测异常值。

    该函数根据不同的统计方法计算数据的动态阈值：
    - method="std"  -> 使用标准差计算阈值：threshold = k * std
    - method="mad"  -> 使用平均绝对偏差（MAD）计算阈值
    - method="iqr"  -> 使用 IQR（四分位距）计算阈值

    参数：
        data (list[float]): 需要计算阈值的数据列表，例如 `body_height`。
        method (str): 选择计算方法，可选值为 'std', 'mad', 'iqr' (默认 "std")。
        k (float): 乘法因子，用于控制阈值的灵敏度 (默认 2)。

    返回：
        float: 计算出的自适应阈值。

    异常：
        ValueError: 如果提供的 `method` 不支持，则抛出异常。
    """
    
    # 将数据转换为 NumPy 数组，确保支持数学计算
    data = np.array(data)
    data = np.array([x if x is not None else 0 for x in data])

    if method == "std":
        # 使用标准差计算阈值
        threshold = k * np.std(data)
    
    elif method == "mad":
        # 计算中位数
        median = np.median(data)
        # 计算平均绝对偏差（MAD）
        mad = np.median(np.abs(data - median))
        threshold = k * mad

    elif method == "iqr":
        # 计算第一四分位数（Q1）和第三四分位数（Q3）
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        # 计算四分位距（IQR）
        iqr = Q3 - Q1
        threshold = k * iqr

    else:
        # 如果传入的 `method` 参数不合法，则抛出错误
        raise ValueError("Unsupported method. Choose from ['std', 'mad', 'iqr']")
    
    return threshold

def detect_change_points(data, percentile=95, window_size=3, visualize=False):
    """
    检测时间序列数据中的突变点（突增或突降）。

    主要逻辑：
    1. 计算数据的变化率（取绝对值差分）。
    2. 根据 `percentile` 计算突变阈值。
    3. 使用 `find_peaks` 方法检测超过阈值的突变点。
    4. 过滤掉短暂突变点，确保变化后维持高值一段时间。
    5. 如果 `visualize=True`，则绘制可视化图表。

    参数:
        data (array-like): 输入的时间序列数据 (如 body_height)。
        percentile (float): 变化点的阈值（分位数），默认 95（取前 5% 最大变化点）。
        window_size (int): 用于过滤短暂突变点的窗口大小（默认 3）。
        visualize (bool): 是否可视化检测结果，默认 True。

    返回:
        list[int]: 变化点的索引列表。

    """

    # 确保数据是 numpy 数组
    data = np.array(data, dtype=float)

    # 计算变化率（取绝对值的差分）
    diff = np.abs(np.diff(data))

    # 计算突变点阈值（取前 `percentile%` 最大变化值）
    threshold = np.percentile(diff, percentile)

    # 检测变化点（峰值点）
    jump_points, _ = find_peaks(diff, height=threshold)

    # 过滤掉短暂突变点
    change_points = [
        p for p in jump_points if np.mean(data[p:p + window_size]) > np.mean(data) + np.std(data)
    ]

    # 可视化检测结果
    if visualize:
        plt.figure(figsize=(12, 5))
        plt.plot(data, label="Data", color="blue")
        plt.scatter(change_points, data[change_points], color='red', label="Change Points", zorder=3)
        plt.legend()
        plt.title("Detected Change Points")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.show()

    return change_points

def remove_large_height_changes(change_points, orientation_segments, orientation, body_height, head_y, fps=30, max_duration_sec=90):
    """
    处理 change_points，删除异常突变的片段。

    主要逻辑：
    1. **计算自适应阈值 threshold**：
       - 通过 `compute_adaptive_threshold()` 计算 body_height 的阈值，用于识别异常变化。
    2. **遍历所有 change_points**：
       - 在 `max_duration_frames` 内，寻找最后一个变化点。
       - 计算该区间的 body_height 均值，并与全局均值比较。
    3. **判断是否删除该区间**：
       - 若 body_height 的突变区域与其他区域的均值差值大于 threshold，则删除该片段。
       - 若该突变发生在前 10% 或后 10% 的视频时间，则直接删除对应区域。
    4. **同步更新 orientation_segments、orientation、body_height 和 head_y**：
       - 删除相关帧，并重新计算有效的片段索引。

    参数：
        change_points (list[int]): 检测到的突变点索引。
        orientation_segments (list[dict]): 姿态信息片段。
        orientation (list[str]): 每帧的姿态信息。
        body_height (list[float]): 每帧的身体高度数据。
        head_y (list[float]): 每帧的头部 Y 坐标数据。
        fps (int): 视频帧率，默认 30。
        method (str): 计算 threshold 的方法 ["std", "mad", "iqr"]，默认 "std"。
        k (float): 计算 threshold 时的乘法因子，默认 2。
        max_duration_sec (int): 突变最大持续时间（秒），默认 90。

    返回：
        tuple:
            - updated_orientation_segments (list[dict]): 更新后的姿态片段数据。
            - updated_orientation (list[str]): 过滤后的姿态信息。
            - updated_body_height (list[float]): 过滤后的身体高度数据。
            - updated_head_y (list[float]): 过滤后的头部 Y 轴数据。
    """

    total_frames = len(body_height)
    max_duration_frames = fps * max_duration_sec

    # 计算自适应阈值 threshold
    threshold = compute_adaptive_threshold(body_height)

    frames_to_remove = set()  # 记录需要删除的帧索引

    # 遍历所有突变点，确定需要删除的片段
    for i, cp in enumerate(change_points):
        # 寻找 max_duration_frames 内的最后一个突变点
        future_changes = [p for p in change_points if p > cp and p <= cp + max_duration_frames]

        end_idx = max(future_changes) if future_changes else cp  # 选取最后的突变点
        start_idx = cp  # 记录突变区间起点

        # 计算该区间 body_height 的均值
        region_mean = np.mean(body_height[start_idx:end_idx])

        # 计算全局 body_height 均值（排除该区间）
        other_frames = [i for i in range(total_frames) if i < start_idx or i > end_idx]
        if other_frames:
            other_mean = np.mean([body_height[i] for i in other_frames])
        else:
            other_mean = np.mean(body_height)  # 仅有该区间存在，取全局均值

        # 计算 body_height 的变化幅度
        height_diff = abs(region_mean - other_mean)

        # 处理前 10% 和后 10% 的变化
        has_early_change = any(p <= max_duration_frames for p in [start_idx, end_idx])
        has_late_change = any(p >= total_frames - max_duration_frames for p in [start_idx, end_idx])

        if height_diff > threshold:
            if has_early_change:
                # 删除前 10% 内的突变区域
                frames_to_remove.update(range(0, end_idx + 1))
            elif has_late_change:
                # 删除后 10% 内的突变区域
                frames_to_remove.update(range(start_idx, total_frames))
            else:
                # 删除该突变区间
                frames_to_remove.update(range(start_idx, end_idx + 1))

    # 过滤 orientation_segments，并同步删除相应帧的数据
    new_frames_to_remove = frames_to_remove.copy()

    updated_orientation_segments = []
    for seg in orientation_segments:
        # 如果该片段的帧被标记为删除，则丢弃
        if not any(frame in frames_to_remove for frame in range(seg["start_frame"], seg["end_frame"] + 1)):
            updated_orientation_segments.append(seg)
        else:
            new_frames_to_remove.update(range(seg["start_frame"], seg["end_frame"] + 1))

    # 重新计算有效帧索引
    frames_to_keep = set(range(total_frames)) - new_frames_to_remove
    updated_orientation = [orientation[i] for i in frames_to_keep]
    updated_body_height = [body_height[i] for i in frames_to_keep]
    updated_head_y = [head_y[i] for i in frames_to_keep]

    # 重新计算 orientation_segments，使索引保持连续
    if updated_orientation_segments:
        new_segments = []
        prev_end_frame = 0

        for seg in updated_orientation_segments:
            duration = seg["end_frame"] - seg["start_frame"]
            seg["start_frame"] = prev_end_frame
            seg["end_frame"] = prev_end_frame + duration
            prev_end_frame = seg["end_frame"] + 1
            new_segments.append(seg)

        updated_orientation_segments = new_segments

    return updated_orientation_segments, updated_orientation, updated_body_height, updated_head_y

def merge_alternating_orients(orientation_segments, fps=30, max_swaps=15, min_duration_sec=3):
    """
    合并短时交替的姿态片段，以减少抖动和误判。

    主要逻辑：
    1. **合并相邻的相同姿势片段**：
       - 如果当前片段的方向与上一个相同，则合并它们，更新 `end_frame` 和 `duration`。
    2. **遍历姿态片段，合并交替变换的短时片段**：
       - 记录 `swap_count`，跟踪短时交替出现的次数。
       - 如果 `swap_count` 超过 `max_swaps`，则合并该区域，并创建一个新的片段。
       - 否则，保持原状，防止过度合并。
    3. **处理未被合并的片段**：
       - 追加最后一个未处理的片段，防止数据丢失。

    参数：
        orientation_segments (list[dict]): 姿态信息片段，每个片段是一个字典。
        fps (int): 视频的帧率（默认 30）。
        max_swaps (int): 允许的最大交替切换次数，超过此值则进行合并（默认 15）。
        min_duration_sec (int): 最小持续时间（秒），小于该值的片段将被合并（默认 3 秒）。

    返回：
        list[dict]: 处理后的 `orientation_segments` 片段列表。
    """

    # **合并相邻的相同姿势片段**
    merged_segments = []
    for segment in orientation_segments:
        if merged_segments and merged_segments[-1]["orient"] == segment["orient"]:
            # 如果当前片段与上一个片段方向相同，则合并
            merged_segments[-1]["end_frame"] = segment["end_frame"]
            merged_segments[-1]["duration_sec"] = (merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1) / fps
            merged_segments[-1]["duration_frames"] = merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1
        else:
            merged_segments.append(segment)
    
    # 更新 `orientation_segments`
    orientation_segments = merged_segments 

    # 计算最小持续时间（转换为帧数）
    min_duration_frames = fps * min_duration_sec
    result = []
    i = 0  # 迭代索引
    
    while i < len(orientation_segments) - 1:
        current_orient = orientation_segments[i]['orient']
        current_frame = orientation_segments[i]['duration_frames']
        if current_frame < min_duration_frames:
            swap_count = 0  # 记录交替变换次数
            combined_segments = [orientation_segments[i]]  # 存储待合并片段
            next_orient = orientation_segments[i + 1]['orient']
            next_frame = orientation_segments[i + 1]['duration_frames']
            j = i + 1  # 用于收集后续片段的索引
            
            # **如果当前片段时长较短，且下一个片段的方向不同，则尝试合并**
            if current_orient != next_orient and next_frame < min_duration_frames:
                combined_segments.append(orientation_segments[j])
                j += 1
                
                # **继续查找更多的短时交替片段**
                while j < len(orientation_segments):
                    third_orient = orientation_segments[j]['orient']
                    third_segment = orientation_segments[j]
                    
                    # **如果第三个片段的方向属于 (current_orient, next_orient)，且短时交替，则继续合并**
                    if (third_orient in [current_orient, next_orient] and
                        third_orient != combined_segments[-1]['orient'] and
                        third_segment['duration_frames'] < min_duration_frames):
                        swap_count += 1  # 记录交替切换次数
                        combined_segments.append(third_segment)
                        j += 1  # 继续遍历
                    else:
                        break  # 规则被破坏，停止合并
                
                # **如果交替切换次数超过 `max_swaps`，合并这些片段**
                if swap_count > max_swaps:
                    if len(current_orient) > len(next_orient):
                        combined_orient = f"{current_orient}-{next_orient}"
                    else: 
                        combined_orient = f"{next_orient}-{current_orient}"  # 组合方向
                    merged_segment = {
                        'orient': combined_orient,
                        'start_frame': combined_segments[0]['start_frame'],
                        'end_frame': combined_segments[-1]['end_frame'],
                        'duration_sec': sum(seg['duration_sec'] for seg in combined_segments),
                        'duration_frames': sum(seg['duration_frames'] for seg in combined_segments)
                    }
                    result.append(merged_segment)  # 存储合并后的片段
                    print(merged_segment)  # 打印合并信息（可选）
                else:
                    result.extend(combined_segments)  # 交替次数较少，不合并
                
                # **跳到下一个未处理的片段**
                i = j  
            else:
                # **当前片段不符合合并条件，直接添加到结果**
                result.append(orientation_segments[i])
                i += 1  # 继续主循环遍历
        else:
            # **当前片段不符合合并条件，直接添加到结果**
                result.append(orientation_segments[i])
                i += 1  # 继续主循环遍历
    
    # **追加最后一个 segment，如果它未被处理**
    if i == len(orientation_segments) - 1:
        result.append(orientation_segments[i])
    
    return result

def merge_orientation_segments(orientation_segments, orientation, body_height, head_y, fps=30, min_duration_sec=3, max_duration_sec=15):
    """
    合并短时的姿态片段，去除不稳定的片段，并优化方向数据。

    主要逻辑：
    1. **合并短片段**：
       - 如果片段 `duration_frames < min_duration_frames`，则合并到前一个姿势段，直到所有短片段被合并完毕。
    2. **合并相邻的相同姿势片段**：
       - 如果相邻片段的 `orient` 相同，则合并。
    3. **移除时长小于 max_duration_sec 的首尾片段**：
       - 如果首尾片段的 `duration_frames < max_duration_frames`，则删除该片段的帧数据。
    4. **调整短片段的姿势**：
       - 如果片段 `duration_frames < max_duration_frames`，并且它的前后片段方向相同，则设为该方向，否则设为前一个方向。
    5. **再次合并相邻的相同姿势片段**：
       - 避免因调整姿势后产生的重复片段。

    参数：
        orientation_segments (list[dict]): 姿态信息片段，每个片段是一个字典。
        orientation (list[str]): 每帧的姿态信息。
        body_height (list[float]): 每帧的身体高度数据。
        head_y (list[float]): 每帧的头部 Y 轴数据。
        fps (int): 视频的帧率（默认 30）。
        min_duration_sec (int): 最小持续时间（秒），小于该值的片段将被合并（默认 3 秒）。
        max_duration_sec (int): 最大合并时长（秒），超过该值的片段才会被保留（默认 15 秒）。

    返回：
        tuple:
            - orientation_segments (list[dict]): 处理后的姿态片段。
            - orientation (list[str]): 过滤后的姿态信息。
            - body_height (list[float]): 过滤后的身体高度数据。
            - head_y (list[float]): 过滤后的头部 Y 轴数据。
    """

    # **计算帧数阈值**
    min_duration_frames = fps * min_duration_sec
    max_duration_frames = fps * max_duration_sec

    # **第一步：合并短片段，直到所有短片段被合并完毕**
    final_segments = orientation_segments[:]
    while True:
        updated_segments = []
        merged = False  # 记录是否发生了合并

        for i, segment in enumerate(final_segments):
            if updated_segments and segment["duration_frames"] < min_duration_frames:
                # **将短片段合并到前一个姿势段**
                updated_segments[-1]["end_frame"] = segment["end_frame"]
                updated_segments[-1]["duration_sec"] = (
                    updated_segments[-1]["end_frame"] - updated_segments[-1]["start_frame"] + 1
                ) / fps
                updated_segments[-1]["duration_frames"] = (
                    updated_segments[-1]["end_frame"] - updated_segments[-1]["start_frame"] + 1
                )
                merged = True  # 记录合并发生
            else:
                updated_segments.append(segment)

        if not merged:
            break  # 没有发生合并，跳出循环
        
        final_segments = updated_segments
        orientation_segments = final_segments  # 更新 segments

    # **第二步：合并相邻的相同姿势片段**
    merged_segments = []
    for segment in orientation_segments:
        if merged_segments and merged_segments[-1]["orient"] == segment["orient"]:
            # **合并相邻相同姿势片段**
            merged_segments[-1]["end_frame"] = segment["end_frame"]
            merged_segments[-1]["duration_sec"] = (merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1) / fps
            merged_segments[-1]["duration_frames"] = merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1
        else:
            merged_segments.append(segment)
    
    orientation_segments = merged_segments  # 更新 segments

    frames_to_remove = set()  # 记录要删除的帧

     # 从头开始遍历
    while orientation_segments:
        first_segment = orientation_segments[0]
        if first_segment['duration_frames'] < max_duration_frames:
            print(f"🗑 删除头部片段 (小于 {max_duration_sec} 秒): {first_segment}")
            frames_to_remove.update(range(first_segment['start_frame'], first_segment['end_frame'] + 1))
            orientation_segments.pop(0)
        else:
            break  # 遇到符合要求的片段，停止从头部遍历

    # 从尾部开始遍历
    while orientation_segments:
        last_segment = orientation_segments[-1]
        if last_segment['duration_frames'] < max_duration_frames:
            print(f"🗑 删除尾部片段 (小于 {max_duration_sec} 秒): {last_segment}")
            frames_to_remove.update(range(last_segment['start_frame'], last_segment['end_frame'] + 1))
            orientation_segments.pop(-1)
        else:
            break  # 遇到符合要求的片段，停止从尾部遍历

     # **第四步：删除最后一个小于 max_duration_sec 的片段**
    if orientation_segments and orientation_segments[-1]["duration_frames"] < 2 * max_duration_frames and orientation_segments[-1]['orient'] == 'neutral':
        last_segment = orientation_segments[-1]
        print(f"🗑 删除尾部片段 (小于 {max_duration_sec} 秒): {last_segment}")
        frames_to_remove.update(range(last_segment["start_frame"], last_segment["end_frame"] + 1))
        orientation_segments.pop(-1)  # 删除片段

    # **第五步：删除 `orientation`、`body_height` 和 `head_y` 中的相应帧**
    orientation = [orient for i, orient in enumerate(orientation) if i not in frames_to_remove]
    body_height = [height for i, height in enumerate(body_height) if i not in frames_to_remove]
    head_y = [head_y for i, head_y in enumerate(head_y) if i not in frames_to_remove]

    # **第六步：重新调整 segment 索引**
    if orientation_segments:
        new_segments = []
        prev_end_frame = 0
        
        for seg in orientation_segments:
            duration = seg["end_frame"] - seg["start_frame"]
            seg["start_frame"] = prev_end_frame
            seg["end_frame"] = prev_end_frame + duration
            prev_end_frame = seg["end_frame"] + 1
            new_segments.append(seg)
        
        orientation_segments = new_segments  # 更新 segments

    # **第七步：调整短片段的方向**
    for i in range(1, len(orientation_segments) - 1):  # 避免访问超出范围
        segment = orientation_segments[i]

        if segment["duration_frames"] < max_duration_frames:
            prev_orient = orientation_segments[i - 1]["orient"]

            # **如果前后姿势相同，则设为该姿势，否则设为前一个片段的姿势**
            segment["orient"] = prev_orient

    # **第八步：再次合并相邻相同姿势**
    merged_segments = []
    for segment in orientation_segments:
        if merged_segments and merged_segments[-1]["orient"] == segment["orient"]:
            merged_segments[-1]["end_frame"] = segment["end_frame"]
            merged_segments[-1]["duration_sec"] = (merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1) / fps
            merged_segments[-1]["duration_frames"] = merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1
        else:
            merged_segments.append(segment)

    orientation_segments = merged_segments  # 更新 segments
    
    return orientation_segments, orientation, body_height, head_y

def split_head_y_by_orientation(orientation_segments, head_y):
    """
    根据 orientation_segments 中的 start_frame 和 end_frame，分割 head_y 数据。

    主要逻辑：
    1. **遍历 orientation_segments**：
       - 每个片段包含 `start_frame` 和 `end_frame`，用于确定数据分割范围。
    2. **提取 head_y 片段**：
       - 取 `head_y[start:end+1]`，确保 `end_frame` 对应的帧也被包含在内。
    3. **存储分割后的 head_y 片段**：
       - 将切片结果存入 `segmented_head_y` 列表中，保持索引一致。

    参数：
        orientation_segments (list[dict]): 姿态信息片段，每个片段包含 `start_frame` 和 `end_frame`。
        head_y (list[float]): 头部 Y 坐标数据列表。

    返回：
        list[list[float]]: 分割后的 head_y 片段列表，每个片段对应一个 `orientation_segments` 片段。
    """

    segmented_head_y = []  # 存储分割后的 head_y 片段
    
    for segment in orientation_segments:
        start = segment['start_frame']
        end = segment['end_frame'] + 1  # 包含 `end_frame` 所在的索引
        head_y_segment = head_y[start:end]  # 提取对应的 head_y 数据
        
        segmented_head_y.append(head_y_segment)

    return segmented_head_y

def process_segmented_head_y(segmented_head_y, frame_window=400, max_timestamps=8, smooth_window=5, max_iterations=10):
    """
    处理 segmented_head_y，迭代检测突变点，分割数据，清理无效数据，并平滑断点。

    主要逻辑：
    1. **迭代处理数据**（最多 `max_iterations` 次）：
       - 逐个检查 `segmented_head_y`，移除短片段，并检测突变点。
    2. **检测并标记突变点**：
       - 计算 `threshold` 作为变化检测标准。
       - 使用 `ruptures` 进行突变点检测，识别显著变化区域。
    3. **去除无效数据**：
       - 若突变点在前 `frame_window` 帧或后 `frame_window` 帧，则标记为无效。
       - 对相邻突变点进行合并，减少误判。
       - 若突变点数量超过 `max_timestamps`，跳过该段数据，防止误分割。
    4. **分割数据**：
       - 依据突变点对数据进行分割，避免数据混乱。
       - 若相邻突变点间距过短，则跳过分割，以避免碎片化。
    5. **平滑数据**：
       - 对于每个分割片段，使用 `savgol_filter` 进行平滑，以减少噪声。
    6. **终止条件**：
       - 若数据在某次迭代后不再发生变化，则终止迭代，避免无限循环。

    参数：
        segmented_head_y (list of list): 头部 Y 轴数据，每个子列表表示一个时间序列片段。
        frame_window (int): 用于检测突变点的前后窗口大小（默认 400 帧）。
        max_timestamps (int): 允许的最大突变点数量，超出则跳过该段数据（默认 8）。
        smooth_window (int): 平滑窗口大小（默认 5）。
        max_iterations (int): 最大迭代次数，防止无限循环（默认 10）。

    返回：
        tuple:
            - processed_data (list of list): 处理后的分割数据。
            - split_info (list): 记录 `segmented_head_y` 的第几个元素被分割几次。
    """

    # **初始输入**
    processed_data = segmented_head_y
    split_info = list(range(len(segmented_head_y)))  # 记录初始索引
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        new_processed_data = []
        new_split_info = []

        has_changes = False  # 追踪是否有新的分割或数据清理

        for idx, segment in zip(split_info, processed_data):
            if len(segment) < 2 * frame_window:  # **数据过短，则跳过**
                new_processed_data.append(segment)
                new_split_info.append(idx)
                continue

            segment = np.array(segment, dtype=float)

            # ✅ **1. 计算自适应阈值**
            threshold = compute_adaptive_threshold(segment, "std", 1)

            # ✅ **2. 检测突变点**
            algo = rpt.Pelt(model="l2").fit(segment)
            change_points = algo.predict(pen=1)  # 获取突变点索引

            # ✅ **3. 处理突变点**
            invalid_indices = set()
            timestamps = []
            middle_timestamps = []  # 记录中间突变点（排除前 400 帧和后 400 帧）

            for cp in change_points:
                if cp < frame_window:  # **前 400 帧内**
                    before_mean = np.mean(segment[:cp])
                    after_mean = np.mean(segment[cp:cp + frame_window])
                    if abs(after_mean - before_mean) > threshold:
                        invalid_indices.update(range(0, cp + frame_window))

                elif cp > len(segment) - frame_window:  # **后 400 帧内**
                    before_mean = np.mean(segment[cp - frame_window:cp])
                    after_mean = np.mean(segment[cp:])
                    if abs(after_mean - before_mean) > threshold:
                        invalid_indices.update(range(cp - frame_window, len(segment)))

                else:  # **中间部分**
                    before_mean = np.mean(segment[cp - frame_window:cp])
                    after_mean = np.mean(segment[cp:cp + frame_window])
                    if abs(after_mean - before_mean) > threshold:
                        timestamps.append(cp)
                        middle_timestamps.append(cp)  # 记录中间部分突变点

            # ✅ **4. 处理中间的突变点**
            if len(middle_timestamps) > max_timestamps:
                print(f"Skipping segment {idx} due to too many middle timestamps: {len(middle_timestamps)}")
                timestamps = []  # 清空突变点，避免误分割

            # ✅ **5. 处理相邻突变点**
            new_timestamps = []
            last_cp = None

            for cp in timestamps:
                if last_cp is not None and (cp - last_cp) < frame_window:
                    invalid_indices.update(range(last_cp, cp))  # 标记该数据无效
                else:
                    new_timestamps.append(cp)
                    last_cp = cp
            timestamps = new_timestamps  # **更新 timestamps**

            # ✅ **6. 去除无效数据**
            valid_indices = sorted(set(range(len(segment))) - invalid_indices)
            filtered_segment = segment[valid_indices]  # 仅保留有效数据

            if len(valid_indices) < len(segment):  # **数据被修改**
                has_changes = True

            # ✅ **7. 分割数据**
            split_segments = []
            last_cp = 0
            for cp in timestamps:
                if cp - last_cp > 10:  # **避免分割太短**
                    split_segments.append(filtered_segment[last_cp:cp])
                    new_split_info.append(idx)
                last_cp = cp

            if last_cp < len(filtered_segment):  # **添加最后一个片段**
                split_segments.append(filtered_segment[last_cp:])
                new_split_info.append(idx)

            if len(split_segments) > 1:  # **发生了分割**
                has_changes = True

            # ✅ **8. 平滑断点**
            final_segments = []
            for sub_segment in split_segments:
                if len(sub_segment) > smooth_window:
                    sub_segment = savgol_filter(sub_segment, smooth_window, polyorder=2)
                final_segments.append(sub_segment)

            new_processed_data.extend(final_segments)

        # **检查是否还有变化**
        if not has_changes:
            print(f"Converged after {iteration} iterations.")
            break

        # **更新 processed_data 和 split_info**
        processed_data = new_processed_data
        split_info = new_split_info

    return processed_data, split_info

def detect_periodicity_acf_with_peaks(data, threshold=0.2, max_lag=300, min_ratio=0.4, min_alternations=6):
    """
    使用自相关函数 (ACF) 检测时间序列是否具有周期性，并计算最高峰值和最低峰值。

    主要逻辑：
    1. **计算 ACF（自相关函数）**：
       - 计算 `max_lag` 内的自相关值，用于分析数据的周期性。
    2. **统计滞后步长中显著相关的比例**：
       - 计算 `|ACF| > threshold` 的滞后值占比 `ratio`。
    3. **计算 ACF 的符号变化**：
       - 计算 `sign_changes`（ACF 的正负号）。
       - 计算 `alternation_count`（ACF 的正负交替次数）。
    4. **判断周期性**：
       - 只有当 `ratio > min_ratio` 且 `alternation_count >= min_alternations` 时，认为数据具有周期性。
    5. **计算均值和振幅**：
       - 计算数据的 `mean`。
       - 计算 `amp`（数据的 FFT 振幅，需调用 `compute_amplitude_fft`）。

    参数：
        data (array-like): 时间序列数据。
        threshold (float): 自相关系数阈值，绝对值大于此值才算显著相关（默认 0.2）。
        max_lag (int): 计算 ACF 时的最大滞后步长（默认 300）。
        min_ratio (float): 多少比例的滞后值需要超过 `threshold` 才算周期性（默认 0.4）。
        min_alternations (int): 至少多少次正负交替才算周期性（默认 6）。

    返回：
        tuple:
            - periodic (bool): 是否存在周期性。
            - mean (float): 数据均值。
            - amp (float): 数据的 FFT 振幅。
    """

    # **计算 ACF**
    acf_values = sm.tsa.acf(data, nlags=max_lag)

    # **统计 |ACF| 超过 threshold 的滞后点数量**
    above_threshold = np.sum(np.abs(acf_values[1:]) > threshold)  # 统计显著相关的点
    ratio = above_threshold / max_lag  # 计算占比

    # **计算 ACF 的正负变化**
    sign_changes = np.sign(acf_values[1:])  # 获取 ACF 的正负号 (+1 或 -1)
    alternation_count = np.sum(np.diff(sign_changes) != 0)  # 计算正负交替次数

    # **判断周期性**
    periodic = (ratio > min_ratio) and (alternation_count >= min_alternations)

    # **计算均值**
    mean = np.mean(data)

    # **计算数据的 FFT 振幅**
    amp = compute_amplitude_fft(data)  # 需要 `compute_amplitude_fft()` 方法支持

    return periodic, mean, amp

def split_orientation_segments(orientation_segments, segmented_head_y, split_info):
    """
    根据 segmented_head_y 和 split_info 对 orientation_segments 进行相应的分割，
    并按比例分配帧数，以保持数据完整性。

    主要逻辑：
    1. **计算原始片段的 frame 分配情况**：
       - 记录每个 `orientation_segments` 片段对应的 `segmented_head_y` 片段总长度 (`segment_lengths`)。
    2. **遍历 segmented_head_y 并进行分割**：
       - 按比例计算新的 `duration_frames`，确保帧数分配合理。
       - 确保 `start_frame` 和 `end_frame` 连续，避免数据不连贯。
    3. **生成新片段**：
       - 计算新的 `duration_sec`，保持 `FPS` 一致。
       - 创建新的 `orientation_segments`，存储在 `new_segments` 列表中。

    参数：
        orientation_segments (list[dict]): 原始姿态片段，每个字典包含:
            - "orient": 姿态方向
            - "start_frame": 片段起始帧索引
            - "end_frame": 片段结束帧索引
            - "duration_sec": 片段持续时间（秒）
            - "duration_frames": 片段持续帧数
        segmented_head_y (list[list[float]]): 分割后的时间序列数据，每个子列表对应一个分割部分。
        split_info (list[int]): 指示 `segmented_head_y` 每个元素属于哪个 `orientation_segments` 片段。

    返回：
        list[dict]: 重新分割后的 `orientation_segments` 片段列表。
    """

    new_segments = []

    # 记录每个原始片段的 frame 分配情况
    segment_allocations = {}  

    # 计算每个 segment_index 关联的 segmented_head_y 片段总长度
    segment_lengths = {}
    for i, segment_data in enumerate(segmented_head_y):
        segment_index = split_info[i]
        segment_lengths[segment_index] = segment_lengths.get(segment_index, 0) + len(segment_data)

    # 遍历 segmented_head_y 并进行分割
    for i, segment_data in enumerate(segmented_head_y):
        segment_index = split_info[i]  # 该数据片段属于哪个原 `orientation_segments` 片段
        orig_segment = orientation_segments[segment_index]  # 获取原始 `orientation_segments` 片段

        # 获取原始片段的信息
        orig_start_frame = orig_segment["start_frame"]
        orig_duration_frames = orig_segment["duration_frames"]

        # 按比例计算新的 `duration_frames`
        total_segment_length = segment_lengths[segment_index]  # 该片段所有 `segmented_head_y` 数据总长度
        ratio = len(segment_data) / total_segment_length
        new_duration_frames = round(ratio * orig_duration_frames)

        # 确保片段是连续的
        if segment_index not in segment_allocations:
            segment_allocations[segment_index] = orig_start_frame
        start_frame = segment_allocations[segment_index]
        end_frame = start_frame + new_duration_frames - 1

        # 计算帧率 (FPS) 以转换 `duration_frames -> duration_sec`
        fps = orig_segment["duration_sec"] / orig_duration_frames
        duration_sec = new_duration_frames * fps

        # 生成新片段
        new_segment = {
            "orient": orig_segment["orient"],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_sec": duration_sec,
            "duration_frames": new_duration_frames,
        }
        new_segments.append(new_segment)

        # 更新起始位置，确保下一片段的 `start_frame` 连续
        segment_allocations[segment_index] = end_frame + 1

    return new_segments

def compute_amplitude_fft(time_series):
    """
    计算主频及其对应的振幅（基于 FFT）。

    主要逻辑：
    1. **计算 FFT（快速傅里叶变换）**：
       - 获取 `time_series` 的频谱信息。
    2. **计算振幅谱**：
       - 归一化计算振幅，使得振幅大小独立于数据长度。
    3. **获取主频的振幅**：
       - 仅使用正频率部分（FFT 结果的前半部分）。
       - 忽略零频（直流分量），找到振幅最大的频率分量。

    参数：
        time_series (array-like): 输入的时间序列数据。

    返回：
        float: 主频对应的振幅。
    """

    N = len(time_series)  # **数据长度**
    fft_values = np.fft.fft(time_series)  # **计算 FFT**
    
    # **计算振幅谱（归一化处理）**
    amplitude_spectrum = (2 / N) * np.abs(fft_values)  # **振幅归一化**

    # **取正频率部分（去掉负频率）**
    positive_amplitude = amplitude_spectrum[:N // 2]

    # **找到主频索引（忽略零频）**
    main_freq_index = np.argmax(positive_amplitude[1:]) + 1  # **跳过直流分量（DC）**
    main_amplitude = positive_amplitude[main_freq_index]

    return main_amplitude

def update_orientation_segments(orientation_segments, periodics, means, amps):
    """
    根据 `periodics`、`means` 和 `amps` 更新 `orientation_segments`，添加 `head_y` 值：
    
    主要逻辑：
    1. **判断是否存在周期性**：
       - 若 `periodics[i] == True`，则 `head_y = [means[i] - amps[i], means[i] + amps[i]]`。
       - 若 `periodics[i] == False`，则 `head_y = means[i]`（无明显周期性，直接赋值）。
    2. **更新 `orientation_segments`**：
       - 遍历 `orientation_segments`，为每个片段计算 `head_y` 并存入字典。

    参数：
        orientation_segments (list[dict]): 待更新的姿态片段列表，每个字典包含：
            - "orient": 姿态方向
            - "start_frame": 片段起始帧索引
            - "end_frame": 片段结束帧索引
            - "duration_sec": 片段持续时间（秒）
            - "duration_frames": 片段持续帧数
        periodics (list[bool]): 是否存在周期性 (True / False)。
        means (list[float]): 每个片段的均值。
        amps (list[float]): 每个片段的振幅（周期性振幅）。

    返回：
        list[dict]: 包含 `head_y` 信息的 `orientation_segments`。
    """

    for i in range(len(orientation_segments)):
        if periodics[i]:
            orientation_segments[i]["head_y"] = [means[i] - amps[i], means[i] + amps[i]]  # **设定区间**
        else:
            orientation_segments[i]["head_y"] = means[i]  # **无周期性，直接赋值**

    return orientation_segments

def plot_combined_single_axis(head_y, orientation_durations):
    """
    在同一张图上绘制头部高度变化曲线，并用填充区域表示不同的面部朝向区域。
    对于 `motion_state` 为 "Static" 的区域，使用斜线填充进行标记。

    主要逻辑：
    1. **检查输入数据是否为空**：
       - 如果 `head_y` 或 `orientation_durations` 为空，则返回错误提示。
    2. **确定绘制区间**：
       - 计算 `start_frame` 和 `end_frame`，确保索引范围在 `head_y` 有效范围内。
    3. **处理 `head_y` 数据**：
       - 用 `NaN` 代替 `None` 值，并使用 `pandas.interpolate()` 进行线性插值，以平滑数据。
    4. **绘制头部高度变化曲线**：
       - 使用 `plt.plot()` 绘制 `head_y` 随时间的变化。
    5. **填充面部朝向区域**：
       - 定义 `Height_map`，用于将不同的 `orientation` 映射到特定的可视化高度。
       - 定义 `color_map`，用于指定不同 `orientation` 对应的填充颜色。
       - 遍历 `orientation_durations`，使用 `plt.fill_between()` 填充区域。
    6. **标记 `motion_state` 为 "Static" 的片段**：
       - 如果片段的 `motion_state` 为 "Static"，在填充区域上叠加黑色斜线 `hatch='//'` 进行标记。

    参数：
        head_y (list[float]): 头部高度数据，每个索引对应一个帧。
        orientation_durations (list[dict]): 包含面部朝向信息的片段，每个片段包含：
            - "start_frame": 片段起始帧索引。
            - "end_frame": 片段结束帧索引。
            - "orient": 片段的面部朝向（如 "neutral", "right", "up", "down"）。
            - "motion_state" (可选): 是否为 "Static"，如果是，则用斜线填充。

    返回：
        None: 直接在 `matplotlib` 画布上绘制图像，不返回值。

    """

    # **检查输入数据是否为空**
    if not head_y:
        print("错误: head_y 为空，无法绘制图表。")
        return
    
    if not orientation_durations:
        print("错误: orientation_durations 为空，无法确定绘制区间。")
        return

    # **确定绘制区间**
    start_frame = min(seg["start_frame"] for seg in orientation_durations)
    end_frame = max(seg["end_frame"] for seg in orientation_durations)

    start_frame = max(0, start_frame)
    end_frame = min(len(head_y) - 1, end_frame)

    # **处理 head_y 数据：插值填充缺失值**
    filtered_head_y = [head_y[i] if head_y[i] is not None else np.nan for i in range(start_frame, end_frame + 1)]
    filtered_head_y = pd.Series(filtered_head_y).interpolate(method='linear').tolist()

    x_values = np.arange(start_frame, end_frame + 1)

    # **创建 Matplotlib 画布**
    plt.figure(figsize=(10, 5))

    # **绘制 head_y 高度变化曲线**
    plt.plot(x_values, filtered_head_y, marker='o', markersize=3, linestyle='-', color='b', label='Height Variation')

    # **定义姿势对应的可视化高度**
    Height_map = {
        'neutral': 2,
        'right': 4,
        'up': 3,
        'down': 1,
        'down-neutral': 1.5,
        'neutral-down': 1.5,
        'up-neutral': 2.5,
        'neutral-up': 2.5
    }

    # **定义颜色映射**
    color_map = {
        'neutral': '#8dd3c7',
        # 'right': '#ffffb3',
        'up': '#fb8072',
        'down': '#ffffb3',
        'down-neutral': '#fdb462',
        'neutral-down': '#fdb462',
        'up-neutral': '#b3de69',
        'neutral-up': '#b3de69',
        'down-up': '#fccde5',
        'up-down': '#fccde5'
    }

    # **遍历 orientation_durations，填充姿势区域**
    for entry in orientation_durations:
        start_time = entry["start_frame"]
        end_time = entry["end_frame"]
        Height = Height_map.get(entry["orient"], 2)  # 默认值为 2，防止 key 错误

        # **填充背景色**
        plt.fill_between([start_time, end_time], 0, Height, 
                         color=color_map.get(entry["orient"], 'gray'), alpha=0.5, 
                         label=entry["orient"] if entry["orient"] not in plt.gca().get_legend_handles_labels()[1] else "")

        # **标记 motion_state 为 "Static" 的区域**
        if entry.get("motion_state") == "Static":
            plt.fill_between([start_time, end_time], 0, Height, 
                             facecolor='none', edgecolor='black', hatch='//', alpha=0.5)

    # **添加图例、标题、标签、网格**
    plt.xlabel("Frame Index")
    plt.ylabel("Height Level / Face Orientation")
    plt.title("Head Height & Face Orientation Variation Over Time")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # **显示图像**
    plt.show()

def plot_orientation_segments(orientation_segments):
    """
    绘制 `head_y` 变化（基于 `orientation_segments["head_y"]`）并填充片段下方的区域，
    处理片段间断点，并在 `Static` 片段上覆盖交叉线，同时标注 `orient` 方向。

    参数：
        orientation_segments (list[dict]): 姿态片段列表，每个字典包含：
            - "start_frame": 片段起始帧索引。
            - "end_frame": 片段结束帧索引。
            - "head_y": 头部高度 (单值或 `[min, max]` 区间)。
            - "orient": 姿势方向（如 "neutral", "right", "up", "down"）。

    返回：
        None: 直接在 `matplotlib` 画布上绘制图像，不返回值。
    """

    if not orientation_segments:
        print("错误: orientation_segments 为空，无法绘制图表。")
        return
    
    try:
        # 读取图片
        # 读取图片
        img_path = "full_body.png"
        img = Image.open(img_path)
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height
        print(f"✅ 图片加载成功: {img_path} (宽: {img_width}, 高: {img_height})")
    except Exception as e:
        print(f"❌ 图片加载失败: {e}")
        return
    
    
    
    # 创建主图
    fig, ax = plt.subplots(figsize=(12, 6))

    # **定义颜色映射**
    color_map = {
        'neutral': '#8dd3c7',
        # 'right': '#ffffb3',
        'up': '#fb8072',
        'down': '#bebada',
        'down-neutral': '#fdb462',
        'neutral-down': '#fdb462',
        'up-neutral': '#b3de69',
        'neutral-up': '#b3de69',
        'down-up': '#fccde5',
        'up-down': '#fccde5'
    }

    # **遍历 orientation_segments，绘制 head_y 轨迹**
    for entry in orientation_segments:
        start_time = entry["start_frame"]
        end_time = entry["end_frame"]
        head_y = entry["head_y"]
        orient = entry["orient"]

        # **获取颜色**
        color = color_map.get(orient, 'gray')

        # **生成 x 轴数据**
        x_values = np.arange(start_time, end_time + 1)

        # **生成 y 轴数据**
        if isinstance(head_y, (int, float)):  # **单值，绘制水平直线**
            y_values = np.full_like(x_values, head_y, dtype=float)

        elif isinstance(head_y, (list, tuple)) and len(head_y) == 2:  # **区间值，绘制振荡曲线**
            min_val, max_val = head_y
            num_points = len(x_values)
            num_oscillations = 3  # 指定往返的次数

            # 中间值 (起点与终点)
            mid_val = (min_val + max_val) / 2

            # 计算每次往返占用的点数（两个来回为一组）
            points_per_oscillation = num_points // (num_oscillations * 4)
            
            indices = []
            for _ in range(num_oscillations):
                # 生成一个完整的往返：中间值 -> max_val -> 中间值 -> min_val -> 中间值
                indices.extend(np.linspace(mid_val, max_val, points_per_oscillation))   # 中间值 -> 最大值
                indices.extend(np.linspace(max_val, mid_val, points_per_oscillation))   # 最大值 -> 中间值
                indices.extend(np.linspace(mid_val, min_val, points_per_oscillation))   # 中间值 -> 最小值
                indices.extend(np.linspace(min_val, mid_val, points_per_oscillation))   # 最小值 -> 中间值

            # 如果点数不够，补上中间点
            if len(indices) < num_points:
                indices = np.concatenate([indices, [mid_val] * (num_points - len(indices))])
            
            y_values = np.array(indices[:num_points])  # 确保 y_values 的长度与 x_values 一致
                
        else:
            continue  # **数据格式错误，跳过**

        # **填充曲线下方的区域**
        plt.fill_between(x_values, y_values, 0, color=color, alpha=0.5, label=orient if orient not in plt.gca().get_legend_handles_labels()[1] else "")

        # **在 orientation 片段顶部标注 orient**
        mid_x = (start_time + end_time) / 2
        mid_y = max(y_values) + 0.01  # **让文本稍微高于曲线**
        if '-' in orient:  # 如果是连接词
            word1, word2 = orient.split('-')
            # 判断较长的单词和较短的单词
            if len(word1) >= len(word2):
                plt.text(mid_x, mid_y + 0.03, word1, fontsize=10 , ha='center', va='bottom', color='black', fontfamily='Arial')
                plt.text(mid_x, mid_y, f'&{word2}', fontsize=10, ha='center', va='bottom', color='black', fontfamily='Arial')
            else:
                plt.text(mid_x, mid_y + 0.03, word2, fontsize=10, ha='center', va='bottom', color='black', fontfamily='Arial')
                plt.text(mid_x, mid_y, f'&{word1}', fontsize=10, ha='center', va='bottom', color='black', fontfamily='Arial')
        else:  # 如果是单词
            plt.text(mid_x, mid_y, orient, fontsize=10, ha='center', va='bottom', color='black', fontfamily='Arial')


    # **添加图例、标签、网格**
    plt.ylim(0, 1.1)
    plt.xlabel("Frame Index")
    plt.ylabel("Eye Height (Normalized)")
    plt.title("Nose Height and Facial Orientation Over Time")
    plt.legend(prop={'family': 'Arial'})
    plt.grid(True, linestyle='--', alpha=0.6)

    # 在左侧添加图片
    target_height =  0.72
    target_width = target_height * aspect_ratio
    ax_img = fig.add_axes([0.03, 0.1, target_width, target_height], anchor='W')  # 确保图片的高度与 0-1 对齐
    ax_img.imshow(img)
    ax_img.axis('off')
    ax_img.set_zorder(0)

    # **显示图像**
    plt.show()

def plot_orientation_segments(orientation_segments, save_path):
    """
    绘制 `head_y` 变化（基于 `orientation_segments["head_y"]`）并填充片段下方的区域，
    处理片段间断点，并在 `Static` 片段上覆盖交叉线，同时标注 `orient` 方向。

    参数：
        orientation_segments (list[dict]): 姿态片段列表，每个字典包含：
            - "start_frame": 片段起始帧索引。
            - "end_frame": 片段结束帧索引。
            - "head_y": 头部高度 (单值或 `[min, max]` 区间)。
            - "orient": 姿势方向（如 "neutral", "right", "up", "down"）。

    返回：
        None: 直接在 `matplotlib` 画布上绘制图像，不返回值。
    """

    if not orientation_segments:
        print("错误: orientation_segments 为空，无法绘制图表。")
        return
    
    try:
        # 读取图片
        # 读取图片
        left_img_path = os.path.join(IMAGE_FOLDER, 'full_body.png')
        left_img = Image.open(left_img_path)
        img_width, img_height = left_img.size
        aspect_ratio = img_width / img_height
        # right_img_path = os.path.join(IMAGE_FOLDER, 'indicator.png')
        # right_img = Image.open(right_img_path)
    except Exception as e:
        print(f"❌ 图片加载失败: {e}")
        return
    
    
    target_width_px, target_height_px = 1280, 600
    dpi = 100  # 每英寸的像素点数
    # 转换为英寸
    fig_width, fig_height = target_width_px / dpi, target_height_px / dpi
    # 创建图形
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    original_position = ax.get_position()  # 获取 ax 的位置信息
    new_position = [original_position.x0 + 0.05, original_position.y0,
                original_position.width, original_position.height]
    print(new_position)
    ax.set_position(new_position)

    # **定义颜色映射**
    color_map = {
        'neutral': '#59C9A5',
        # 'right': '#ffffb3',
        'up': '#FC9F5B',
        'down': '#D7B9D5',
        'neutral-down': '#1E96FC',
        'neutral-up': '#b3de69',
        'down-up': '#DE425B',
    }

    # previous_orient = None
    # accumulated_start_time = None
    # accumulated_end_time = None
    # accumulated_y_values = []

    total_frames = 0
    standing_frames = 0
    down_frames = 0
    transitions_count = 0  # Count of transitions
    total_low_frames = 0  # Total frames where head_y <= 0.6
    store_frames = 0

    # Track the previous state (None means uninitialized)
    previous_state = None  # Can be 'high' (>0.8), 'low' (<0.6), or None
    store_state = None

    # **遍历 orientation_segments，绘制 head_y 轨迹**
    for entry in orientation_segments:
        start_time = entry["start_frame"]
        end_time = entry["end_frame"]
        head_y = entry["head_y"]
        orient = entry["orient"]
        duration_frames = entry['duration_frames']

        total_frames += duration_frames
        current_state = None
        if isinstance(head_y, (int, float)):
            if head_y > 0.6:
                current_state = 'high'
                standing_frames += duration_frames
            else:
                current_state = 'low'
                total_low_frames += duration_frames
                if orient == 'down' or ('down' in orient.split('-')):
                    down_frames += duration_frames
                
        elif isinstance(head_y, list) and len(head_y) == 2:  # List of two values case
            if sum(head_y) / len(head_y) > 0.6:
                current_state = 'high'
                standing_frames += duration_frames
            else:
                current_state = 'low'
                total_low_frames += duration_frames
                if orient == 'down' or ('down' in orient.split('-')):
                    down_frames += duration_frames

        # Check if there is a transition from high to low or low to high
        if previous_state and current_state and previous_state != current_state:
            transitions_count += 1
            store_state = previous_state
            store_frames = total_frames
        # Update previous state
        if current_state:
            previous_state = current_state        


        # **获取颜色**
        color = color_map.get(orient, 'gray')

        # **生成 x 轴数据**
        x_values = np.arange(start_time, end_time + 1)

        # **生成 y 轴数据**
        if isinstance(head_y, (int, float)):  # **单值，绘制水平直线**
            y_values = np.full_like(x_values, head_y, dtype=float)
            head_y_value = head_y

        elif isinstance(head_y, (list, tuple)) and len(head_y) == 2:  # **区间值，绘制振荡曲线**
            min_val, max_val = head_y
            num_points = len(x_values)
            num_oscillations = 3  # 指定往返的次数

            # 中间值 (起点与终点)
            mid_val = (min_val + max_val) / 2
            head_y_value = mid_val

            # 计算每次往返占用的点数（两个来回为一组）
            points_per_oscillation = num_points // (num_oscillations * 4)
            
            indices = []
            for _ in range(num_oscillations):
                # 生成一个完整的往返：中间值 -> max_val -> 中间值 -> min_val -> 中间值
                indices.extend(np.linspace(mid_val, max_val, points_per_oscillation))   # 中间值 -> 最大值
                indices.extend(np.linspace(max_val, mid_val, points_per_oscillation))   # 最大值 -> 中间值
                indices.extend(np.linspace(mid_val, min_val, points_per_oscillation))   # 中间值 -> 最小值
                indices.extend(np.linspace(min_val, mid_val, points_per_oscillation))   # 最小值 -> 中间值

            # 如果点数不够，补上中间点
            if len(indices) < num_points:
                indices = np.concatenate([indices, [mid_val] * (num_points - len(indices))])
            
            y_values = np.array(indices[:num_points])  # 确保 y_values 的长度与 x_values 一致
                
        else:
            continue  # **数据格式错误，跳过**
        
        plt.plot(x_values, y_values, color=color, linewidth=2, label=orient if orient not in plt.gca().get_legend_handles_labels()[1] else "")
        
        # plt.fill_between(x_values, y_values, 0, color=color, alpha=0.3, label=orient if orient not in plt.gca().get_legend_handles_labels()[1] else "")
        
        # 计算增强后的颜色，使其更加鲜艳
        enhance_factor = 0.5  # 可以调整这个值来改变增强效果
        original_color = np.array(to_rgb(color))
        enhanced_color = original_color + (1 - original_color) * enhance_factor

        white_color = np.array([1, 1, 1])  # 白色

        # 先画底色填充区域 (开始是增强的颜色)
        plt.fill_between(x_values, y_values, 0, color=to_hex(enhanced_color))

        # 设置分层数量
        max_y = max(y_values)
        slice_size = 1 / 50  # 每次减少的值
        num_slices = int(max_y / slice_size) + 1  # 用整数除法并加1确保全覆盖
        delta_color = (white_color - enhanced_color) / num_slices  # 渐变步长

        current_y_values = y_values.copy()

        for slice_index in range(num_slices):
            # 计算新的颜色逐步变为白色
            current_color = enhanced_color + delta_color * slice_index
            current_color = np.clip(current_color, 0, 1)
            hex_color = to_hex(current_color)

            max_y = max(current_y_values)
            new_y_values = np.where(current_y_values > (max_y - slice_size), max_y - slice_size, current_y_values)
            
            # 只在值减少的情况下绘制
            if np.any(new_y_values < current_y_values):
                plt.fill_between(x_values, new_y_values, 0, color=hex_color)
                current_y_values = new_y_values
    

    # **添加图例、标签、网格**
    plt.rcParams['font.family'] = 'Segoe UI'
    plt.ylim(0, 1)
    plt.xlabel("Frame Index", fontsize=12, color='#525252')
    plt.ylabel("Eye Height (Normalized)", fontsize=12, color='#525252')
    # ax.get_yaxis().set_visible(False)
    ax.yaxis.set_ticks([])
    ax.set_xlim(left=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.8), fontsize=12, labelspacing=1.5)

    # ax.xaxis.set_major_locator(plt.MultipleLocator(1000))  # 每 10 单位画一条竖线
    max_x = orientation_segments[-1]["end_frame"]
    for step_size in [1000, 2000, 3000, 4000, 5000]:
        if (max_x / step_size) <= 15:  # 确保总共的刻度线数量不超过 15 条
            break
    ax.xaxis.set_major_locator(plt.MultipleLocator(step_size))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))  # 每 0.05 单位画一条横线
    ax.tick_params(axis='x', labelsize=10, labelcolor='#525252')
    ax.tick_params(axis='y', labelsize=10, labelcolor='#525252')
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.3)


    # 在左侧添加图片
    target_height =  0.79
    target_width = target_height * aspect_ratio
    ax_img1 = fig.add_axes([0.04, 0.09, target_width, target_height], anchor='W')  # 确保图片的高度与 0-1 对齐
    ax_img1.imshow(left_img)
    ax_img1.axis('off')
    ax_img1.set_zorder(0)

    # # 获取图片的原始像素大小
    # img_width, img_height = right_img.size

    # # 获取 fig 的大小 (单位是英寸)
    # fig_width, fig_height = fig.get_size_inches()

    # # 将图片的像素尺寸转化为规范化坐标 (0-1 范围)
    # norm_width = img_width / (fig_width * fig.dpi)
    # norm_height = img_height / (fig_height * fig.dpi)
    # x0, y0, width, height = new_position

    #  # Calculate ratios
    # standing_ratio = standing_frames / total_frames if total_frames > 0 else 0
    # down_ratio = down_frames / total_low_frames if total_low_frames > 0 else 0
    
    # image = []
    # if standing_ratio > 0.8:
    #     new_y = y0 - 0.01 + height *  2 / 3
    #     image.append(1)
    # elif transitions_count > 0:
    #     new_y = y0 - 0.01 + height *  1 / 3
    #     image.append(1)
    # else:
    #     new_y = y0 - 0.01
    #     if down_ratio > 0.8:
    #         image.append(3)
    #     else:
    #         image.append(2)

    # # 添加第二个图片
    # ax_img2 = fig.add_axes([0.85, new_y, norm_width, norm_height], anchor='W')
    # ax_img2.imshow(right_img)
    # ax_img2.axis('off')
    # ax_img2.set_zorder(0)

    # 保存图像到指定路径
    plt.savefig(save_path)
    plt.close(fig)

def analyze_video_orientation(orientation_segments, fps):
    total_frames = 0
    standing_frames = 0
    down_frames = 0
    transitions_count = 0  # Count of transitions
    total_low_frames = 0  # Total frames where head_y <= 0.6
    store_frames = 0

    # Track the previous state (None means uninitialized)
    previous_state = None  # Can be 'high' (>0.8), 'low' (<0.6), or None
    store_state = None

    for segment in orientation_segments:
        head_y = segment['head_y']
        duration_frames = segment['duration_frames']
        orient = segment['orient']

        # Update total frames
        total_frames += duration_frames

        # Determine the current state based on head_y
        current_state = None

        # Check if head_y is a single value
        if isinstance(head_y, (int, float)):
            if head_y > 0.6:
                current_state = 'high'
                standing_frames += duration_frames
            else:
                current_state = 'low'
                total_low_frames += duration_frames
                if orient == 'down' or ('down' in orient.split('-')):
                    down_frames += duration_frames
                
        elif isinstance(head_y, list) and len(head_y) == 2:  # List of two values case
            if all(value > 0.6 for value in head_y):
                current_state = 'high'
                standing_frames += duration_frames
            else:
                current_state = 'low'
                total_low_frames += duration_frames
                if orient == 'down' or ('down' in orient.split('-')):
                    down_frames += duration_frames

        # Check if there is a transition from high to low or low to high
        if previous_state and current_state and previous_state != current_state:
            transitions_count += 1
            store_state = previous_state
            store_frames = total_frames
        # Update previous state
        if current_state:
            previous_state = current_state        

    # Calculate ratios
    standing_ratio = standing_frames / total_frames if total_frames > 0 else 0
    down_ratio = down_frames / total_low_frames if total_low_frames > 0 else 0

    # Generate result sentences
    image = []
    segment1 = ""
    segment2 = ""
    segment3 = ""
    segment4 = ""
    if standing_ratio > 0.8:
        segment1 = "Height: On Stand"
        segment2 = "The video features standing movements for more than **80%** of its duration. It is recommended to place the playback device on a stand for optimal viewing."
        segment3 = "Orientation: Long Edge Side"
        segment4 = "It is recommended to place the device on one side of the long edge of the yoga mat."
        image.append(1)
    elif transitions_count > 1 :
        segment1 = "Height: > 2 Changes"
        segment2 = "Multiple adjustments to the playback device's height are required."
        segment3 = "Orientation: > 2 Changes"
        segment4 = "Multiple adjustments to the playback device's horizontal position are required."
    elif transitions_count == 1 :
        segment1 = "Height: 1 Change"
        minutes = round(store_frames / fps / 60)
        if store_state == 'high':
            segment2 = f"The first {minutes} minutes of this video consist of standing movements, so it is recommended to place the playback device on a stand. The later part features non-standing movements, for which it is advisable to place the playback device on the floor."
            if down_ratio > 0.8:
                segment3 = "Orientation: 1 Changes"
                segment4 = "It is recommended to place the device along the long edge of the yoga mat at first, and later along the short edge."
                image.append(1)
                image.append(3)
            else:
                segment3 = "Orientation: Long Edge Side"
                segment4 = "It is recommended to place the device along the long edge of the yoga mat."
                image.append(1)
                image.append(2)
        else:
            segment2 = f"The first {minutes} minutes of this video consist of non-standing movements, so it is recommended to place the playback device on the floor. The later part features standing movements, for which it is advisable to place the playback device on a stand."
            if down_ratio > 0.8:
                segment3 = "Orientation: 1 Changes"
                segment4 = "It is recommended to place the device on one side of the short edge of the yoga mat."
                image.append(3)
                image.append(1)
            else:
                segment3 = "Orientation: Long Edge Side"
                segment4 = "It is recommended to place the device along the long edge of the yoga mat."
                image.append(2)
                image.append(1)
    else:
        segment1 = "Height: Floor Level"
        segment2 = "All of the video's duration features non-standing movements, so it is recommended to place the playback device on the floor."
        if down_ratio > 0.8:
            segment3 = "Orientation: Short Edge Side"
            segment4 = ("It is recommended to place the device on one side of the short edge of the yoga mat.")
            image.append(3)
        else:
            segment3 = "Orientation: Long Edge Side"
            segment4 = ("It is recommended to place the device on one side of the long edge of the yoga mat.")
            image.append(2)

    segments = {
        "Segment1": segment1,
        "Segment2": segment2,
        "Segment3": segment3,
        "Segment4": segment4,
    }

    return segments, image

def cal_abs_value(num1, num2):
    return num1 / num2

def plot_orientation_bar_chart(orientation_segments, save_path):
    if not orientation_segments:
        print("错误: orientation_segments 为空，无法绘制柱状图。")
        return

    # 收集每个 orient 的累计时间
    orient_times = defaultdict(int)

    for entry in orientation_segments:
        start_time = entry["start_frame"]
        end_time = entry["end_frame"]
        orient = entry["orient"]
        duration = end_time - start_time + 1

        orient_times[orient] += duration

    total_frames = orientation_segments[-1]["end_frame"]
    sorted_orients = sorted(orient_times.items(), key=lambda x: x[1], reverse=False)
    orients = [item[0] for item in sorted_orients]
    times = [item[1] for item in sorted_orients]
    percentages = [round((time / total_frames) * 100) for time in times]

    # 定义颜色映射
    color_map = {
        'neutral': '#59C9A5',
        # 'right': '#ffffb3',
        'up': '#FC9F5B',
        'down': '#D7B9D5',
        'neutral-down': '#1E96FC',
        'neutral-up': '#b3de69',
        'down-up': '#DE425B',
    }
    colors = [color_map.get(orient, 'gray') for orient in orients]

    bar_height = 60
    bar_spacing = 20

    target_width_px = 1280
    if len(orients) == 1 :
        target_height_px =  bar_height + 20
    else: 
        target_height_px = bar_height * len(orients) + bar_spacing * (len(orients) - 1) + 20
    dpi = 100
    fig_width, fig_height = target_width_px / dpi, target_height_px / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    original_position = ax.get_position()  # 获取 ax 的位置信息
    new_position = [original_position.x0 + 0.05, original_position.y0 + 0.05,
                original_position.width, original_position.height]
    ax.set_position(new_position)


    plt.rcParams['font.family'] = 'Segoe UI'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_visible(False)


    y_positions = []

    for i, (orient_name, color) in enumerate(zip(orients, colors)):
        y_pos1 = cal_abs_value(10 + i * (bar_height + bar_spacing), target_height_px)
        y_pos2 = cal_abs_value(10 + i * (bar_height + bar_spacing) + 2, target_height_px)
        y_pos3 = cal_abs_value(10 + i * (bar_height + bar_spacing) + 30, target_height_px)
        y_positions.append(cal_abs_value(40 + i * (bar_height + bar_spacing), target_height_px))

        fancy_box = FancyBboxPatch(
            xy=(0.015, y_pos1),               # 矩形的左下角位置
            width=0.924,                   # 宽度
            height=cal_abs_value(bar_height, target_height_px), 
            boxstyle="round,pad=0.01",      # 矩形样式为圆角，带有内边距        
            facecolor=color,       # 填充颜色
            linewidth=0,                 # 边框宽度
            mutation_scale=1.5,          # 圆角的缩放比例
            alpha=0.3                    # 透明度
        )
        ax.add_patch(fancy_box)
        # ax.text(0.9, y_pos3, times[i], fontsize=14, ha='center', va='center', color=color)

        fancy_box = FancyBboxPatch(
            xy=(0.015, y_pos2),               # 矩形的左下角位置
            width=cal_abs_value(times[i], total_frames) * 0.9,  
            height=cal_abs_value(bar_height*0.95, target_height_px),                  # 高度
            boxstyle="round,pad=0.01",   # 矩形样式为圆角，带有内边距
            facecolor=color,       # 填充颜色
            linewidth=0,                 # 边框宽度
            mutation_scale=1.5,          # 圆角的缩放比例
            alpha= 1                  # 透明度
        )
        # ax.text(cal_abs_value(times[i], total_frames) * 0.9 + 0.08, y_pos3, f'{percentages[i]}%', fontsize=14, ha='center', va='center', color="black")
        ax.text(0.9, y_pos3, f'{percentages[i]}%', fontsize=14, ha='center', va='center', color=color)
        ax.add_patch(fancy_box)

            # 添加图片
        images_to_add = []
        if '-' in orient_name:
            words = orient_name.split('-')
            images_to_add = words  # 分割后的每个单词都代表一个图片
        else:
            images_to_add = [orient_name]  # 单个词标签
        
        # 添加图片
        for i, word in enumerate(images_to_add):
            image_path = os.path.join(IMAGE_FOLDER, f'{word}.png')
            if os.path.exists(image_path):  # 确保图片存在
                img = Image.open(image_path)
                imagebox = OffsetImage(img)
                if len(images_to_add) == 1:
                    ab = AnnotationBbox(imagebox, (0.01, y_pos3), 
                                    frameon=False, xycoords='axes fraction', box_alignment=(0, 0.5))
                else:
                    ab = AnnotationBbox(imagebox, (0.01 + 0.04 * i, y_pos3), 
                                    frameon=False, xycoords='axes fraction', box_alignment=(0, 0.5))
                ax.add_artist(ab)
        

    ax.set_yticks(y_positions)
    orients = [label.capitalize() for label in orients]
    ax.set_yticklabels(orients)
    ax.tick_params(axis='y', labelsize=12, labelcolor='#525252')

    # 保存并关闭图表
    plt.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":

    filename="tv6.json"
    fps, people_counts, body_height, orientation, head_y = load_json_data(filename)
    

    people_counts, orientation = smooth_stable_data(people_counts, orientation)
    orientation_segments = first_orientation_segments(orientation, body_height, head_y, fps)
    orientation_segments, orientation, body_height, head_y = filter_invalid_orientation_segments(orientation_segments, orientation, body_height, head_y, fps)
    
    change_points = detect_change_points(body_height, visualize=False)
    orientation_segments, orientation, body_height, head_y = remove_large_height_changes(change_points, orientation_segments, orientation, body_height, head_y, fps)
    orientation_segments = merge_alternating_orients(orientation_segments, fps)
    orientation_segments, orientation, body_height, head_y = merge_orientation_segments(orientation_segments, orientation, body_height, head_y, fps)


    segmented_head_y = split_head_y_by_orientation(orientation_segments, head_y)
    segmented_head_y, split_info = process_segmented_head_y(segmented_head_y)

    periodics = []
    means = []
    amps = []
    for segment in segmented_head_y:
        segment = np.array(segment, dtype=float)
        periodic, mean, amp = detect_periodicity_acf_with_peaks(segment)
        if periodic:
            if amp < 0.05:
                periodic = False
        periodics.append(periodic)
        means.append(mean)
        amps.append(amp)
    
    orientation_segments = split_orientation_segments(orientation_segments, segmented_head_y, split_info)
    orientation_segments = update_orientation_segments(orientation_segments, periodics, means, amps)


    IMAGE_FOLDER = "MoveMate/static/images"
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)

    image_path_1 = os.path.join(IMAGE_FOLDER, 'result_plot_1.png')
    image_path_2 = os.path.join(IMAGE_FOLDER, 'result_plot_2.png')
    


    plot_orientation_segments(orientation_segments, image_path_1)
    plot_orientation_bar_chart(orientation_segments, image_path_2)
    segments, image = analyze_video_orientation(orientation_segments, fps)
    print(segments)



    image_urls = {}
    image_urls[f"image_url_1"] = "/" + image_path_1 if image_path_1 else None
    image_urls[f"image_url_2"] = "/" + image_path_2 if image_path_2 else None
    image_urls["image_url1"] = None
    image_urls["image_url2"] = None
    i = 1
    for index, img in enumerate(image, start=1):  # 从1开始编号
        if img == 1:
            image_urls[f"image_url{i}"] = "/" + os.path.join(IMAGE_FOLDER, '1.png')
            i += 1
        elif img == 2:
            image_urls[f"image_url{i}"] = "/" + os.path.join(IMAGE_FOLDER, '2.png')
            i += 1
        elif img == 3:
            image_urls[f"image_url{i}"] = "/" + os.path.join(IMAGE_FOLDER, '3.png')

    # 构建返回的 JSON 数据
    response_data = {
        "done": True
    }
    response_data.update(segments)
    response_data.update(image_urls)  # 添加图片 URL 键值对


    print(orientation_segments)
    
    # plot_orientation_segments_with_images(orientation_segments)
    
    # plot_combined_single_axis(head_y, orientation_segments)


    

#     if previous_orient is None:  # 初始状态
    #         accumulated_start_time = start_time
    #         accumulated_end_time = end_time
    #         accumulated_y_values = y_values
    #     elif orient == previous_orient:  # 连续相同的 orient
    #         accumulated_end_time = end_time
    #         accumulated_y_values = np.concatenate((accumulated_y_values, y_values))
    #     else:  # 不同的 orient，进行文本绘制
    #         mid_x = (accumulated_start_time + accumulated_end_time) / 2
    #         mid_y = max(accumulated_y_values) + 0.03

    #         if '-' in previous_orient:  # 如果是连接词
    #             word1, word2 = previous_orient.split('-')
    #             plt.text(mid_x, mid_y + 0.04, word1, fontsize=12, ha='center', va='bottom', color='black')
    #             plt.text(mid_x, mid_y, f'&{word2}', fontsize=12, ha='center', va='bottom', color='black')
    #         else:  # 如果是单词
    #             plt.text(mid_x, mid_y, previous_orient, fontsize=12, ha='center', va='bottom', color='black')

    #         # 重置积累的值
    #         accumulated_start_time = start_time
    #         accumulated_end_time = end_time
    #         accumulated_y_values = y_values

    #     previous_orient = orient  # 更新当前的 orient

    #     # **在 orientation 片段顶部标注 orient**
    #     # mid_x = (start_time + end_time) / 2
    #     # mid_y = max(y_values) + 0.03  # **让文本稍微高于曲线**
    #     # if '-' in orient:  # 如果是连接词
    #     #     word1, word2 = orient.split('-')
    #     #     plt.text(mid_x, mid_y + 0.04, word1, fontsize=12, ha='center', va='bottom', color='black')
    #     #     plt.text(mid_x, mid_y, f'&{word2}', fontsize=12, ha='center', va='bottom', color='black')
    #     # else:  # 如果是单词
    #     #     plt.text(mid_x, mid_y, orient, fontsize=12, ha='center', va='bottom', color='black')
    
    # if previous_orient is not None:
    #     mid_x = (accumulated_start_time + accumulated_end_time) / 2
    #     mid_y = max(accumulated_y_values) + 0.03

    #     if '-' in previous_orient:  
    #         word1, word2 = previous_orient.split('-')
    #         plt.text(mid_x, mid_y + 0.04, word1, fontsize=12, ha='center', va='bottom', color='black')
    #         plt.text(mid_x, mid_y, f'&{word2}', fontsize=12, ha='center', va='bottom', color='black')
    #     else:
    #         plt.text(mid_x, mid_y, previous_orient, fontsize=12, ha='center', va='bottom', color='black')


# body_height = smooth_body_height(body_height, fps)
# orientation_segments = merge_RL_orientation_segments(orientation_segments, fps)


# def smooth_body_height(body_height, fps=30, min_duration_sec=2, window_size=5):
#     """
#     平滑 body_height 数据，检测并修正突变点（包含突然变大和突然变小的情况）。
    
#     主要逻辑：
#     1. 计算自适应阈值 threshold，用于检测 body_height 的突变区域。
#     2. 遍历数据，检测突变：
#        - 如果当前值与前一个值的差异超过 threshold，则标记为突变点。
#        - 找到突变区间的起点和终点。
#        - 如果突变区间的持续时间小于 `min_duration_sec`，则进行修正。
#     3. 修正方法：
#        - 计算突变区间前后 `window_size` 以内的数据的平均值。
#        - 用计算得到的局部均值替换突变区间内的值，使数据平滑。

#     参数：
#         body_height (list[float]): 需要平滑的 body_height 数据列表。
#         fps (int): 每秒帧数 (默认 30)。
#         method (str): 计算自适应阈值的方法（"std", "mad", "iqr"），默认 "std"。
#         k (float): 计算阈值时的乘数因子，控制敏感度 (默认 2)。
#         min_duration_sec (int): 最小突变持续时间（秒），低于该时间的突变区间会被修正 (默认 2)。
#         window_size (int): 计算局部均值时的窗口大小 (默认 5)。

#     返回：
#         list[float]: 平滑后的 body_height 数据。

#     """
#     # 计算突变的最小持续时间（帧数）
#     min_duration = fps * min_duration_sec
    
#     # 复制原始数据，避免修改输入列表
#     smoothed_body_height = body_height.copy()
#     n = len(body_height)

#     # 计算自适应 threshold，用于检测突变
#     threshold = compute_adaptive_threshold(body_height)

#     # 遍历数据，检测并修正突变
#     i = 1  # 从索引1开始遍历，避免访问负索引

#     while i < n - 1:
#         prev_height = body_height[i - 1]
#         current_height = body_height[i]

#         # 检测 body_height 的突变点
#         if abs(current_height - prev_height) > threshold:
#             start_idx = i  # 记录突变区间的起点

#             # 找到突变区间终点（数据恢复到原值域）
#             while i < n - 1 and abs(body_height[i] - prev_height) > threshold:
#                 i += 1
            
#             end_idx = i  # 记录突变区间终点

#             # 如果突变区间的持续时间小于 `min_duration`，则进行平滑处理
#             if (end_idx - start_idx) < min_duration:
#                 # 计算局部均值（使用前 `window_size` 个点 + 后 `window_size` 个点）
#                 window_start = max(0, start_idx - window_size)
#                 window_end = min(n, end_idx + window_size + 1)
#                 local_avg = sum(body_height[window_start:window_end]) / (window_end - window_start)

#                 # 替换突变区间的值，使数据平滑
#                 for j in range(start_idx, end_idx):
#                     smoothed_body_height[j] = local_avg

#                 print(f"Smoothed spike from index {start_idx} to {end_idx - 1}: replaced with {local_avg}")

#         i += 1  # 继续遍历数据

#     return smoothed_body_height


# def merge_RL_orientation_segments(orientation_segments, fps=30, min_duration_sec=3, max_duration_sec=15):
#     """
#     合并相邻的 'right' 和 'left' 方向片段，以减少短时波动。

#     主要逻辑：
#     1. **遍历所有片段**：
#        - 如果当前片段的方向是 `'right'` 或 `'left'`，则检查是否能与后续相同方向的片段合并。
#     2. **检查合并条件**：
#        - 如果后续片段与当前方向相同，则继续扩展合并区间。
#        - 如果遇到不同方向的片段，且其 `duration_frames` 超过 `min_duration_frames`，则停止合并。
#     3. **计算合并条件**：
#        - 若合并后该方向的总帧数大于 `total_frames / 2`，且总时长超过 `max_duration_frames`，则合并。
#     4. **标记已合并的片段**：
#        - 记录 `start_frame` 以避免重复合并。
#     5. **检查未合并的 'right' 和 'left' 片段**：
#        - 如果某个片段未被合并，则将其更改为 `'neutral'`，避免短时波动。

#     参数：
#         orientation_segments (list[dict]): 姿态信息片段，每个片段是一个字典。
#         fps (int): 视频的帧率（默认 30）。
#         min_duration_sec (int): 最小持续时间（秒），小于该值的片段不会单独保留（默认 3 秒）。
#         max_duration_sec (int): 最大合并时长（秒），超过该值的片段才会合并（默认 15 秒）。

#     返回：
#         list[dict]: 处理后的 `orientation_segments` 片段列表。
#     """

#     # 计算最小和最大允许的帧数
#     min_duration_frames = fps * min_duration_sec
#     max_duration_frames = fps * max_duration_sec

#     merged_segments = []  # 存储合并后的片段
#     i = 0
#     merged_right_left_frames = set()  # 记录已合并的片段 `start_frame`，防止索引失效

#     while i < len(orientation_segments):
#         segment = orientation_segments[i]
#         orient = segment['orient']

#         # 仅处理 'right' 或 'left' 方向
#         if orient in ('right', 'left'):
#             j = i + 1
#             last_matching_index = i  # 记录最后一个匹配片段的索引

#             while j < len(orientation_segments):
#                 next_segment = orientation_segments[j]
#                 if next_segment['orient'] == orient:
#                     last_matching_index = j  # 继续扩展合并区间

#                 # 如果遇到不同方向的片段，且时长大于 `min_duration_frames`，则停止合并
#                 if next_segment['orient'] != orient and next_segment['duration_frames'] > min_duration_frames:
#                     break
                
#                 j += 1  # 确保 j 递增，避免死循环

#             # 获取所有待合并的片段
#             collected_segments = [orientation_segments[k] for k in range(i, last_matching_index + 1)]
#             orient_frames = sum(seg['duration_frames'] for seg in collected_segments if seg['orient'] == orient)
#             total_frames = sum(seg['duration_frames'] for seg in collected_segments)

#             # 如果合并后符合条件，则合并
#             if orient_frames > total_frames / 2 and total_frames > max_duration_frames:
#                 merged_segment = {
#                     'orient': orient,
#                     'start_frame': collected_segments[0]['start_frame'],
#                     'end_frame': collected_segments[-1]['end_frame'],
#                     'duration_sec': sum(seg['duration_sec'] for seg in collected_segments),
#                     'duration_frames': total_frames
#                 }
#                 merged_segments.append(merged_segment)
                
#                 # 记录合并片段的 `start_frame`，避免索引失效
#                 merged_right_left_frames.update(seg['start_frame'] for seg in collected_segments)
                
#                 i = last_matching_index + 1  # 跳过合并的片段
#             else:
#                 # 不符合合并条件，直接添加
#                 merged_segments.append(segment)
#                 i += 1
#         else:
#             # 非 'right' 或 'left' 片段，直接添加
#             merged_segments.append(segment)
#             i += 1

#     # 修正逻辑：检查 `merged_segments`，若片段未合并，则改为 'neutral'
#     for segment in merged_segments:
#         if segment['orient'] in ('right', 'left') and segment['start_frame'] not in merged_right_left_frames:
#             segment['orient'] = 'neutral'

#     return merged_segments


 # 自定义图例位置
    # legend_labels = list(labels_used)
    # spacing_between_labels = 0.15  # 每个标签的上下距离增大
    # legend_positions = [(1.03, 0.95 - i * spacing_between_labels) for i in range(len(legend_labels))] 

    # for label, pos in zip(legend_labels, legend_positions):
    #     color = color_map.get(label, 'gray')
        
    #     # 添加颜色矩形
    #     rect = Rectangle((pos[0] - 0.015, pos[1] - 0.015), 0.01, 0.03, 
    #                      transform=ax.transAxes, color=color, clip_on=False, alpha=0.5)
    #     ax.add_patch(rect)
        
    #     # 添加标签文字
    #     ax.text(pos[0], pos[1], label, transform=ax.transAxes, fontsize=12, va='center', ha='left')
        
    #     # 添加图片
    #     images_to_add = []
    #     if '-' in label:
    #         words = label.split('-')
    #         images_to_add = words  # 分割后的每个单词都代表一个图片
    #     else:
    #         images_to_add = [label]  # 单个词标签
        
    #     # 添加图片
    #     for i, word in enumerate(images_to_add):
    #         image_path = os.path.join(IMAGE_FOLDER, f'{word}.png')
    #         if os.path.exists(image_path):  # 确保图片存在
    #             img = Image.open(image_path)
    #             imagebox = OffsetImage(img)
    #             if len(images_to_add) == 1:
    #                 ab = AnnotationBbox(imagebox, (pos[0] + 0.06 + 0.05 * i, pos[1]), 
    #                                 frameon=False, xycoords='axes fraction')
    #             else:
    #                 ab = AnnotationBbox(imagebox, (pos[0] + 0.09 + 0.04 * i, pos[1]), 
    #                                 frameon=False, xycoords='axes fraction')
    #             ax.add_artist(ab)

    # def plot_orientation_bar_chart(orientation_segments, save_path):
#     if not orientation_segments:
#         print("错误: orientation_segments 为空，无法绘制柱状图。")
#         return

#     # 收集每个 orient 的累计时间
#     orient_times = defaultdict(int)

#     for entry in orientation_segments:
#         start_time = entry["start_frame"]
#         end_time = entry["end_frame"]
#         orient = entry["orient"]
#         duration = end_time - start_time + 1

#         orient_times[orient] += duration

#     total_frames = orientation_segments[-1]["end_frame"]
#     sorted_orients = sorted(orient_times.items(), key=lambda x: x[1], reverse=False)
#     orients = [item[0] for item in sorted_orients]
#     times = [item[1] for item in sorted_orients]
#     percentages = [round((time / total_frames) * 100) for time in times]

#     # 定义颜色映射
#     color_map = {
#         'neutral': '#8dd3c7',
#         'up': '#fb8072',
#         'down': '#bebada',
#         'neutral-down': '#fdb462',
#         'neutral-up': '#b3de69',
#         'down-up': '#fccde5',
#     }
#     colors = [color_map.get(orient, 'gray') for orient in orients]

#     # 创建图表
#     target_width_px = 1000
#     target_height_px = 120 * len(orients) + 50
#     dpi = 100
#     fig_width, fig_height = target_width_px / dpi, target_height_px / dpi
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height))
#     original_position = ax.get_position()  # 获取 ax 的位置信息
#     new_position = [original_position.x0 + 0.01, original_position.y0 + 0.05,
#                 original_position.width, original_position.height]
#     ax.set_position(new_position)


#     plt.rcParams['font.family'] = 'Segoe UI'
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.get_xaxis().set_visible(False)

#     target_height_px = 100
#     data_height = fig_height * dpi
#     spacing_factor = 1
#     height_spacing = (target_height_px / data_height) * len(orients) * spacing_factor
#     height  = height_spacing * 0.7

#     y_poss = []

#     for i, (orient_name, color) in enumerate(zip(orients, colors)):
#         y_pos = float(i * height_spacing)
#         y_poss.append(y_pos)
#         rgba_color = np.array(to_rgb(color) + (0.2,))  # 添加透明度

#         ax.barh(y_pos, total_frames, color=rgba_color, height=height)
#         ax.barh(y_pos, times[i], color=color, height=height * 0.85)

#         ax.text(times[i] * 1.05, y_pos, f'{percentages[i]}%',
#                 va='center', ha='left', fontsize=14, color='black') 
#         ax.text(total_frames * 0.92, y_pos, f'{times[i]}',
#                 va='center', ha='left', fontsize=14, color=color)
        
#         # images = []
#         # total_width = 0
#         # for part in orient_name.split('-'):
#         #     img_path = os.path.join(IMAGE_FOLDER, f'{part}.png')
#         #     if os.path.exists(img_path):
#         #         img = Image.open(img_path)
#         #         images.append(img)
#         #         total_width += img.width

#         # if images:  # 如果有加载到的图片
#         #     x_offset = -total_width - 5  # 放在柱子左边
#         #     y_offset = y_pos - height / 2  # 用 i 而不是 y_pos
#         #     for img in images:
#         #         img_width, img_height = img.size
#         #         ax.imshow(img, extent=[x_offset, x_offset + img_width, y_offset, y_offset + height], aspect='auto')
#         #         x_offset += img_width

#     ax.set_yticks(y_poss)
#     ax.set_yticklabels(orients)
#     ax.tick_params(axis='y', labelsize=14)

#     # 保存并关闭图表
#     plt.savefig(save_path)
#     plt.close(fig)