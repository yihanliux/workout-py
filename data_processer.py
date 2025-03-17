import json
import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import statsmodels.api as sm

# ========================= 1. 读取 JSON 数据 =========================
def load_json_data(filename):
    """读取 JSON 文件并解析数据"""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"错误: 无法读取 {filename}，请检查文件路径或格式。")
        return 0, [], [], [], [], []
    
    # 解析 JSON 数据
    fps = data.get("fps", 30)  # 读取 fps，默认 30
    frames = data.get("frames", [])  # 读取帧数据列表
    
    people_counts = [frame.get("people_count") for frame in frames]
    body_height = [frame.get("body_height") for frame in frames]  
    orientation = [frame.get("orientation") for frame in frames] 
    head_y = [frame.get("head_y") for frame in frames]
    motion_states = [frame.get("motion_state") for frame in frames]
    
    print("视频总共有", len(people_counts), "帧，帧率:", fps, "FPS")
    
    return fps, people_counts, body_height, orientation, head_y, motion_states

def smooth_stable_data(people_counts, orientation, motion_states, window_size=10, consensus_ratio=0.8):
    """
    平滑数据，移除噪音，同时保留所有数据。
    使 `people_counts`, `orientation` 更稳定。
    
    参数:
        people_counts: list[int] - 每一帧的人数数据
        orientation list[str] - 每一帧的面部朝向
        motion_states: list[str] - 每一帧的运动状态 ('static' 或 'dynamic')
        window_size: int - 滑动窗口大小
        consensus_ratio: float - 认定最常见值的比例 (默认 80%)

    返回:
        filtered_people_counts, filtered_orientation, filtered_motion_states
    """
    filtered_people_counts = people_counts[:]
    filtered_orientation = orientation[:]
    filtered_motion_states = motion_states[:]

    for i in range(len(people_counts)):
        start, end = max(0, i - window_size), min(len(people_counts), i + window_size)

        # 计算滑动窗口内的最常见值
        most_common_people = max(set(people_counts[start:end]), key=people_counts[start:end].count)
        most_common_orientation = max(set(orientation[start:end]), key=orientation[start:end].count)
        most_common_motion = max(set(motion_states[start:end]), key=motion_states[start:end].count)

        # 计算最常见值的占比
        people_consensus = people_counts[start:end].count(most_common_people) / (end - start)
        orientation_consensus = orientation[start:end].count(most_common_orientation) / (end - start)
        motion_consensus = motion_states[start:end].count(most_common_motion) / (end - start)

        # 如果最常见值的比例超过 `consensus_ratio`，就采用它，否则保持原值
        filtered_people_counts[i] = most_common_people if people_consensus >= consensus_ratio else people_counts[i]
        filtered_orientation[i] = most_common_orientation if orientation_consensus >= consensus_ratio else orientation[i]
        filtered_motion_states[i] = most_common_motion if motion_consensus >= consensus_ratio else motion_states[i]

    return filtered_people_counts, filtered_orientation, filtered_motion_states

def filter_invalid_orientation_segments(orientation, body_height, head_y, fps=30, min_duration_sec=3, max_duration_sec=90):
    """
    1. 记录所有 people_counts == 1 的连续间隔
    2. 合并短于 fps * duration_sec 的区间到前一个姿势
    3. 合并相邻的相同姿势
    4. 删除最后一个 'Invalid' 片段
    5. 计算每个片段的 body_height 方差，删除异常片段
    """
    min_duration_frames = fps * min_duration_sec
    max_duration_frames = fps * max_duration_sec
    total_frames = len(orientation)
    orient_segments = []
    current_orient, start_frame = None, None

     # 预处理，将 None 变成 'Invalid'
    orientation = ['Invalid' if orient is None else orient for orient in orientation]

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

    # 找出中间 80%（不在前10% 和 后10%）的 `Invalid` 片段
    long_invalid_segments = [
        segment for segment in orient_segments
        if segment["orient"] == "Invalid" and segment["duration_frames"] > min_duration_frames
        and first_10_percent <= segment["start_frame"] < last_10_percent
    ]

    # 按持续时间降序排序
    long_invalid_segments.sort(key=lambda seg: seg["duration_frames"], reverse=True)

    # 处理所有的长 `Invalid` 片段
    processed_segments = set()  # 记录已处理的 `Invalid` 片段

    while long_invalid_segments:
        # 取当前最长的 `Invalid` 片段
        segment = long_invalid_segments.pop(0)

        search_start = max(first_10_percent, segment["start_frame"] - max_duration_frames)
        search_end = min(last_10_percent, segment["end_frame"] + max_duration_frames)

        # 找到 2 分钟内最早和最晚的 `Invalid` 片段
        earliest_invalid = segment
        latest_invalid = segment

        for seg in orient_segments:
            if seg["orient"] == "Invalid" and seg["end_frame"] >= search_start and seg["start_frame"] <= search_end:
                if seg["start_frame"] < earliest_invalid["start_frame"]:
                    earliest_invalid = seg
                if seg["end_frame"] > latest_invalid["end_frame"]:
                    latest_invalid = seg

        # 找到 earliest_invalid 到 latest_invalid 之间的所有片段，并合并成一个片段
        merged_start = earliest_invalid["start_frame"]
        merged_end = latest_invalid["end_frame"]
        merged_duration = merged_end - merged_start + 1

        # 先删除这些片段
        orient_segments = [
            seg for seg in orient_segments
            if seg["end_frame"] < merged_start or seg["start_frame"] > merged_end
        ]

        # 然后新增合并后的 "Invalid" 片段
        merged_segment = {
            "orient": "Invalid",
            "start_frame": merged_start,
            "end_frame": merged_end,
            "duration_sec": merged_duration / fps,
            "duration_frames": merged_duration
        }
        # 插入合并的 Invalid 片段
        orient_segments.append(merged_segment)

        processed_segments.add((merged_segment["start_frame"], merged_segment["end_frame"]))

        # 更新 long_invalid_segments，去掉已处理片段，并找新的最长片段
        long_invalid_segments = [
            seg for seg in orient_segments
            if seg["orient"] == "Invalid" and seg["duration_frames"] > min_duration_frames
            and first_10_percent <= seg["start_frame"] < last_10_percent
            and (seg["start_frame"], seg["end_frame"]) not in processed_segments
        ]

        # 按持续时间降序排序，确保下次处理最长的
        long_invalid_segments.sort(key=lambda seg: seg["duration_frames"], reverse=True)

    # 4️⃣ 删除超过 1 秒的 "Invalid" 片段
    frames_to_remove = set()
    for segment in orient_segments:
        if segment["orient"] == "Invalid" and segment["duration_frames"] > min_duration_frames:
            frames_to_remove.update(range(segment["start_frame"], segment["end_frame"] + 1))
    
    updated_orient_segments = []
    for segment in orient_segments:
        if segment["orient"] == "Invalid" and segment["duration_frames"] > min_duration_frames:
            print(f"Deleted Invalid segment: Start {segment['start_frame']}, End {segment['end_frame']}, Duration {segment['duration_frames']} frames ({segment['duration_sec']} sec)")
        else:
            updated_orient_segments.append(segment)  # 只保留未被删除的片段

    frames_to_keep = set(range(total_frames)) - frames_to_remove
    updated_orientation = [orientation[i] for i in frames_to_keep]
    updated_body_height = [body_height[i] for i in frames_to_keep]
    updated_head_y = [head_y[i] for i in frames_to_keep]
    
    new_orient_segments = []
    frames_to_remove = set()  # 记录需要删除的帧索引
    
    # 🚀 计算全局 body_height 中位数，防止 None
    non_none_body_height = [h for h in updated_body_height if h is not None]
    global_median_body_height = np.median(non_none_body_height) if non_none_body_height else 0

    non_none_head_y = [h for h in updated_head_y if h is not None]
    global_median_head_y = np.median(non_none_head_y) if non_none_head_y else 0
    

    # 遍历所有片段，找到 "Invalid" 片段
    for i in range(len(updated_orient_segments)):
        segment = updated_orient_segments[i]

        if segment["orient"] == "Invalid":
            prev_segment = updated_orient_segments[i - 1] if i > 0 else None
            next_segment = updated_orient_segments[i + 1] if i < len(updated_orient_segments) - 1 else None

            # 🚀 如果前后片段不存在，则直接删除该 "Invalid" 片段
            if not prev_segment or not next_segment:
                print(f"Deleted Invalid segment (no adjacent): Start {segment['start_frame']}, End {segment['end_frame']}")
                frames_to_remove.update(range(segment["start_frame"], segment["end_frame"] + 1))
                continue  # 跳过这个片段，不加入 new_orient_segments
            
            # 🚀 计算 body_height
            prev_frames = range(prev_segment["start_frame"], prev_segment["end_frame"] + 1)
            next_frames = range(next_segment["start_frame"], next_segment["end_frame"] + 1)
            invalid_frames = range(segment["start_frame"], segment["end_frame"] + 1)
            
            if prev_segment["orient"] == next_segment["orient"]:
                all_frames = list(prev_frames) + list(next_frames)
                all_valid_heights = [updated_body_height[f] for f in all_frames if updated_body_height[f] is not None]
                avg_body_height = np.mean(all_valid_heights) if all_valid_heights else global_median_body_height
                all_valid_heights = [updated_head_y[f] for f in all_frames if updated_head_y[f] is not None]
                avg_head_y = np.mean(all_valid_heights) if all_valid_heights else global_median_head_y
                segment["orient"] = prev_segment["orient"]
            else:
                valid_prev_heights = [updated_body_height[f] for f in prev_frames if updated_body_height[f] is not None]
                avg_body_height = np.mean(valid_prev_heights) if valid_prev_heights else global_median_body_height
                valid_prev_heights = [updated_head_y[f] for f in prev_frames if updated_head_y[f] is not None]
                avg_head_y = np.mean(valid_prev_heights) if valid_prev_heights else global_median_head_y
                segment["orient"] = prev_segment["orient"]

            # 🚀 更新 "Invalid" 片段对应的 body_height
            for f in invalid_frames:
                updated_body_height[f] = avg_body_height
                updated_head_y[f] = avg_head_y

            # 加入更新后的片段
            new_orient_segments.append(segment)
        else:
            new_orient_segments.append(segment)

    # 🚀 过滤 updated_orientation 和 updated_body_height，删除指定帧
    updated_orientation = [orient for i, orient in enumerate(updated_orientation) if i not in frames_to_remove]
    updated_body_height = [height for i, height in enumerate(updated_body_height) if i not in frames_to_remove]
    updated_head_y = [head_y for i, head_y in enumerate(updated_head_y) if i not in frames_to_remove]

    # 🚀 处理 `None` 值：用前一个值填充
    for i in range(len(updated_body_height)):
        if updated_body_height[i] is None:
            updated_body_height[i] = updated_body_height[i - 1] if i > 0 else global_median_body_height
    
    # 🚀 处理 `None` 值：用前一个值填充
    for i in range(len(updated_head_y)):
        if updated_head_y[i] is None:
            updated_head_y[i] = updated_head_y[i - 1] if i > 0 else global_median_head_y

    if new_orient_segments:
        new_segments = []
        prev_end_frame = 0
        
        for seg in new_orient_segments:
            duration = seg["end_frame"] - seg["start_frame"]
            seg["start_frame"] = prev_end_frame
            seg["end_frame"] = prev_end_frame + duration
            prev_end_frame = seg["end_frame"] + 1
            new_segments.append(seg)
        
        new_orient_segments = new_segments

    return new_orient_segments, updated_orientation, updated_body_height, updated_head_y
    
def compute_adaptive_threshold(data, method="std", k=2):
    """
    计算数据的自适应阈值：
    - method="std"  -> 使用标准差 threshold = k * std
    - method="mad"  -> 使用平均绝对偏差（MAD）
    - method="iqr"  -> 使用 IQR（四分位距）
    
    参数：
        - data: 需要计算阈值的 body_height 数据列表
        - method: 选择计算方法
        - k: 乘法因子，控制敏感度
    返回：
        - threshold: 计算出的动态阈值
    """
    data = np.array(data)

    if method == "std":
        threshold = k * np.std(data)
    elif method == "mad":
        median = np.median(data)
        mad = np.median(np.abs(data - median))  # 计算MAD
        threshold = k * mad
    elif method == "iqr":
        Q1 = np.percentile(data, 25)  # 第一四分位数
        Q3 = np.percentile(data, 75)  # 第三四分位数
        iqr = Q3 - Q1  # 四分位距
        threshold = k * iqr
    else:
        raise ValueError("Unsupported method. Choose from ['std', 'mad', 'iqr']")
    
    return threshold

def smooth_body_height(body_height, fps=30, method="std", k=2, min_duration_sec=2, window_size=5):
    """
    平滑 body_height 数据，检测并修正突变点（包含突然变大和突然变小的点）。
    
    参数：
        - body_height: 需要平滑的 body_height 列表
        - threshold: 变化超过该值，则认为是突变点（默认0.2米）
        - max_duration: 如果突变区间小于该值（帧数），则修正
        - window_size: 计算局部平均值的窗口大小（默认3）
    """
    min_duration = fps * min_duration_sec
    smoothed_body_height = body_height.copy()  # 复制数据
    n = len(body_height)

    # 🚀 计算自适应 threshold
    threshold = compute_adaptive_threshold(body_height, method, k)
    print(f"Computed adaptive threshold ({method} method): {threshold}")

    i = 1  # 从索引1开始遍历，避免访问负索引

    while i < n - 1:
        prev_height = body_height[i - 1]
        current_height = body_height[i]

        # 🚀 检测突变区间开始（如果当前值比前一个值 "突然变大" 或 "突然变小"）
        if abs(current_height - prev_height) > threshold:
            start_idx = i  # 记录突变区间起点

            # 找到突变区间结束（数据恢复到原值域）
            while i < n - 1 and abs(body_height[i] - prev_height) > threshold:
                i += 1
            
            end_idx = i  # 记录突变区间终点

            # 如果突变持续时间小于 min_duration，则修正
            if (end_idx - start_idx) < min_duration:
                # 计算局部平均值（前 window_size 个点 + 后 window_size 个点）
                window_start = max(0, start_idx - window_size)
                window_end = min(n, end_idx + window_size + 1)
                local_avg = sum(body_height[window_start:window_end]) / (window_end - window_start)

                # 🚀 替换整个突变区间的值
                for j in range(start_idx, end_idx):
                    smoothed_body_height[j] = local_avg

                print(f"Smoothed spike from index {start_idx} to {end_idx - 1}: replaced with {local_avg}")

        i += 1  # 继续遍历
    

    return smoothed_body_height

def detect_change_points(data, percentile=95, window_size=3, visualize=True):
    """
    检测数据中的突变点（上升或下降）。

    参数:
    - data (array-like): 输入的时间序列数据 (body_height)
    - percentile (float): 变化点的阈值（分位数），默认 99（取前 1% 最大变化点）
    - window_size (int): 变化后维持高值的窗口大小，默认 5
    - visualize (bool): 是否可视化结果，默认 True

    返回:
    - change_points (list): 变化点的索引列表
    """

    # 确保 data 是 numpy 数组
    data = np.array(data, dtype=float)

    # 计算变化率（取绝对值）
    diff = np.abs(np.diff(data))
    # data = np.nan_to_num(data)

    # 计算阈值（取前 percentile% 最大变化点）
    threshold = np.percentile(diff, percentile)

    # 检测变化点
    jump_points, _ = find_peaks(diff, height=threshold)

    # 过滤掉短暂突变的点（确保后续数值维持在高值）
    change_points = [
        p for p in jump_points if np.mean(data[p:p + window_size]) > np.mean(data) + np.std(data)
    ]

    # 可视化
    if visualize:
        plt.figure(figsize=(12, 5))
        plt.plot(data, label="Data")
        plt.scatter(change_points, data[change_points], color='red', label="Change Points")
        plt.legend()
        plt.title("Detected Change Points")
        plt.show()

    return change_points

def remove_large_height_changes(change_points, orientation_segments, orientation, body_height, head_y, fps=30, method="std", k=2, max_duration_sec=90):
    """
    处理 change_points，删除异常突变片段：
    1. 先处理前 10% 和后 10% 的突变点，删除超出 threshold 的片段。
    2. 遍历所有 change_points，寻找 max_duration_frames 内的最后一个突变点，计算 body_height 均值变化。
    3. 如果 body_height 的突变区域和其他区域均值相差超过 threshold，则删除该片段。

    参数：
        - change_points: 突变点的索引数组
        - orientation_segments: 姿态信息片段
        - orientation: 每帧的姿态信息
        - body_height: 每帧的身体高度数组
        - head_y: 头部高度数组
        - fps: 视频帧率
        - method: 计算 threshold 的方法 ["std", "mad", "iqr"]
        - k: 计算 threshold 的乘法因子
        - max_duration_sec: 最大持续时间（秒）

    返回：
        - 更新后的 orientation_segments, orientation, body_height, head_y
    """
    total_frames = len(body_height)
    max_duration_frames = fps * max_duration_sec
    threshold = compute_adaptive_threshold(body_height, method, k)
    frames_to_remove = set()

    # 🚀 遍历所有 change_points，寻找 max_duration_frames 内的最后一个 change_point
    for i, cp in enumerate(change_points):
        future_changes = [p for p in change_points if p > cp and p <= cp + max_duration_frames]

        if future_changes:
            end_idx = max(future_changes)  # 选取 max_duration_frames 内最后的 change_point
        else:
            end_idx = cp  # 仅一个突变点，保持不变

        start_idx = cp  # 当前 change_point 起点

        # 计算该区间 body_height 的均值
        region_mean = np.mean(body_height[start_idx:end_idx])

        # 计算全局 body_height 均值（排除该区间）
        other_frames = [i for i in range(total_frames) if i < start_idx or i > end_idx]
        if other_frames:
            other_mean = np.mean([body_height[i] for i in other_frames])
        else:
            other_mean = np.mean(body_height)  # 只有该区间存在，直接取全局均值

        # 🚀 计算差距
        height_diff = abs(region_mean - other_mean)

        # 🚀 检查 `change_point` 是否在前 max_duration_frames 或后 max_duration_frames
        has_early_change = any(p <= max_duration_frames for p in [start_idx, end_idx])
        has_late_change = any(p >= total_frames - max_duration_frames for p in [start_idx, end_idx])

        if height_diff > threshold:
            if has_early_change:
                #print(f"Removing frames from 0 to {end_idx} due to large height change: {height_diff}")
                frames_to_remove.update(range(0, end_idx + 1))
            elif has_late_change:
                #print(f"Removing frames from {start_idx} to {total_frames} due to large height change: {height_diff}")
                frames_to_remove.update(range(start_idx, total_frames))
            else:
                #print(f"Removing frames from {start_idx} to {end_idx} due to large height change: {height_diff}")
                frames_to_remove.update(range(start_idx, end_idx + 1))

    # 🚀 过滤 orientation_segments，并同步删除相应帧的数据
    new_frames_to_remove = frames_to_remove

    updated_orientation_segments = []
    for seg in orientation_segments:
        if not any(frame in frames_to_remove for frame in range(seg["start_frame"], seg["end_frame"] + 1)):
            updated_orientation_segments.append(seg)
        else:
            new_frames_to_remove.update(range(seg["start_frame"], seg["end_frame"] + 1))
    
    frames_to_keep = set(range(total_frames)) - new_frames_to_remove
    updated_orientation = [orientation[i] for i in frames_to_keep]
    updated_body_height = [body_height[i] for i in frames_to_keep]
    updated_head_y = [head_y[i] for i in frames_to_keep]

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

def merge_RL_orientation_segments(orientation_segments, fps=30, min_duration_sec=3, max_duration_sec=15):
    min_duration_frames = fps * min_duration_sec
    max_duration_frames = fps * max_duration_sec
    merged_segments = []
    i = 0
    merged_right_left_frames = set()  # ✅ 用 `start_frame` 记录合并片段，避免索引失效

    while i < len(orientation_segments):
        segment = orientation_segments[i]
        orient = segment['orient']

        if orient in ('right', 'left'):
            j = i + 1
            last_matching_index = i

            while j < len(orientation_segments):
                next_segment = orientation_segments[j]
                if next_segment['orient'] == orient:
                    last_matching_index = j
                if next_segment['orient'] != orient and next_segment['duration_frames'] > min_duration_frames:
                    break
                j += 1  # ✅ 确保 j 递增，避免死循环

            collected_segments = [orientation_segments[k] for k in range(i, last_matching_index + 1)]
            orient_frames = sum(seg['duration_frames'] for seg in collected_segments if seg['orient'] == orient)
            total_frames = sum(seg['duration_frames'] for seg in collected_segments)

            if orient_frames > total_frames / 2 and total_frames > max_duration_frames:
                merged_segment = {
                    'orient': orient,
                    'start_frame': collected_segments[0]['start_frame'],
                    'end_frame': collected_segments[-1]['end_frame'],
                    'duration_sec': sum(seg['duration_sec'] for seg in collected_segments),
                    'duration_frames': total_frames
                }
                merged_segments.append(merged_segment)
                
                # ✅ 记录合并片段的 `start_frame`，避免索引失效
                merged_right_left_frames.update(seg['start_frame'] for seg in collected_segments)
                
                i = last_matching_index + 1
            else:
                merged_segments.append(segment)
                i += 1
        else:
            merged_segments.append(segment)
            i += 1

    # ✅ 修正逻辑：遍历 `merged_segments`，使用 `start_frame` 检查是否合并过
    for segment in merged_segments:
        if segment['orient'] in ('right', 'left') and segment['start_frame'] not in merged_right_left_frames:
            segment['orient'] = 'neutral'

    return merged_segments

def merge_alternating_orients(orientation_segments, fps=30, max_swaps=18, min_duration_sec=3):

    # **合并相邻相同姿势**
    merged_segments = []
    for segment in orientation_segments:
        if merged_segments and merged_segments[-1]["orient"] == segment["orient"]:
            # 合并到前一个相同姿势段
            merged_segments[-1]["end_frame"] = segment["end_frame"]
            merged_segments[-1]["duration_sec"] = (merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1) / fps
            merged_segments[-1]["duration_frames"] = merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1
        else:
            merged_segments.append(segment)
    orientation_segments = merged_segments 

    min_duration_frames = fps * min_duration_sec
    result = []
    i = 0
    
    while i < len(orientation_segments) - 1:
        current_orient = orientation_segments[i]['orient']
        next_orient = orientation_segments[i + 1]['orient']
        swap_count = 0
        combined_segments = [orientation_segments[i]]
        j = i + 1  # 用 j 来收集后续片段
        
        if current_orient != next_orient:
            combined_segments.append(orientation_segments[j])
            j += 1
            
            while j < len(orientation_segments):
                third_orient = orientation_segments[j]['orient']
                third_segment = orientation_segments[j]
                
                if (third_orient in [current_orient, next_orient] and
                    third_orient != combined_segments[-1]['orient'] and
                    third_segment['duration_frames'] < min_duration_frames):
                    swap_count += 1
                    combined_segments.append(third_segment)
                    j += 1  # 继续遍历
                else:
                    break  # 规则被破坏，停止合并
            
            if swap_count > max_swaps:
                combined_orient = f"{current_orient}-{next_orient}"
                merged_segment = {
                    'orient': combined_orient,
                    'start_frame': combined_segments[0]['start_frame'],
                    'end_frame': combined_segments[-1]['end_frame'],
                    'duration_sec': sum(seg['duration_sec'] for seg in combined_segments),
                    'duration_frames': sum(seg['duration_frames'] for seg in combined_segments)
                }
                result.append(merged_segment)
                print(merged_segment)
            else:
                result.extend(combined_segments)
            
            i = j  # 跳到下一个未处理的片段
        else:
            result.append(orientation_segments[i])
            i += 1  # 继续主循环遍历
    
    # 追加最后一个 segment，如果它未被处理
    if i == len(orientation_segments) - 1:
        result.append(orientation_segments[i])
    
    return result


def merge_orientation_segments(orientation_segments, orientation, body_height, head_y, fps=30, min_duration_sec=3, max_duration_sec=15):

    min_duration_frames = fps * min_duration_sec
    max_duration_frames = fps * max_duration_sec
    
    
    final_segments = orientation_segments[:]  # 先复制 orient_segments
    while True:  # 进入循环，直到所有短片段都被合并
        updated_segments = []
        merged = False  # 记录是否发生了合并

        for i, segment in enumerate(final_segments):
            if updated_segments and segment["duration_frames"] < min_duration_frames:
                # **合并短片段到前一个姿势段**
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

        # 如果本轮没有发生合并，跳出循环
        if not merged:
            break
        # 更新 final_segments，进行下一轮合并检查
        final_segments = updated_segments
        orientation_segments = final_segments
    
    # **合并相邻相同姿势**
    merged_segments = []
    for segment in orientation_segments:
        if merged_segments and merged_segments[-1]["orient"] == segment["orient"]:
            # 合并到前一个相同姿势段
            merged_segments[-1]["end_frame"] = segment["end_frame"]
            merged_segments[-1]["duration_sec"] = (merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1) / fps
            merged_segments[-1]["duration_frames"] = merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1
        else:
            merged_segments.append(segment)
    orientation_segments = merged_segments 

    frames_to_remove = set()

     # **最后一步检查：删除第一个小于 15 秒的片段**
    if orientation_segments and orientation_segments[0]["duration_frames"] < max_duration_frames:
        first_segment = orientation_segments[0]  # 先存储要删除的片段
        print(f"🗑 删除首个片段 (小于 {max_duration_sec} 秒): {first_segment}")
        frames_to_remove.update(range(first_segment["start_frame"], first_segment["end_frame"] + 1))
        orientation_segments.pop(0)  # 现在安全地删除

    # **删除最后一个小于 15 秒的片段**
    if orientation_segments and orientation_segments[-1]["duration_frames"] < max_duration_frames:
        last_segment = orientation_segments[-1]  # 先存储要删除的片段
        print(f"🗑 删除尾部片段 (小于 {max_duration_sec} 秒): {last_segment}")
        frames_to_remove.update(range(last_segment["start_frame"], last_segment["end_frame"] + 1))
        orientation_segments.pop(-1)  # 现在安全地删除


    # 🚀 过滤 updated_orientation 和 updated_body_height，删除指定帧
    orientation = [orient for i, orient in enumerate(orientation) if i not in frames_to_remove]
    body_height = [height for i, height in enumerate(body_height) if i not in frames_to_remove]
    head_y = [head_y for i, head_y in enumerate(head_y) if i not in frames_to_remove]

    if orientation_segments:
        new_segments = []
        prev_end_frame = 0
        
        for seg in orientation_segments:
            duration = seg["end_frame"] - seg["start_frame"]
            seg["start_frame"] = prev_end_frame
            seg["end_frame"] = prev_end_frame + duration
            prev_end_frame = seg["end_frame"] + 1
            new_segments.append(seg)
        
        orientation_segments = new_segments

    # 🚀 处理小于 max_duration_frames 的片段
    for i in range(1, len(orientation_segments) - 1):  # 避免访问超出范围
        segment = orientation_segments[i]

        if segment["duration_frames"] < max_duration_frames:
            prev_orient = orientation_segments[i - 1]["orient"]  # 前一个片段的姿势
            next_orient = orientation_segments[i + 1]["orient"]  # 后一个片段的姿势

            if prev_orient == next_orient:
                segment["orient"] = prev_orient  # 如果前后相同，设为这个姿势
            else:
                segment["orient"] = prev_orient  # 否则设为前一个片段的姿势
    
    # **合并相邻相同姿势**
    merged_segments = []
    for segment in orientation_segments:
        if merged_segments and merged_segments[-1]["orient"] == segment["orient"]:
            # 合并到前一个相同姿势段
            merged_segments[-1]["end_frame"] = segment["end_frame"]
            merged_segments[-1]["duration_sec"] = (merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1) / fps
            merged_segments[-1]["duration_frames"] = merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1
        else:
            merged_segments.append(segment)
    orientation_segments = merged_segments 
    
    
    return orientation_segments, orientation, body_height, head_y

def refine_orientation_segments_with_motion(orientation_segments, motion_states, fps=30, duration_sec=15):
    """
    细化姿势片段，基于 motion_state 进行二次分割，合并短片段，并合并相邻的相同姿势+motion_state。

    参数:
        orientation_segments: list[dict] - 姿势片段，每个包含 start_frame, end_frame, orientation
        motion_states: list[str] - 每一帧的 motion_state（'Static' 或 'Dynamic'）
        fps: int - 每秒的帧数，默认 30
        duration_sec: int - 最小合并阈值（小于该时间的片段会合并到后一个片段）

    返回:
        refined_segments: list[dict] - 细化后的姿势片段
    """
    min_duration_frames = fps * duration_sec  # 计算最小 15 秒对应的帧数

    refined_segments = []

    for segment in orientation_segments:
        start, end, orient = segment["start_frame"], segment["end_frame"], segment["orient"]
        motion_segment_list = []
        current_motion, motion_start = None, None

        # 遍历姿势片段内部的 motion_state
        for i in range(start, end + 1):
            motion = motion_states[i]

            if current_motion is None:
                current_motion, motion_start = motion, i
            elif motion != current_motion:
                motion_end = i - 1
                duration = motion_end - motion_start + 1
                motion_segment_list.append({
                    "orient": orient,
                    "motion_state": current_motion,
                    "start_frame": motion_start,
                    "end_frame": motion_end,
                    "duration_sec": duration / fps,
                    "duration_frames": duration
                })
                current_motion, motion_start = motion, i

        # 记录最后一个 motion_state 片段
        if current_motion is not None:
            motion_end = end
            duration = motion_end - motion_start + 1
            motion_segment_list.append({
                "orient": orient,
                "motion_state": current_motion,
                "start_frame": motion_start,
                "end_frame": motion_end,
                "duration_sec": duration / fps,
                "duration_frames": duration
            })

        # **循环合并短片段**
        while True:
            merged_segments = []
            merged = False
            i = 0
            while i < len(motion_segment_list):
                if i > 0 and motion_segment_list[i]["duration_frames"] < min_duration_frames:
                    # **合并短片段到前一个片段**
                    merged_segments[-1]["end_frame"] = motion_segment_list[i]["end_frame"]
                    merged_segments[-1]["duration_sec"] = (
                        merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1
                    ) / fps
                    merged_segments[-1]["duration_frames"] = (
                        merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1
                    )
                    merged = True
                else:
                    merged_segments.append(motion_segment_list[i])
                i += 1

            motion_segment_list = merged_segments

            # **如果没有发生合并，则停止循环**
            if not merged:
                break

        # **合并第一个片段到后面，而不是删除**
        if len(motion_segment_list) > 1 and motion_segment_list[0]["duration_frames"] < min_duration_frames:
            print(f"🔄 合并第一个短片段到后一个: {motion_segment_list[0]}")
            motion_segment_list[1]["start_frame"] = motion_segment_list[0]["start_frame"]
            motion_segment_list[1]["duration_sec"] = (
                motion_segment_list[1]["end_frame"] - motion_segment_list[1]["start_frame"] + 1
            ) / fps
            motion_segment_list[1]["duration_frames"] = (
                motion_segment_list[1]["end_frame"] - motion_segment_list[1]["start_frame"] + 1
            )
            motion_segment_list.pop(0)  # 删除第一个片段（已合并）

        refined_segments.extend(motion_segment_list)

    # **合并相邻相同姿势+motion_state**
    final_segments = []
    for segment in refined_segments:
        if final_segments and (
            final_segments[-1]["orient"] == segment["orient"]
            and final_segments[-1]["motion_state"] == segment["motion_state"]
        ):
            # **合并到前一个相同姿势+motion_state 片段**
            final_segments[-1]["end_frame"] = segment["end_frame"]
            final_segments[-1]["duration_sec"] = (
                final_segments[-1]["end_frame"] - final_segments[-1]["start_frame"] + 1
            ) / fps
            final_segments[-1]["duration_frames"] = (
                final_segments[-1]["end_frame"] - final_segments[-1]["start_frame"] + 1
            )
        else:
            final_segments.append(segment)

    return final_segments

def split_head_y_by_orientation(orientation_segments, head_y):
    """
    根据 orientation_segments 的 start_frame 和 end_frame，分割 head_y。
    :param orientation_segments: 包含方向信息的字典列表，每个字典有 start_frame 和 end_frame。
    :param head_y: 包含头部 Y 坐标的数组。
    :return: 分割后的 head_y 片段列表。
    """
    segmented_head_y = []
    
    for segment in orientation_segments:
        start = segment['start_frame']
        end = segment['end_frame'] + 1  # 包含 end_frame 所在的索引
        head_y_segment = head_y[start:end]
        
        segmented_head_y.append(head_y_segment)
    
    return segmented_head_y


    """
    绘制 head_y 数组中 Y 轴高度的变化折线图。
    :param head_y: 每个元素是一个数值，表示某个时间点的中心高度。
    :param orientation_durations: 姿势段的时间范围，包含 start_frame 和 end_frame。
    """
    if not head_y:
        print("错误: head_y 为空，无法绘制图表。")
        return
    
    if not orientation_durations:
        print("错误: orientation_durations 为空，无法确定绘制区间。")
        return
    
    # 计算姿势的整体时间区间
    start_frame = min(seg["start_frame"] for seg in orientation_durations)
    end_frame = max(seg["end_frame"] for seg in orientation_durations)
    
    # 确保索引在合理范围内
    start_frame = max(0, start_frame)
    end_frame = min(len(head_y) - 1, end_frame)
    
    # 替换 None 值为 NaN，以保持数据长度一致
    filtered_head_y = [head_y[i] if head_y[i] is not None else np.nan for i in range(start_frame, end_frame + 1)]
    
    # 线性插值填充 None（可选，如果你希望图表更加平滑）
    filtered_head_y = pd.Series(filtered_head_y).interpolate(method='linear').tolist()
    
    # 生成 x 轴数据（时间步）
    x_values = np.arange(start_frame, end_frame + 1)  # 确保 x 轴数据长度一致
    
    # 绘制折线图
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, filtered_head_y, marker='o', markersize=3, linestyle='-', color='b', label='Height Variation')
    plt.xlabel("Frame Index")
    plt.ylabel("Height (y-coordinate)")
    plt.title("Head Height Variation Over Time")
    plt.legend()
    plt.grid()
    plt.show()

def plot_combined_single_axis(head_y, orientation_durations):
    """在同一张图上绘制高度变化和姿势变化区域，并在Static片段覆盖交叉线"""

    if not head_y:
        print("错误: head_y 为空，无法绘制图表。")
        return
    
    if not orientation_durations:
        print("错误: orientation_durations 为空，无法确定绘制区间。")
        return
    
    start_frame = min(seg["start_frame"] for seg in orientation_durations)
    end_frame = max(seg["end_frame"] for seg in orientation_durations)
    
    start_frame = max(0, start_frame)
    end_frame = min(len(head_y) - 1, end_frame)
    
    filtered_head_y = [head_y[i] if head_y[i] is not None else np.nan for i in range(start_frame, end_frame + 1)]
    filtered_head_y = pd.Series(filtered_head_y).interpolate(method='linear').tolist()
    
    x_values = np.arange(start_frame, end_frame + 1)

    plt.figure(figsize=(10, 5))

    # 绘制 head_y 高度变化曲线
    plt.plot(x_values, filtered_head_y, marker='o', markersize=3, linestyle='-', color='b', label='Height Variation')

    # 定义姿势对应的高度
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

    # 颜色映射
    color_map = {
        'neutral': 'lightblue',
        'right': 'yellow',
        'up': 'lightgreen',
        'down': 'lightsalmon',
        'Invalid': 'lightgray',
        'down-neutral': 'lightyellow',
        'neutral-down': 'lightyellow',
        'up-neutral':'blueviolet',
        'neutral-up': 'blueviolet'
    }

    for entry in orientation_durations:
        start_time = entry["start_frame"]
        end_time = entry["end_frame"]
        Height = Height_map[entry["orient"]]

        # 填充颜色
        poly = plt.fill_between([start_time, end_time], 0, Height, 
                         color=color_map[entry["orient"]], alpha=0.5, 
                         label=entry["orient"] if entry["orient"] not in plt.gca().get_legend_handles_labels()[1] else "")

        # 确保 motion_state 存在并且是 'Static'，否则跳过
        if entry.get("motion_state") == "Static":
            plt.fill_between([start_time, end_time], 0, Height, 
                             facecolor='none', edgecolor='black', hatch='//', alpha=0.5)

    plt.xlabel("Frame Index")
    plt.ylabel("Height Level / Face orientation")
    plt.title("Head Height & Face orientation Variation Over Time")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def process_segmented_head_y(segmented_head_y, frame_window=400, max_timestamps=8, smooth_window=5, max_iterations=10):
    """
    处理 segmented_head_y，迭代检测突变点，分割数据，清理无效数据，并平滑断点。

    参数：
    - segmented_head_y (list of list): 时间序列数据（每个子数组是一个时间序列）
    - frame_window (int): 前后帧窗口（默认 400 帧）
    - max_timestamps (int): 突变点最大阈值，仅对 **中间** 突变点生效
    - smooth_window (int): 平滑窗口大小
    - max_iterations (int): 最大迭代次数，防止无限循环

    返回：
    - final_processed_data (list of list): 处理后的分割数据
    - final_split_info (list): 记录 `segmented_head_y` 的第几个元素被分割几次
    """

    # 初始输入
    processed_data = segmented_head_y
    split_info = list(range(len(segmented_head_y)))  # 记录初始索引
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        new_processed_data = []
        new_split_info = []

        has_changes = False  # 追踪是否有新的分割或去除无效片段

        for idx, segment in zip(split_info, processed_data):
            if len(segment) < 2 * frame_window:  # 数据太短则跳过
                new_processed_data.append(segment)
                new_split_info.append(idx)
                continue

            segment = np.array(segment, dtype=float)

            # ✅ 1. 计算阈值（自适应计算）
            threshold_1 = compute_adaptive_threshold(segment, "std", 2)
            threshold_2 = compute_adaptive_threshold(segment, "std", 1)

            # ✅ 2. 检测突变点
            algo = rpt.Pelt(model="l2").fit(segment)
            change_points = algo.predict(pen=3)  # 获取突变点

            # ✅ 3. 处理突变点
            invalid_indices = set()
            timestamps = []
            middle_timestamps = []  # 存储中间部分的突变点（排除前400和后400）

            for cp in change_points:
                if cp < frame_window:  # 在前400帧
                    before_mean = np.mean(segment[:cp])
                    after_mean = np.mean(segment[cp:cp + frame_window])
                    if abs(after_mean - before_mean) > threshold_2:
                        invalid_indices.update(range(0, cp + frame_window))

                elif cp > len(segment) - frame_window:  # 在后400帧
                    before_mean = np.mean(segment[cp - frame_window:cp])
                    after_mean = np.mean(segment[cp:])
                    if abs(after_mean - before_mean) > threshold_2:
                        invalid_indices.update(range(cp - frame_window, len(segment)))

                else:  # 在中间部分
                    before_mean = np.mean(segment[cp - frame_window:cp])
                    after_mean = np.mean(segment[cp:cp + frame_window])
                    if abs(after_mean - before_mean) > threshold_2:
                        timestamps.append(cp)  # 添加定位戳
                        middle_timestamps.append(cp)  # 只把中间的突变点存入

            # ✅ 4. 处理中间的突变点数
            if len(middle_timestamps) > max_timestamps:
                print(f"Skipping segment {idx} due to too many middle timestamps: {len(middle_timestamps)}")
                timestamps = []  # 清空时间戳，防止分割

            # ✅ 5. 处理相邻突变点 (更新 timestamps 并标记无效数据)
            new_timestamps = []
            last_cp = None

            for cp in timestamps:
                if last_cp is not None and (cp - last_cp) < frame_window:
                    invalid_indices.update(range(last_cp, cp))  # 标记这段数据为无效
                else:
                    new_timestamps.append(cp)
                    last_cp = cp
            timestamps = new_timestamps  # 更新 timestamps

            # ✅ 6. 去除无效数据
            valid_indices = sorted(set(range(len(segment))) - invalid_indices)
            filtered_segment = segment[valid_indices]  # 只保留有效数据

            if len(valid_indices) < len(segment):  # 数据被修改了
                has_changes = True

            # ✅ 7. 分割数据
            split_segments = []
            last_cp = 0
            for cp in timestamps:
                if cp - last_cp > 10:  # 避免片段过短
                    split_segments.append(filtered_segment[last_cp:cp])  # 使用 filtered_segment
                    new_split_info.append(idx)  # 记录分割信息
                last_cp = cp

            if last_cp < len(filtered_segment):  # 添加最后一个片段
                split_segments.append(filtered_segment[last_cp:])
                new_split_info.append(idx)

            if len(split_segments) > 1:  # 发生了分割
                has_changes = True

            # ✅ 8. 平滑断点
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
    使用 ACF 计算时间序列是否具有周期性，并找到周期的最高峰值和最低峰值
    :param data: 时间序列数据
    :param threshold: 自相关系数阈值，绝对值大于此值才算显著相关
    :param max_lag: 计算 ACF 时的最大滞后步长
    :param min_ratio: 多少比例的滞后值需要超过 threshold 才算周期性
    :param min_alternations: 至少多少次正负交替才算周期性
    :param plot: 是否绘制 ACF 图
    :return: (是否存在周期性, 最高峰值 (lag, ACF), 最低峰值 (lag, ACF))
    """

    # 计算 ACF
    acf_values = sm.tsa.acf(data, nlags=max_lag)

    # 统计 |ACF| 超过 threshold 的滞后点数量
    above_threshold = np.sum(np.abs(acf_values[1:]) > threshold)
    ratio = above_threshold / max_lag  # 计算占比

    # 计算 ACF 的符号变化 (正负交替)
    sign_changes = np.sign(acf_values[1:])  # 获取 ACF 的正负号 (+1 or -1)
    alternation_count = np.sum(np.diff(sign_changes) != 0)  # 计算正负交替次数

    # 确保：
    # 1. 绝对值超过 threshold 的比例足够大
    # 2. 至少有 min_alternations 组正负交替
    periodic = (ratio > min_ratio) and (alternation_count >= min_alternations)

    # 计算均值和标准差
    mean = np.mean(data)
    amp = compute_amplitude_fft(data)

    return periodic, mean, amp

def split_orientation_segments(orientation_segments, segmented_head_y, split_info):
    """
    根据 segmented_head_y 和 split_info 对 orientation_segments 进行相应的分割，并按比例分配帧数。

    :param orientation_segments: 原始的 orientation 片段列表
    :param segmented_head_y: 分割后的时间序列数据，每个子数组对应一个分割部分
    :param split_info: 指示 segmented_head_y 每个元素属于哪个 orientation 片段
    :return: 重新分割后的 orientation 片段列表
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
        segment_index = split_info[i]  # 该数据片段属于哪个原 orientation 片段
        orig_segment = orientation_segments[segment_index]  # 获取原始 orientation 片段

        # 获取原始片段的信息
        orig_start_frame = orig_segment["start_frame"]
        orig_duration_frames = orig_segment["duration_frames"]

        # 按比例计算新的 duration_frames
        total_segment_length = segment_lengths[segment_index]  # 该片段所有的 segmented_head_y 数据总长度
        ratio = len(segment_data) / total_segment_length
        new_duration_frames = round(ratio * orig_duration_frames)

        # 确保片段是连续的
        if segment_index not in segment_allocations:
            segment_allocations[segment_index] = orig_start_frame
        start_frame = segment_allocations[segment_index]
        end_frame = start_frame + new_duration_frames - 1

        # 计算帧率 (FPS) 以转换 duration_frames -> duration_sec
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

        # 更新起始位置
        segment_allocations[segment_index] = end_frame + 1

    return new_segments

def compute_amplitude_fft(time_series):
    """
    计算主频及其对应的振幅
    :param time_series: 输入的时间序列数据
    :param sampling_rate: 采样率（Hz）
    :return: (主频, 该主频的振幅)
    """
    N = len(time_series)  # 数据长度
    fft_values = np.fft.fft(time_series)  # 计算FFT

    # 计算单边振幅谱
    amplitude_spectrum = (2 / N) * np.abs(fft_values)  # 归一化

    # 取正频率部分
    positive_amplitude = amplitude_spectrum[:N // 2]

    # 找到主频索引（忽略零频）
    main_freq_index = np.argmax(positive_amplitude[1:]) + 1  # 跳过直流分量
    main_amplitude = positive_amplitude[main_freq_index]

    return main_amplitude

def update_orientation_segments(orientation_segments, periodics, means, amps):
    """
    根据 periodics、means 和 amps 更新 orientation_segments，添加 head_y 值：
    - 若 periodics[i] 为 True，则 head_y = means[i]
    - 若 periodics[i] 为 False，则 head_y = [means[i] - amps[i], means[i] + amps[i]]

    :param orientation_segments: 更新后的 orientation 片段列表
    :param periodics: 是否存在周期性 (True / False)
    :param means: 每个片段的均值
    :param amps: 每个片段的振幅
    :return: 包含 head_y 的更新 orientation_segments
    """

    for i in range(len(orientation_segments)):
        if periodics[i]:
            orientation_segments[i]["head_y"] = [means[i] - amps[i], means[i] + amps[i]]  # 设定区间
        else:
            orientation_segments[i]["head_y"] = means[i]  # 直接赋值

    return orientation_segments

def plot_orientation_segments(orientation_segments):
    """
    在同一张图上绘制 head_y 变化（基于 orientation_segments["head_y"]）和姿势变化区域，
    并在 Static 片段上覆盖交叉线，同时标注 orient 方向。
    片段之间的断点将被连接以形成连续曲线。

    :param orientation_segments: 包含 start_frame, end_frame, head_y, orient 等信息的字典列表
    """
    if not orientation_segments:
        print("错误: orientation_segments 为空，无法绘制图表。")
        return
    
    plt.figure(figsize=(12, 6))

    # 颜色映射
    color_map = {
        'neutral': 'lightblue',
        'right': 'yellow',
        'up': 'lightgreen',
        'down': 'lightsalmon',
        'Invalid': 'lightgray',
        'down-neutral': 'yellow',
        'neutral-down': 'yellow',
        'up-neutral': 'blueviolet',
        'neutral-up': 'blueviolet'
    }

    previous_end_frame = None  # 记录前一个片段的 end_frame
    previous_y = None  # 记录前一个片段的最后一个 y 值

    # 遍历 orientation_segments，绘制 head_y 轨迹
    for entry in orientation_segments:
        start_time = entry["start_frame"]
        end_time = entry["end_frame"]
        head_y = entry["head_y"]
        orient = entry["orient"]

        # 确定颜色
        color = color_map.get(orient, 'gray')

        # 生成 x 轴数据
        x_values = np.arange(start_time, end_time + 1)

        # 生成 y 轴数据
        if isinstance(head_y, (int, float)):  # 单值，绘制水平直线
            y_values = np.full_like(x_values, head_y, dtype=float)
        elif isinstance(head_y, (list, tuple)) and len(head_y) == 2:  # 区间值，绘制振荡曲线
            min_val, max_val = head_y
            amplitude = (max_val - min_val) / 2  # 振幅
            mean_val = (max_val + min_val) / 2   # 均值
            y_values = mean_val + amplitude * np.sin(2 * np.pi * np.linspace(0, 2, len(x_values)))  # 振荡
        else:
            continue  # 数据格式错误，跳过

        # **解决片段之间的断点问题**
        if previous_end_frame is not None and previous_end_frame + 1 < start_time:
            # 连接前一个片段的最后一个点和当前片段的第一个点
            plt.plot([previous_end_frame, start_time], [previous_y, y_values[0]], linestyle='-', color=color, alpha=0.6)

        # 画线
        plt.plot(x_values, y_values, linestyle='-', marker='', color=color, label=orient if orient not in plt.gca().get_legend_handles_labels()[1] else "")

        # 记录当前片段的结束位置
        previous_end_frame = end_time
        previous_y = y_values[-1]  # 记录最后一个 y 值

        # 在 orientation 片段顶部标注 orient
        mid_x = (start_time + end_time) / 2
        mid_y = max(y_values) + 0.01  # 让文本稍微高于曲线
        plt.text(mid_x, mid_y, orient, fontsize=10, ha='center', va='bottom', color='black', fontweight='bold')


    plt.xlabel("Frame Index")
    plt.ylabel("Head Y Value / Orientation")
    plt.title("Head Y Variation & Orientation Over Time")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# ========================= 6. 运行主程序 =========================
if __name__ == "__main__":

    filename="output_data11.json"
    fps, people_counts, body_height, orientation, head_y, motion_states = load_json_data(filename)
    

    people_counts, orientation, motion_states = smooth_stable_data(people_counts, orientation, motion_states)
    orientation_segments, orientation, body_height, head_y = filter_invalid_orientation_segments(orientation, body_height, head_y, fps)
    body_height = smooth_body_height(body_height, fps)
    change_points = detect_change_points(body_height, visualize=False)
    orientation_segments, orientation, body_height, head_y = remove_large_height_changes(change_points, orientation_segments, orientation, body_height, head_y, fps)
    orientation_segments = merge_RL_orientation_segments(orientation_segments, fps)
    orientation_segments = merge_alternating_orients(orientation_segments, fps)
    orientation_segments, orientation, body_height, head_y = merge_orientation_segments(orientation_segments, orientation, body_height, head_y, fps)
    segmented_head_y = split_head_y_by_orientation(orientation_segments, head_y)
    segmented_head_y, split_info = process_segmented_head_y(segmented_head_y)
    print(split_info)

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
    print(periodics)
    orientation_segments = update_orientation_segments(orientation_segments, periodics, means, amps)

    plot_orientation_segments(orientation_segments)
    print(orientation_segments)
    
    
    # plot_combined_single_axis(head_y, orientation_segments)

    