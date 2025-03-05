import json
import numpy as np
from collections import deque, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ========================= 1. 读取 JSON 数据 =========================
def load_json_data(filename="output_data5.json"):
    """读取 JSON 文件并解析数据"""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"错误: 无法读取 {filename}，请检查文件路径或格式。")
        return [], [], [], []
    
    # 解析 JSON 数据
    people_counts = [frame.get("people_count", 0) for frame in data]
    person_sizes = [frame.get("person_size") for frame in data]  # 允许 None
    max_distances = [frame.get("max_distance", 0.0) for frame in data]
    postures = [frame.get("posture") for frame in data]  #
    print(len(people_counts))

    return people_counts, person_sizes, max_distances, postures


# ========================= 2. 计算 Significant Changes =========================
def compute_change_threshold(person_sizes, window_size=10, skip_threshold=0.1):
    """计算 person_size 的变化阈值"""
    valid_sizes = np.array([s for s in person_sizes if s is not None])
    if len(valid_sizes) < 2:
        return None, None

    rolling_avg = np.convolve(valid_sizes, np.ones(window_size) / window_size, mode='valid')
    relative_changes = np.abs(np.diff(rolling_avg) / rolling_avg[:-1])
    
    # 过滤掉超过 60% 变化的异常值
    filtered_changes = relative_changes[relative_changes <= 0.6]
    median_change, max_change = np.median(filtered_changes), np.max(filtered_changes)
    
    if np.abs(median_change - max_change) < skip_threshold:
        first_valid_frame = next((i for i, s in enumerate(person_sizes) if s is not None), None)
        return None, [first_valid_frame] if first_valid_frame is not None else []

    print(median_change)
    print(max_change)
    return (median_change + max_change) / 2, None

def identify_significant_changes(person_sizes, window_size=10, fps=30, skip_seconds=60):
    """检测 person_size 变化剧烈的帧索引"""
    change_threshold, preset_large_change_frames = compute_change_threshold(person_sizes, window_size)

    # 如果 `compute_change_threshold()` 直接给了预设的显著变化帧，就直接返回
    if preset_large_change_frames is not None:
        return preset_large_change_frames, change_threshold

    rolling_window = deque(maxlen=window_size)
    prev_avg_size, large_change_frames = None, []
    skip_frames = fps * skip_seconds
    check_range = max(0, len(person_sizes) - skip_frames)

    first_valid_frame = next((i for i, s in enumerate(person_sizes) if s is not None), None)

    for i in range(check_range):
        if person_sizes[i] is None:
            continue

        rolling_window.append(person_sizes[i])
        if len(rolling_window) < window_size:
            continue

        current_avg_size = np.mean(rolling_window)
        if prev_avg_size and abs(current_avg_size - prev_avg_size) / prev_avg_size > change_threshold:
            large_change_frames.append(i)

        prev_avg_size = current_avg_size

    # **如果 large_change_frames 为空，就使用 first_valid_frame**
    if not large_change_frames and first_valid_frame is not None:
        large_change_frames.append(first_valid_frame)

    return large_change_frames, change_threshold



# ========================= 3. 过滤稳定数据 =========================
def filter_stable_data(people_counts, postures, last_significant_frame, window_size=10, consensus_ratio=0.8):
    """平滑数据，移除噪音"""
    filtered_people_counts, filtered_postures = [], []
    
    for i in range(last_significant_frame, len(people_counts)):
        if postures[i] == 'Not classified':
            continue  

        start, end = max(0, i - window_size), min(len(people_counts), i + window_size)
        most_common_people = max(set(people_counts[start:end]), key=people_counts[start:end].count)
        most_common_posture = max(set(postures[start:end]), key=postures[start:end].count)
        
        people_consensus = people_counts[start:end].count(most_common_people) / (end - start)
        posture_consensus = postures[start:end].count(most_common_posture) / (end - start)
        
        filtered_people_counts.append(most_common_people if people_consensus >= consensus_ratio else people_counts[i])
        filtered_postures.append(most_common_posture if posture_consensus >= consensus_ratio else postures[i])
    
    return filtered_people_counts, filtered_postures


# ========================= 4. 计算 Posture Durations =========================
def analyze_posture_durations(people_counts, postures, fps=30, duration_sec=20):
    """只保留 >= duration_sec 的姿势，并合并相邻相同的姿势"""
    min_duration_frames = duration_sec * fps
    posture_segments = []
    current_posture, start_frame = None, None

    for i in range(len(postures)):
        if people_counts[i] != 1:
            continue

        if current_posture is None:
            current_posture, start_frame = postures[i], i
        elif postures[i] != current_posture:
            end_frame, duration = i - 1, i - start_frame
            if duration >= min_duration_frames:
                posture_segments.append({
                    "posture": current_posture, "start_frame": start_frame, "end_frame": end_frame,
                    "duration_sec": duration / fps
                })
            current_posture, start_frame = postures[i], i

    if current_posture:
        end_frame, duration = len(postures) - 1, len(postures) - start_frame
        if duration >= min_duration_frames:
            posture_segments.append({
                "posture": current_posture, "start_frame": start_frame, "end_frame": end_frame,
                "duration_sec": duration / fps
            })
    
    # 合并相邻的相同姿势
    merged = True
    while merged:
        merged, new_segments, i = False, [], 0
        while i < len(posture_segments):
            if i < len(posture_segments) - 1 and posture_segments[i]["posture"] == posture_segments[i + 1]["posture"]:
                merged = True
                new_segments.append({
                    "posture": posture_segments[i]["posture"],
                    "start_frame": posture_segments[i]["start_frame"],
                    "end_frame": posture_segments[i + 1]["end_frame"],
                    "duration_sec": (posture_segments[i + 1]["end_frame"] - posture_segments[i]["start_frame"] + 1) / fps
                })
                i += 2
            else:
                new_segments.append(posture_segments[i])
                i += 1
        posture_segments = new_segments

    return posture_segments


# ========================= 5. 绘图方法 =========================
def plot_posture_durations(posture_durations):
    """绘制姿势变化的时间段图"""
    color_map = {'Standing': 'blue', 'Prone': 'red', 'Supine': 'green'}
    fig, ax = plt.subplots(figsize=(12, 6))

    for data in posture_durations:
        ax.barh(y=data['posture'], width=data['end_frame']-data['start_frame'], 
                left=data['start_frame'], height=0.4, 
                color=color_map.get(data['posture'], 'gray'), edgecolor='black')

    ax.legend(handles=[mpatches.Patch(color=color, label=label) for label, color in color_map.items()], title="Postures")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Posture")
    ax.set_title("Posture Transitions Over Time")

    plt.show()


# ========================= 6. 运行主程序 =========================
if __name__ == "__main__":
    people_counts, person_sizes, max_distances, postures = load_json_data()
    large_change_frames, optimal_threshold = identify_significant_changes(person_sizes)

    if large_change_frames:
        filtered_people_counts, filtered_postures = filter_stable_data(people_counts, postures, large_change_frames[-1])
        posture_durations = analyze_posture_durations(filtered_people_counts, filtered_postures)
        plot_posture_durations(posture_durations)

    print("自动计算的阈值:", optimal_threshold)
    print("剧烈变化发生在帧:", large_change_frames)
    print(posture_durations)