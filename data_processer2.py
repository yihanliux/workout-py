import json
import numpy as np
import pandas as pd
from collections import deque, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ========================= 1. 读取 JSON 数据 =========================
def load_json_data(filename="output_data8.json"):
    """读取 JSON 文件并解析数据"""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"错误: 无法读取 {filename}，请检查文件路径或格式。")
        return 0, [], [], [], []
    
    # 解析 JSON 数据
    fps = data.get("fps", 30)  # 读取 fps，默认 30
    frames = data.get("frames", [])  # 读取帧数据列表
    
    people_counts = [frame.get("people_count") for frame in frames]
    body_height = [frame.get("body_height") for frame in frames]  
    orientation = [frame.get("orientation") for frame in frames] 
    head_y = [frame.get("head_y") for frame in frames]  
    
    print("视频总共有", len(people_counts), "帧，帧率:", fps, "FPS")
    
    return fps, people_counts, body_height, orientation, head_y


def filter_stable_data(people_counts, orientation, window_size=10, consensus_ratio=0.8):
    """
    平滑数据，移除噪音，同时保留所有数据。
    使 `people_counts`, `orientation` 更稳定。
    
    参数:
        people_counts: list[int] - 每一帧的人数数据
        orientation list[str] - 每一帧的面部朝向
        window_size: int - 滑动窗口大小
        consensus_ratio: float - 认定最常见值的比例 (默认 80%)

    返回:
        filtered_people_counts, filtered_orientation
    """
    filtered_people_counts = people_counts[:]
    filtered_orientation = orientation[:]

    for i in range(len(people_counts)):
        start, end = max(0, i - window_size), min(len(people_counts), i + window_size)

        # 计算滑动窗口内的最常见值
        most_common_people = max(set(people_counts[start:end]), key=people_counts[start:end].count)
        most_common_orientation = max(set(orientation[start:end]), key=orientation[start:end].count)

        # 计算最常见值的占比
        people_consensus = people_counts[start:end].count(most_common_people) / (end - start)
        orientation_consensus = orientation[start:end].count(most_common_orientation) / (end - start)

        # 如果最常见值的比例超过 `consensus_ratio`，就采用它，否则保持原值
        filtered_people_counts[i] = most_common_people if people_consensus >= consensus_ratio else people_counts[i]
        filtered_orientation[i] = most_common_orientation if orientation_consensus >= consensus_ratio else orientation[i]
    
    return filtered_people_counts, filtered_orientation


def analyze_orientation_durations(people_counts, orientation, body_height, fps=30, duration_sec=15, factor=1.3):
    """
    1. 记录所有 people_counts == 1 的连续间隔
    2. 合并短于 fps * duration_sec 的区间到前一个姿势
    3. 合并相邻的相同姿势
    4. 删除最后一个 'Invalid' 片段
    5. 计算每个片段的 body_height 方差，删除异常片段
    """
    min_duration_frames = fps * duration_sec
    orient_segments = []
    current_orient, start_frame = None, None

    # 预处理，将无效姿势转换为 "Invalid"
    orientation = [
        'Invalid' if orient is None else orient
        for orient in orientation
    ]

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

    # 处理最后一个片段如果是 None，则尝试删除无效片段
    if orient_segments and orient_segments[-1]["orient"] == "Invalid":
        new_segments = orient_segments[:-1]  # 复制去掉最后一个 None 的片段
        last_invalid_segment = orient_segments[-1]

        # 删除所有持续时间小于 min_duration_frames 的片段
        while new_segments:
            last_segment = new_segments[-1]
            if last_segment["duration_frames"] >= min_duration_frames:
                break  
            new_segments.pop()  # 移除已合并片段
        orient_segments = new_segments

    final_segments = orient_segments[:]  # 先复制 orient_segments
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

    # **合并相邻相同姿势**
    merged_segments = []
    for segment in final_segments:
        if merged_segments and merged_segments[-1]["orient"] == segment["orient"]:
            # 合并到前一个相同姿势段
            merged_segments[-1]["end_frame"] = segment["end_frame"]
            merged_segments[-1]["duration_sec"] = (merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1) / fps
            merged_segments[-1]["duration_frames"] = merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1
        else:
            merged_segments.append(segment)

    # **计算每个片段的 body_height 均值和方差**
    segment_stats = []
    for idx, segment in enumerate(merged_segments):
        start, end = segment["start_frame"], segment["end_frame"]
        values = [body_height[i] for i in range(start, end + 1) if body_height[i] is not None]

        if values:
            mean_value = np.mean(values)
            variance = np.var(values)
            segment_stats.append((segment, mean_value, variance))
            print(f"  片段 {idx+1} ({segment['orient']}): 均值 = {mean_value:.2f}, 方差 = {variance:.2f}")

    if not segment_stats:
        print("⚠️ 没有有效的 `body_height` 数据，返回原数据")
        return merged_segments  # 没有有效数据，返回原数据

    # 获取首个片段
    first_segment, first_mean, first_var = segment_stats[0]

    # 找出均值和方差的最高值
    max_mean_segment = max(segment_stats, key=lambda x: x[1])
    max_var_segment = max(segment_stats, key=lambda x: x[2])

    # 找出第二高的均值和方差片段（如果有多个片段）
    second_max_mean = sorted(segment_stats, key=lambda x: x[1], reverse=True)[1] if len(segment_stats) > 1 else None
    second_max_var = sorted(segment_stats, key=lambda x: x[2], reverse=True)[1] if len(segment_stats) > 1 else None

     # 判断是否删除第一个片段
    deleted_reason = None
    if second_max_mean and first_mean > second_max_mean[1] * factor:  # 均值明显最高
        deleted_reason = f"均值最高 ({first_mean:.2f})，且比第二高 ({second_max_mean[1]:.2f}) 高 {factor * 100 - 100:.0f}% 以上"
    elif second_max_var and first_var > second_max_var[2] * factor:  # 方差明显最高
        deleted_reason = f"方差最高 ({first_var:.2f})，且比第二高 ({second_max_var[2]:.2f}) 高 {factor * 100 - 100:.0f}% 以上"

    # 如果第一个片段需要删除，打印原因并删除
    if deleted_reason:
        print(f"\n❌ 删除首个片段（{deleted_reason}）")
        return merged_segments[1:]  # 移除第一个片段
    
    # **最后一步检查：删除第一个小于 15 秒的片段**
    if merged_segments and merged_segments[0]["duration_frames"] < min_duration_frames:
        print(f"🗑 删除首个片段 (小于 {duration_sec} 秒): {merged_segments[0]}")
        merged_segments.pop(0)

    # **最后一步检查：删除最后一个 'Invalid' 片段**
    if merged_segments and merged_segments[-1]["orient"] == "Invalid":
        print(f"🗑 删除最后一个 'Invalid' 片段: {merged_segments[-1]}")
        merged_segments.pop(-1)
    
    # **最后一步检查：删除第一个 'Invalid' 片段**
    if merged_segments and merged_segments[0]["orient"] == "Invalid":
        print(f"🗑 删除最后一个 'Invalid' 片段: {merged_segments[-1]}")
        merged_segments.pop(-1)

    return merged_segments

# ========================= 5. 绘图方法 =========================
def plot_orientation_durations(orientation_durations):
    """绘制姿势变化的时间段图（X轴使用帧索引）"""

    # 定义姿势对应的高度
    Height_map = {
        'neutral': 2,
        'up': 3,
        'down': 1,
        'Invalid':0
    }

    # 颜色映射
    color_map = {
        'neutral': 'lightblue',
        'up': 'lightgreen',
        'down': 'lightsalmon',
        'Invalid': 'lightgray'
    }

    # 绘制折线图并填充颜色
    plt.figure(figsize=(10, 5))

    for entry in orientation_durations:
        start_time = entry["start_frame"]  # 直接使用帧索引
        end_time = entry["end_frame"]
        Height = Height_map[entry["orient"]]
        
        # 绘制水平线表示姿势持续时间
        plt.plot([start_time, end_time], [Height, Height], 'k-', linewidth=1)
        
        # 填充颜色
        plt.fill_between(
            [start_time, end_time], 0, Height, 
            color=color_map[entry["orient"]], alpha=0.5, 
            label=entry["orient"] if entry["orient"] not in plt.gca().get_legend_handles_labels()[1] else ""
        )

    # 图表细节
    plt.xlabel("Frame Index")  # X 轴单位改为帧索引
    plt.ylabel("Height Level")
    plt.title("Orient Over Time (Frame-based)")
    plt.yticks([1, 2, 3], ['Low', 'Medium', 'High'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 显示图表
    plt.show()


def plot_height_variation(head_y, orientation_durations):
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
    """在同一张图上绘制高度变化和姿势变化区域"""

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
        'up': 3,
        'down': 1,
        'Invalid':0
    }

    # 颜色映射
    color_map = {
        'neutral': 'lightblue',
        'up': 'lightgreen',
        'down': 'lightsalmon',
        'Invalid': 'lightgray'
    }

    for entry in orientation_durations:
        start_time = entry["start_frame"]
        end_time = entry["end_frame"]
        Height = Height_map[entry["orient"]]

        plt.plot([start_time, end_time], [Height, Height], 'k-', linewidth=1)
        plt.fill_between([start_time, end_time], 0, Height, 
                         color=color_map[entry["orient"]], alpha=0.5, 
                         label=entry["orient"] if entry["orient"] not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.xlabel("Frame Index")
    plt.ylabel("Height Level / Y-coordinate")
    plt.title("Head Height & Posture Variation Over Time")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# ========================= 6. 运行主程序 =========================
if __name__ == "__main__":
    fps, people_counts, body_height, orientation, head_y = load_json_data()
    

    filtered_people_counts, filtered_orientation = filter_stable_data(people_counts, orientation)
    orientation_durations = analyze_orientation_durations(filtered_people_counts, filtered_orientation, body_height, fps)
    print(orientation_durations)
    # plot_orientation_durations(orientation_durations)
    # plot_height_variation(head_y, orientation_durations)
    #plot_combined_single_axis(head_y, orientation_durations)


    
def filter_invalid_orientation_segments(orient_segments, orientation, body_height, head_y, fps=30, min_duration_sec=3, max_duration_sec=90):
    """
    过滤和处理无效的方向片段。

    主要目标：
    1. **分割方向片段**：根据 `orientation` 变化记录不同的片段。
    2. **识别无效片段**：
       - 处理 `None` 值，将其转换为 `"Invalid"`，以保证数据完整性。
       - 找出所有 `"Invalid"` 片段，并标记前 10% 和后 10% 位置的无效段落。
    3. **清理无效片段**：
       - 删除持续时间小于 `min_duration_sec` 的 `"Invalid"` 片段。
       - 在 80% 中间区域内，合并相邻的 `"Invalid"` 片段，防止数据碎片化。
    4. **平滑 body_height 和 head_y**：
       - 计算 `body_height` 和 `head_y` 的全局中位数，用于填充 `None` 值。
       - 处理 `"Invalid"` 片段，使其与前后片段尽可能保持一致。
    
    参数:
        orientation (list[str]): 每一帧的面部朝向信息。
        body_height (list[float]): 每一帧的身体高度数据。
        head_y (list[float]): 每一帧的头部 Y 轴坐标。
        fps (int): 视频的帧率 (默认 30)。
        min_duration_sec (int): 片段的最短持续时间（秒），默认 3 秒。
        max_duration_sec (int): 片段的最长持续时间（秒），默认 90 秒。

    返回:
        tuple:
            - new_orient_segments (list[dict]): 处理后的方向片段数据。
            - updated_orientation (list[str]): 过滤后的面部朝向数据。
            - updated_body_height (list[float]): 过滤后的身体高度数据。
            - updated_head_y (list[float]): 过滤后的头部 Y 轴数据。
    """
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
    
    # 确保 `updated_body_height` 是 dict，避免 list 访问越界问题
    if isinstance(updated_body_height, list):
        updated_body_height = {i: updated_body_height[i] for i in range(len(updated_body_height))}

    if isinstance(updated_head_y, list):
        updated_head_y = {i: updated_head_y[i] for i in range(len(updated_head_y))}

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