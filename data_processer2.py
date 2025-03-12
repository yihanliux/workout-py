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


    
