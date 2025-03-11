import json
import numpy as np
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
        return 0, [], [], [], [], []
    
    # 解析 JSON 数据
    fps = data.get("fps", 30)  # 读取 fps，默认 30
    frames = data.get("frames", [])  # 读取帧数据列表
    
    people_counts = [frame.get("people_count", 0) for frame in frames]
    body_height = [frame.get("body_height") for frame in frames]  # 允许 None
    postures = [frame.get("posture") for frame in frames]  #
    motion_states = [frame.get("motion_state") for frame in frames]  #
    weighted_center = [frame.get("weighted_center") for frame in frames]  #
    
    print("视频总共有", len(people_counts), "帧，帧率:", fps, "FPS")
    
    return fps, people_counts, body_height, postures, motion_states, weighted_center

# # ========================= 2. 计算 Significant Changes =========================
# def compute_change_threshold(person_sizes, window_size=10, skip_threshold=0.1):
#     """计算 person_size 的变化阈值"""
#     valid_sizes = np.array([s for s in person_sizes if s is not None])
#     if len(valid_sizes) < 2:
#         return None, None

#     rolling_avg = np.convolve(valid_sizes, np.ones(window_size) / window_size, mode='valid')
#     relative_changes = np.abs(np.diff(rolling_avg) / rolling_avg[:-1])
    
#     # 过滤掉超过 60% 变化的异常值
#     filtered_changes = relative_changes[relative_changes <= 0.6]
#     median_change, max_change = np.median(filtered_changes), np.max(filtered_changes)
    
#     if np.abs(median_change - max_change) < skip_threshold:
#         first_valid_frame = next((i for i, s in enumerate(person_sizes) if s is not None), None)
#         return None, [first_valid_frame] if first_valid_frame is not None else []

#     return (median_change + max_change) / 2, None

# def identify_significant_changes(person_sizes, window_size=10, fps=30, skip_seconds=60):
#     """检测 person_size 变化剧烈的帧索引"""
#     change_threshold, preset_large_change_frames = compute_change_threshold(person_sizes, window_size)

#     # 如果 `compute_change_threshold()` 直接给了预设的显著变化帧，就直接返回
#     if preset_large_change_frames is not None:
#         return preset_large_change_frames, change_threshold

#     rolling_window = deque(maxlen=window_size)
#     prev_avg_size, large_change_frames = None, []
#     skip_frames = fps * skip_seconds
#     check_range = max(0, len(person_sizes) - skip_frames)

#     first_valid_frame = next((i for i, s in enumerate(person_sizes) if s is not None), None)

#     for i in range(check_range):
#         if person_sizes[i] is None:
#             continue

#         rolling_window.append(person_sizes[i])
#         if len(rolling_window) < window_size:
#             continue

#         current_avg_size = np.mean(rolling_window)
#         if prev_avg_size and abs(current_avg_size - prev_avg_size) / prev_avg_size > change_threshold:
#             large_change_frames.append(i)

#         prev_avg_size = current_avg_size

#     # **如果 large_change_frames 为空，就使用 first_valid_frame**
#     if not large_change_frames and first_valid_frame is not None:
#         large_change_frames.append(first_valid_frame)

#     return large_change_frames, change_threshold

# ========================= 3. 过滤稳定数据 =========================
# def filter_stable_data_(people_counts, postures, last_significant_frame, window_size=10, consensus_ratio=0.8):
#     """平滑数据，移除噪音，同时保留所有数据，并将 last_significant_frame 之前的姿势设为 'Not motion'"""
#     filtered_people_counts = people_counts[:]
#     filtered_postures = ['Not motion'] * last_significant_frame + postures[last_significant_frame:]
    
#     for i in range(last_significant_frame, len(people_counts)):
#         if postures[i] == 'Not classified':
#             continue  
        
#         start, end = max(0, i - window_size), min(len(people_counts), i + window_size)
#         most_common_people = max(set(people_counts[start:end]), key=people_counts[start:end].count)
#         most_common_posture = max(set(postures[start:end]), key=postures[start:end].count)
        
#         people_consensus = people_counts[start:end].count(most_common_people) / (end - start)
#         posture_consensus = postures[start:end].count(most_common_posture) / (end - start)
        
#         filtered_people_counts[i] = most_common_people if people_consensus >= consensus_ratio else people_counts[i]
#         filtered_postures[i] = most_common_posture if posture_consensus >= consensus_ratio else postures[i]
    
#     return filtered_people_counts, filtered_postures

def filter_stable_data(people_counts, postures, motion_states, window_size=10, consensus_ratio=0.8):
    """
    平滑数据，移除噪音，同时保留所有数据。
    使 `people_counts`, `postures` 和 `motion_states` 更稳定。
    
    参数:
        people_counts: list[int] - 每一帧的人数数据
        postures: list[str] - 每一帧的姿势 ('Standing', 'Sitting', ...)
        motion_states: list[str] - 每一帧的运动状态 ('static' 或 'dynamic')
        window_size: int - 滑动窗口大小
        consensus_ratio: float - 认定最常见值的比例 (默认 80%)

    返回:
        filtered_people_counts, filtered_postures, filtered_motion_states
    """
    filtered_people_counts = people_counts[:]
    filtered_postures = postures[:]
    filtered_motion_states = motion_states[:]

    for i in range(len(people_counts)):
        start, end = max(0, i - window_size), min(len(people_counts), i + window_size)

        # 计算滑动窗口内的最常见值
        most_common_people = max(set(people_counts[start:end]), key=people_counts[start:end].count)
        most_common_posture = max(set(postures[start:end]), key=postures[start:end].count)
        most_common_motion = max(set(motion_states[start:end]), key=motion_states[start:end].count)

        # 计算最常见值的占比
        people_consensus = people_counts[start:end].count(most_common_people) / (end - start)
        posture_consensus = postures[start:end].count(most_common_posture) / (end - start)
        motion_consensus = motion_states[start:end].count(most_common_motion) / (end - start)

        # 如果最常见值的比例超过 `consensus_ratio`，就采用它，否则保持原值
        filtered_people_counts[i] = most_common_people if people_consensus >= consensus_ratio else people_counts[i]
        filtered_postures[i] = most_common_posture if posture_consensus >= consensus_ratio else postures[i]
        filtered_motion_states[i] = most_common_motion if motion_consensus >= consensus_ratio else motion_states[i]

    return filtered_people_counts, filtered_postures, filtered_motion_states


def analyze_posture_durations(people_counts, postures, body_height, fps=30, duration_sec=15, factor=1.3):
    """
    1. 记录所有 people_counts == 1 的连续间隔
    2. 合并短于 fps * duration_sec 的区间到前一个姿势
    3. 合并相邻的相同姿势
    4. 删除最后一个 'Invalid' 片段
    5. 计算每个片段的 body_height 方差，删除异常片段
    """
    min_duration_frames = fps * duration_sec
    posture_segments = []
    current_posture, start_frame = None, None

    # 预处理，将无效姿势转换为 "Invalid"
    postures = [
        'Invalid' if count != 1 or posture in ['No Person', 'Not classified'] else posture
        for count, posture in zip(people_counts, postures)
    ]

    # 遍历 postures，记录姿势段
    for i, posture in enumerate(postures):
        if current_posture is None:
            current_posture, start_frame = posture, i
        elif posture != current_posture:
            end_frame = i - 1
            duration = end_frame - start_frame + 1
            posture_segments.append({
                "posture": current_posture,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_sec": duration / fps,
                "duration_frames": duration
            })
            current_posture, start_frame = posture, i

    # 记录最后一个姿势段
    if current_posture is not None:
        end_frame = len(postures) - 1
        duration = end_frame - start_frame + 1
        posture_segments.append({
            "posture": current_posture,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_sec": duration / fps,
            "duration_frames": duration
        })

    # **合并无效片段（仅处理最后一个 'Invalid'，然后删除它）**
    if posture_segments and posture_segments[-1]["posture"] == "Invalid":
        new_segments = posture_segments[:-1]  # 移除最后一个 'Invalid' 片段
        last_invalid_segment = posture_segments[-1]

        # 逆序合并短片段到最后的 'Invalid'
        while new_segments:
            last_segment = new_segments[-1]
            if last_segment["duration_frames"] >= min_duration_frames:
                break  # 停止合并

            # 合并短片段
            last_invalid_segment["start_frame"] = last_segment["start_frame"]
            last_invalid_segment["duration_sec"] = (last_invalid_segment["end_frame"] - last_invalid_segment["start_frame"] + 1) / fps
            last_invalid_segment["duration_frames"] = last_invalid_segment["end_frame"] - last_invalid_segment["start_frame"] + 1

            new_segments.pop()  # 移除已合并片段

        # **删除最后的 'Invalid' 片段，不再加入**
        posture_segments = new_segments

    final_segments = posture_segments[:]  # 先复制 posture_segments
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
        if merged_segments and merged_segments[-1]["posture"] == segment["posture"]:
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
            print(f"  片段 {idx+1} ({segment['posture']}): 均值 = {mean_value:.2f}, 方差 = {variance:.2f}")

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
    if merged_segments and merged_segments[-1]["posture"] == "Invalid":
        print(f"🗑 删除最后一个 'Invalid' 片段: {merged_segments[-1]}")
        merged_segments.pop(-1)


    return merged_segments

def refine_posture_segments_with_motion(posture_segments, motion_states, fps=30, duration_sec=15):
    """
    细化姿势片段，基于 motion_state 进行二次分割，合并短片段，并合并相邻的相同姿势+motion_state。

    参数:
        posture_segments: list[dict] - 姿势片段，每个包含 start_frame, end_frame, posture
        motion_states: list[str] - 每一帧的 motion_state（'Static' 或 'Dynamic'）
        fps: int - 每秒的帧数，默认 30
        duration_sec: int - 最小合并阈值（小于该时间的片段会合并到后一个片段）

    返回:
        refined_segments: list[dict] - 细化后的姿势片段
    """
    min_duration_frames = fps * duration_sec  # 计算最小 15 秒对应的帧数

    refined_segments = []

    for segment in posture_segments:
        start, end, posture = segment["start_frame"], segment["end_frame"], segment["posture"]
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
                    "posture": posture,
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
                "posture": posture,
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
            final_segments[-1]["posture"] == segment["posture"]
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



# ========================= 5. 绘图方法 =========================
def plot_posture_durations(posture_durations):
    """绘制姿势变化的时间段图（X轴使用帧索引），Static / Dynamic 用不同的斜线覆盖，并优化视觉效果"""

    # 定义姿势对应的高度
    Height_map = {
        'Standing': 3,
        'Sitting': 2,
        'Supine': 1,
        'Prone': 1
    }

    # 姿势颜色映射
    color_map = {
        'Standing': '#1f77b4',  # 深蓝
        'Sitting': '#2ca02c',   # 绿色
        'Supine': '#d62728',    # 深红
        'Prone': '#ff7f0e'      # 橙色
    }

    # Motion state 斜线样式（Static 和 Dynamic）
    hatch_map = {
        'Static': "//",   # 右斜线
        'Dynamic': "xx"   # 交叉线
    }

    # 创建图像
    fig, ax = plt.subplots(figsize=(12, 5))

    for entry in posture_durations:
        start_time = entry["start_frame"]
        end_time = entry["end_frame"]
        Height = Height_map[entry["posture"]]
        motion_state = entry.get("motion_state", "Static")  # 确保是 Static 或 Dynamic
        posture = entry["posture"]

        # 选择颜色（比填充颜色稍深）
        fill_color = color_map[posture]
        edge_color = color_map[posture]  # 替代黑色边框，使视觉更自然
        hatch_style = hatch_map.get(motion_state, "//")  # 默认 Static 右斜线

        # 绘制矩形
        rect = plt.Rectangle((start_time, 0), end_time - start_time, Height,
                             facecolor=fill_color, edgecolor=edge_color, linewidth=1.5,
                             hatch=hatch_style, alpha=0.75)

        ax.add_patch(rect)

    # 图表细节
    plt.xlabel("Frame Index")
    plt.ylabel("Height Level")
    plt.title("Posture and Motion State Over Time (Frame-based)")
    plt.yticks([1, 2, 3], ['Low', 'Medium', 'High'])
    plt.xlim(0, max(entry["end_frame"] for entry in posture_durations) + 100)
    plt.ylim(0, 4)
    plt.grid(True, linestyle='--', alpha=0.4)

    # 创建图例
    posture_handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[p], label=p) for p in color_map]
    motion_handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=color_map['Standing'], 
                                    hatch=hatch_map[m], alpha=0.5, label=m) for m in hatch_map]

    plt.legend(posture_handles + motion_handles, [p for p in color_map] + ["Static", "Dynamic"], loc="upper right")

    # 显示图表
    plt.show()




def plot_height_variation(weighted_center, posture_durations):
    """
    绘制 weighted_center 数组中 Y 轴高度的变化折线图。
    :param weighted_center: 每个元素是一个数值，表示某个时间点的中心高度。
    :param posture_durations: 姿势段的时间范围，包含 start_frame 和 end_frame。
    """
    if not weighted_center:
        print("错误: weighted_center 为空，无法绘制图表。")
        return
    
    if not posture_durations:
        print("错误: posture_durations 为空，无法确定绘制区间。")
        return
    
        # 计算姿势的整体时间区间
   
    start_frame = min(seg["start_frame"] for seg in posture_durations)
    end_frame = max(seg["end_frame"] for seg in posture_durations)

    
    # 确保索引在合理范围内
    start_frame = max(0, start_frame)
    end_frame = min(len(weighted_center) - 1, end_frame)
    
    # 过滤掉 None 值，并确保索引有效
    filtered_center = [weighted_center[i] for i in range(start_frame, end_frame + 1) if weighted_center[i] is not None]
    
    if not filtered_center:
        print("错误: 过滤后 weighted_center 为空，无法绘制图表。")
        return
    
    # 生成 x 轴数据（时间步）
    x_values = np.arange(start_frame, start_frame + len(filtered_center))
    
    # 绘制折线图
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, filtered_center, marker='o', linestyle='-', color='b', label='Height Variation')
    plt.xlabel("Frame Index")
    plt.ylabel("Height (y-coordinate)")
    plt.title("Weighted Center Height Variation Over Time")
    plt.legend()
    plt.grid()
    plt.show()


# ========================= 6. 运行主程序 =========================
if __name__ == "__main__":
    fps, people_counts, body_height, postures, motion_states, weighted_center = load_json_data()
    print("fps:", fps)

    filtered_people_counts, filtered_postures, filtered_motion_states = filter_stable_data(people_counts, postures, motion_states)
    posture_durations = analyze_posture_durations(filtered_people_counts, filtered_postures, body_height, fps)
    posture_durations = refine_posture_segments_with_motion(posture_durations, filtered_motion_states)
    print(posture_durations)
    plot_posture_durations(posture_durations)
    plot_height_variation(weighted_center, posture_durations)


    # large_change_frames, optimal_threshold = identify_significant_changes(person_sizes, fps)
    # print("剧烈变化发生在帧:", large_change_frames)
    # if large_change_frames:
    #filtered_people_counts, filtered_postures = filter_stable_data(people_counts, postures, large_change_frames[-1])

