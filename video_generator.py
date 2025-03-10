import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import json
from ultralytics import YOLO
from tqdm import tqdm
from itertools import combinations
import math
import numpy as np

import data_processer

# 初始化YOLO和MediaPipe
model = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 滑动窗口存储最近 10 帧的关键点数据
pose_history = []
SIMILARITY_THRESHOLD = 0.05  # 归一化后的坐标，阈值较小
STATIC_THRESHOLD_COUNT = 7  # 过去 10 帧中至少 7 帧相似，则判定为静态

def draw_text(image, text, position=(25, 100), font_scale=1.25, color=(0, 0, 255)):
    """
    绘制文本，并根据图像大小动态调整字体大小和位置，保持在不同分辨率下的可读性。

    参数:
        image: np.ndarray - OpenCV 处理的图像
        text: str - 要绘制的文本
        position: tuple - (x, y) 坐标，文本左下角的像素位置（默认 (25, 100)）
        font_scale: float - 初始字体大小（默认 1.25）
        color: tuple - 文本颜色 (B, G, R)（默认 红色 (0, 0, 255)）
    """
    h, w = image.shape[:2]  # 获取图像的高度和宽度

    # 计算相对字体缩放：基于画面高度调整
    scale_factor = h / 720  # 720p 作为基准
    adjusted_font_scale = font_scale * scale_factor

    # 计算相对文本位置：基于画面大小调整
    adjusted_position = (int(position[0] * scale_factor), int(position[1] * scale_factor))

    # 绘制文本
    cv2.putText(image, text, adjusted_position, cv2.FONT_HERSHEY_SIMPLEX, adjusted_font_scale, color, 2)

def detect_people(image):
    """YOLO 检测人数"""
    results = model(image, verbose=False)
    return sum(int(box.cls) == 0 for box in results[0].boxes)

def normalize_landmarks(cx_list, cy_list):
    """ 归一化关键点坐标，相对于骨盆中心，并按骨盆到肩膀距离缩放 """
    cx_array = np.array(cx_list, dtype=np.float32)
    cy_array = np.array(cy_list, dtype=np.float32)

    # 确保关键点数量正确
    if len(cx_array) < 33 or len(cy_array) < 33:
        return None  # 避免关键点缺失导致错误

    # 计算骨盆中心（点 23 和 24 的中点）
    pelvis_x = (cx_array[23] + cx_array[24]) / 2
    pelvis_y = (cy_array[23] + cy_array[24]) / 2

    # 计算肩膀中心（点 11 和 12 的中点）
    shoulder_x = (cx_array[11] + cx_array[12]) / 2
    shoulder_y = (cy_array[11] + cy_array[12]) / 2

    # 计算骨盆到肩膀的距离，作为归一化尺度
    scale_factor = np.linalg.norm([shoulder_x - pelvis_x, shoulder_y - pelvis_y])

    if scale_factor == 0:
        return None  # 避免除零错误

    # 归一化关键点
    norm_cx = (cx_array - pelvis_x) / scale_factor
    norm_cy = (cy_array - pelvis_y) / scale_factor

    return np.column_stack((norm_cx, norm_cy))

def compute_similarity(current, previous):
    """计算两帧之间的相似度，使用归一化后的欧几里得距离"""
    if current is None or previous is None:
        return float('inf')
    
    distances = np.linalg.norm(current - previous, axis=1)  # 计算 33 个关键点的欧几里得距离
    return np.mean(distances)  # 取平均值作为全局相似度

def determine_motion_state(current_pose):
    """判断当前帧是静态还是动态"""
    global pose_history

    if len(pose_history) == 0:
        pose_history.append(current_pose)  # 存入第一帧，避免后续出错
        return "Not classified"

    # 计算当前帧与过去帧的相似度
    similarity_scores = [compute_similarity(current_pose, past_pose) for past_pose in pose_history]
    
    # 计算过去 10 帧中，有多少帧相似度小于阈值
    static_count = sum(score < SIMILARITY_THRESHOLD for score in similarity_scores)

    # 过去 10 帧中至少 STATIC_THRESHOLD_COUNT 帧相似，则判定为静态
    motion_state = "Static" if static_count >= STATIC_THRESHOLD_COUNT else "Dynamic"

    # 更新滑动窗口，最多保留 10 帧
    pose_history.append(current_pose)
    if len(pose_history) > 10:
        pose_history.pop(0)

    return motion_state


def calculate_body_height(cx_list, cy_list):
    """
    计算人体的近似高度（像素单位），如果关键点数据缺失，则返回 "Insufficient data"。
    
    参数:
        cx_list: list[int | None] - 关键点的 x 坐标（像素单位）。
        cy_list: list[int | None] - 关键点的 y 坐标（像素单位）。
    
    返回:
        total_height: float - 近似人体高度（像素单位）。
        或者
        "Insufficient data" - 如果数据缺失。
    """
    # 关键点索引
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    LEFT_KNEE = 25
    LEFT_ANKLE = 27
    RIGHT_HIP = 24
    RIGHT_KNEE = 26
    RIGHT_ANKLE = 28

    # 计算肩膀中点
    def midpoint(x1, y1, x2, y2):
        """ 计算两个点的中点坐标 """
        if None in [x1, y1, x2, y2]:
            return None, None
        return (x1 + x2) / 2, (y1 + y2) / 2

    # 计算欧几里得距离
    def euclidean_dist(x1, y1, x2, y2):
        """ 计算两个点的欧几里得距离 """
        if None in [x1, y1, x2, y2]:
            return "Insufficient data"
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # 计算鼻子到肩膀中点的距离
    shoulder_mid_x, shoulder_mid_y = midpoint(cx_list[LEFT_SHOULDER], cy_list[LEFT_SHOULDER],
                                              cx_list[RIGHT_SHOULDER], cy_list[RIGHT_SHOULDER])

    head_to_shoulder = euclidean_dist(cx_list[NOSE], cy_list[NOSE], shoulder_mid_x, shoulder_mid_y)
    if head_to_shoulder == "Insufficient data":
        return "Insufficient data"

    # 计算左侧和右侧的高度
    left_height = sum([
        head_to_shoulder,
        euclidean_dist(shoulder_mid_x, shoulder_mid_y, cx_list[LEFT_HIP], cy_list[LEFT_HIP]),
        euclidean_dist(cx_list[LEFT_HIP], cy_list[LEFT_HIP], cx_list[LEFT_KNEE], cy_list[LEFT_KNEE]),
        euclidean_dist(cx_list[LEFT_KNEE], cy_list[LEFT_KNEE], cx_list[LEFT_ANKLE], cy_list[LEFT_ANKLE])
    ])

    right_height = sum([
        head_to_shoulder,
        euclidean_dist(shoulder_mid_x, shoulder_mid_y, cx_list[RIGHT_HIP], cy_list[RIGHT_HIP]),
        euclidean_dist(cx_list[RIGHT_HIP], cy_list[RIGHT_HIP], cx_list[RIGHT_KNEE], cy_list[RIGHT_KNEE]),
        euclidean_dist(cx_list[RIGHT_KNEE], cy_list[RIGHT_KNEE], cx_list[RIGHT_ANKLE], cy_list[RIGHT_ANKLE])
    ])

    # 检查是否有 "Insufficient data" 错误
    if "Insufficient data" in [left_height, right_height]:
        return "Insufficient data"

    # 取左右侧的平均值，确保对称
    total_height = (left_height + right_height) / 2.0
    return total_height

def calculate_center(cy_list, body_height):
    """
    计算人体重心相对于地面的高度（y 轴）。

    参数:
    cy_list - List[float or None]，包含 33 个关键点的 y 坐标，可能包含 None

    返回:
    Ycg - 人体相对于地面的重心 y 坐标 或 "Insufficient data"
    """
    # 关键点索引及更精细的质量比例（基于生物力学）
    body_parts = {
        "head": ([0], 0.069),  # 头部占 6.9%
        "upper_torso": ([11, 12, 23, 24], 0.216),  # **上躯干：肩膀 → 胸部**（21.6%）
        "lower_torso": ([23, 24, 25, 26], 0.2186),  # **下躯干：胸部 → 骨盆**（21.86%）
        "upper_arm": ([13, 14], 0.0265 * 2),  # 上臂（2.65% x2）
        "forearm": ([15, 16], 0.018 * 2),  # 前臂（1.8% x2）
        "thigh": ([23, 24, 25, 26], 0.116 * 2),  # **大腿中心：髋部 + 膝盖**
        "calf": ([25, 26, 27, 28], 0.055 * 2),  # **小腿中心：膝盖 + 脚踝**
        "foot": ([27, 28, 31, 32], 0.0155 * 2),  # **脚中心：脚踝 + 脚尖**
    }

    total_weight = sum(weight for _, weight in body_parts.values())

    # 过滤掉 None 值
    valid_y_list = [y for y in cy_list if y is not None]

    if not valid_y_list:
        return "Insufficient data"
    
    # 计算地面高度（最小 y）
    ground_y = min(valid_y_list)

    # 检查是否有足够的关键点数据
    required_indices = [index for indices, _ in body_parts.values() for index in indices]
    if any(cy_list[i] is None for i in required_indices):
        return "Insufficient data"

    # 计算加权 y 坐标
    Ycg = 0
    for part, (indices, weight) in body_parts.items():
        part_y = np.mean([cy_list[i] for i in indices])  # 取该部位的 y 坐标均值
        Ycg += part_y * weight

    # 计算最终的重心 y
    Ycg /= total_weight

    # 相对地面的重心
    Ycg_norm = (Ycg - ground_y) / body_height


    return Ycg_norm

def classify_posture(image, cx_list, cy_list, body_height):
    """判断站立/仰卧/其他姿态，并返回姿态类型"""

    scalar = 0.1
    posture = "Not classified"

    head_y = cy_list[0]
    shoulder_y = (cy_list[11] + cy_list[12]) / 2
    hip_y = (cy_list[23] +  cy_list[24]) / 2
    knee_y = (cy_list[25] + cy_list[26]) / 2
    foot_y = (cy_list[27] + cy_list[28]) / 2
    toe_y = (cy_list[31] + cy_list[32]) / 2

    shoulder_x = (cx_list[11] + cx_list[12]) / 2
    knee_x = (cx_list[25] + cx_list[26]) / 2

    feet_on = cy_list[32] + cy_list[31] < cy_list[30] + cy_list[29] 
    
    # 膝盖 > 肩, 然后比脚尖和脚跟 
    if knee_y < shoulder_y:
    # 如果膝盖比头还高，直接判断为仰卧（Supine）
        if knee_y < head_y:
            posture = "Supine"
        else:
            # 通过脚后跟 vs 脚趾 的 y 轴坐标来判断朝向
            posture = "Supine" if feet_on else "Prone"

    # 脚跟与髋部垂直距离足够小
    elif abs(foot_y - hip_y) < scalar * body_height:
        if abs(shoulder_y - hip_y) > scalar * body_height:  
            posture = "Sitting"
        elif feet_on:
            posture = "Supine"  # 躺着
        else:
             posture = "Prone"  # 脚掌朝下，趴着
    
    #头 > 肩 > 髋 > 膝盖 > 脚
    elif head_y < shoulder_y < hip_y < knee_y < foot_y:
        posture = "Standing"

    # 如果脚尖和膝盖的垂直距离足够小，直接判定为趴着
    elif abs(cy_list[32] - knee_y) < scalar * body_height or abs(cy_list[31] - knee_y) < scalar * body_height:
        posture = "Prone"  

    # 手和脚尖垂直距离足够小, 然后先比手和脚的垂直距离，再比髋比膝盖的垂直。
    elif abs(cy_list[19] - cy_list[27]) < scalar * body_height or abs(cy_list[20] - cy_list[28]) < scalar * body_height or abs(cy_list[19] - cy_list[28]) < scalar * body_height or abs(cy_list[20] - cy_list[27]) < scalar * body_height:
        if hip_y < knee_y:
            posture = "Prone"
        else:
            posture = "Supine"

    return posture

def process_pose(image):
    """处理人体姿态，返回关键点坐标"""
    img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(img_RGB)
    if not results.pose_landmarks:
        return None
    
    h, w = image.shape[:2]
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # 提取 33 个关键点
    cx_list = [int(landmark.x * w) for landmark in results.pose_landmarks.landmark]
    cy_list = [int(landmark.y * h) for landmark in results.pose_landmarks.landmark]
    
    min_x, max_x = min(cx_list), max(cx_list)
    min_y, max_y = min(cy_list), max(cy_list)
    person_width = max_x - min_x
    person_height = max_y - min_y
    
    # 计算长方形对角线
    body_height = calculate_body_height(cx_list, cy_list)

    norm_list = normalize_landmarks(cx_list, cy_list)
    
    return cx_list, cy_list, body_height, norm_list

def process_frame(image):
    """处理单帧图像，并返回 JSON 记录"""

    global pose_history

    if image is None:
        return None, None
    
    people_count = detect_people(image)
    draw_text(image, f"YOLO: People: {people_count}")

    frame_data = {
        "people_count": people_count,
        "body_height": None,
        "posture": "No Person",
        "motion_state": "Not classified",
        "weighted_center": None
    }

    pose_data = process_pose(image)
    
    if pose_data:
        cx_list, cy_list, body_height, norm_list = pose_data

        frame_data.update({"body_height": body_height})
        draw_text(image, f'Body Height: {body_height}', (25, 200))

        posture = classify_posture(image, cx_list, cy_list, body_height)
        frame_data.update({"posture": posture})
        draw_text(image, f'Posture: {posture}', (25, 300))

        weighted_center = calculate_center(cy_list, body_height)
        frame_data.update({"weighted_center": weighted_center})
        draw_text(image, f'Weighted Center: {weighted_center}', (25, 400))

        motion_state = determine_motion_state(norm_list)
        frame_data.update({"motion_state": motion_state})
        draw_text(image, f'Motion State: {motion_state}', (25, 500))
    
    else:
        draw_text(image, 'No Person', (25, 200))
    
    return image, frame_data

def generate_video(input_video, output_video):
    """读取视频，处理每一帧，并保存为新视频，同时收集数据"""
    try:
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise ValueError("错误: 无法打开视频文件！")
    except Exception as e:
        print(f"视频读取失败: {e}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_data_list = []  # 用于存储所有帧的数据
    
    # 进度条
    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame is None:
                break

            try:
                processed_frame, frame_data = process_frame(frame)
                if processed_frame is None:
                    processed_frame = frame  # 发生错误时仍然写入原始帧，避免跳帧
                out.write(processed_frame)

                if frame_data:
                    frame_data_list.append(frame_data)
            except Exception as e:
                print(f"处理帧时出错: {e}")
                out.write(frame)  # 发生异常时也写入原始帧

            pbar.update(1)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 将数据写入 JSON 文件
    json_output = "output_data9.json"
    output_data = {
    "fps": fps,  
    "frames": frame_data_list     
    }
    with open(json_output, "w") as f:
        json.dump(output_data, f, indent=4)
    
    print(f"处理完成，输出视频已保存至: {output_video}")
    print(f"帧数据已保存至: {json_output}")

if __name__ == "__main__":
    generate_video("video9.mp4", "output_test9.mp4")
