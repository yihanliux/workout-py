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

# 关键点索引
NOSE, LEFT_EAR, RIGHT_EAR = 0, 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

def draw_text(image, text, position=(25, 100), font_scale=1, color=(0, 0, 255)):
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

def euclidean_dist(x1, y1, x2, y2):
    """计算两个点之间的欧几里得距离"""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_body_height(cx_list, cy_list):
    """计算人体的完整高度，包括头顶和脚底，并返回 body_height"""

    # 计算肩膀中点
    shoulder_mid_x = (cx_list[LEFT_SHOULDER] + cx_list[RIGHT_SHOULDER]) / 2
    shoulder_mid_y = (cy_list[LEFT_SHOULDER] + cy_list[RIGHT_SHOULDER]) / 2

    # 计算鼻子到肩膀中点的距离
    head_to_shoulder = euclidean_dist(cx_list[NOSE], cy_list[NOSE], shoulder_mid_x, shoulder_mid_y)

    # 计算左右耳到鼻子的距离，并取平均值
    left_ear_to_nose = euclidean_dist(cx_list[NOSE], cy_list[NOSE], cx_list[LEFT_EAR], cy_list[LEFT_EAR])
    right_ear_to_nose = euclidean_dist(cx_list[NOSE], cy_list[NOSE], cx_list[RIGHT_EAR], cy_list[RIGHT_EAR])
    ear_to_nose_dist = (left_ear_to_nose + right_ear_to_nose) / 2  # 取平均值

    # 使用耳朵到鼻子的距离估算头顶高度
    head_to_top = ear_to_nose_dist * 1.5  # 估算头顶到鼻子的距离

    # 计算左侧和右侧身高（使用欧几里得距离）
    def compute_side_height(shoulder, hip, knee, ankle):
        parts = [
            (cx_list[shoulder], cy_list[shoulder], cx_list[hip], cy_list[hip]),  # 肩到髋
            (cx_list[hip], cy_list[hip], cx_list[knee], cy_list[knee]),          # 髋到膝
            (cx_list[knee], cy_list[knee], cx_list[ankle], cy_list[ankle])       # 膝到踝
        ]
        return sum(euclidean_dist(x1, y1, x2, y2) for x1, y1, x2, y2 in parts)

    left_height = compute_side_height(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    right_height = compute_side_height(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)

    # 计算人体总高度
    body_height = (left_height + right_height) / 2
    #body_height += head_to_shoulder  # 添加头部高度（鼻子到肩膀）
    body_height += head_to_top  # 直接加上头顶到鼻子的高度
    #body_height += body_height * 0.045  # 添加脚底的估算高度

    return body_height

def calculate_head_y(cy_list, body_height):
    """计算头部的归一化高度（相对于地面高度），处理 None 值"""
    
    valid_y_values = [y for y in cy_list if y is not None]
    ground_y = max(valid_y_values)

    head_y = (ground_y - cy_list[NOSE]) / body_height

    return head_y, ground_y

def analyze_orientation_(cy_list, body_height, tilt_threshold=0.04):
    """
    计算耳朵高度差和鼻子与耳朵的相对高度差，以判断头部姿态。

    参数：
    - cy_list: 包含 33 个关键点的 y 坐标列表。
    - body_height: 计算得到的人体高度（单位：像素）
    - tilt_threshold: 归一化的耳朵 y 轴差值阈值（默认 3%）

    返回：
    - "up"（面朝上）
    - "down"（面朝下）
    - "tilted"（左右倾斜）
    - "neutral"（正常朝前）
    """

    # 计算鼻子相对耳朵的 y 轴高度差
    nose_ear_avg_diff = (((cy_list[NOSE] - cy_list[LEFT_EAR]) + (cy_list[NOSE] - cy_list[RIGHT_EAR])) / 2) / body_height

    # 计算正常站立情况下的基准线
    neutral_baseline = 0.02  # 经验值：一般人的鼻子比耳朵低 2% 身高（这个可以调整）

    # 计算归一化的左右耳高度差（判断左右倾斜）
    ear_diff = abs(cy_list[LEFT_EAR] - cy_list[RIGHT_EAR]) / body_height

    # 判断头部姿态
    if nose_ear_avg_diff < (neutral_baseline - tilt_threshold):  
        return "up"
    if nose_ear_avg_diff > (neutral_baseline + tilt_threshold):  
        return "down"
    if ear_diff > tilt_threshold:  
        return "tilted"
    
    return "neutral"

def analyze_orientation(cx_list, cy_list):
    """
    计算鼻子-左右耳的连线角度，以判断头部姿态（0-180°）。

    参数：
    - cx_list: 包含 33 个关键点的 x 坐标列表。
    - cy_list: 包含 33 个关键点的 y 坐标列表。

    返回：
    - "up"（面朝上）
    - "down"（面朝下）
    - "neutral"（正常朝前）
    """

    # 计算鼻子-左耳、鼻子-右耳的角度
    dx_left = abs(cx_list[NOSE] - cx_list[LEFT_EAR])
    dy_left = abs(cy_list[NOSE] - cy_list[LEFT_EAR])

    dx_right = abs(cx_list[NOSE] - cx_list[RIGHT_EAR])
    dy_right = abs(cy_list[NOSE] - cy_list[RIGHT_EAR])

    # 计算左右耳相对鼻子的角度（转换为角度制）
    angle_left = np.degrees(np.arctan2(dy_left, dx_left))
    angle_right = np.degrees(np.arctan2(dy_right, dx_right))

    # 计算平均角度
    avg_angle = (angle_left + angle_right) / 2

    # 根据角度分类姿态
    if 60 <= avg_angle <= 90:
        return "standing"
    elif 0 <= avg_angle < 60:
        return "down"  # 趴着
    elif 90 < avg_angle <= 180:
        return "up"  # 躺着
    else:
        return "unknown"

def process_pose(image):
    """处理人体姿态，返回关键点坐标，如果数据缺失则直接返回 'Insufficient data'"""

    # 颜色转换 (BGR -> RGB)
    img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 进行姿态检测
    results = pose.process(img_RGB)
    
    # 确保 `pose_landmarks` 存在
    if not results.pose_landmarks:
        return "Insufficient data"

    # 获取图像宽高
    h, w = image.shape[:2]

    # 绘制关键点
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 提取所有 33 个关键点的像素坐标
    cx_list = [int(lm.x * w) for lm in results.pose_landmarks.landmark]
    cy_list = [int(lm.y * h) for lm in results.pose_landmarks.landmark]

    # 需要的关键点索引
    required_points = [NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER,
                       LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]

    # 关键点完整性检查
    if any(cx_list[p] is None or cy_list[p] is None for p in required_points):
        return "Insufficient data"

    return cx_list, cy_list

def process_frame(image):
    """处理单帧图像，并返回 JSON 记录"""

    if image is None:
        return None, None
    
    frame_data = {
        "people_count": None,
        "body_height": None,
        "orientation": None,
        "head_y": None
    }
    
    people_count = detect_people(image)
    frame_data.update({"people_count": people_count})
    draw_text(image, f"YOLO: People: {people_count}")


    pose_data = process_pose(image)
    if pose_data == "Insufficient data":
        draw_text(image, 'No Person', (25, 200))
    else:
        cx_list, cy_list = pose_data

        body_height = calculate_body_height(cx_list, cy_list)
        frame_data.update({"body_height": body_height})
        draw_text(image, f'Body Height: {body_height}', (25, 200))

        orientation = analyze_orientation(cy_list, body_height)
        frame_data.update({"orientation": orientation})
        draw_text(image, f'Orientation: {orientation}', (25, 300))

        head_y, ground_y = calculate_head_y(cy_list, body_height)
        frame_data.update({"head_y": head_y})
        draw_text(image, f'Head Height: {head_y}', (25, 400))
        draw_text(image, f'ground_y: {ground_y}', (25, 500))
    
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
    json_output = "output_data8_.json"
    output_data = {
    "fps": fps,  
    "frames": frame_data_list     
    }
    with open(json_output, "w") as f:
        json.dump(output_data, f, indent=4)
    
    print(f"处理完成，输出视频已保存至: {output_video}")
    print(f"帧数据已保存至: {json_output}")


def generate_video2(input_video, output_video):
    """读取视频，每 2 帧读取 1 帧，并让未处理的帧复制上一帧的图像和数据"""

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
    frame_data_list = []  # 存储帧数据
    last_frame_data = None  # 记录上一帧的数据
    last_processed_frame = None  # 记录上一帧的图像

    with tqdm(total=total_frames) as pbar:
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame is None:
                break

            if frame_idx % 2 == 0:  # 每 2 帧处理 1 帧
                try:
                    processed_frame, frame_data = process_frame(frame)
                    if processed_frame is None:
                        processed_frame = frame  # 发生错误时仍然写入原始帧，避免跳帧
                    last_frame_data = frame_data  # 更新上一帧数据
                    last_processed_frame = processed_frame  # 记录当前帧的图像
                except Exception as e:
                    print(f"处理帧时出错: {e}")
                    processed_frame = frame  # 发生异常时使用原始帧
                    last_processed_frame = processed_frame
            else:
                processed_frame = last_processed_frame  # 复制上一帧的图像
                frame_data = last_frame_data  # 复制上一帧的数据

            out.write(processed_frame)
            frame_data_list.append(frame_data)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 将数据写入 JSON 文件
    json_output = "output_data.json"
    output_data = {
        "fps": fps,  
        "frames": frame_data_list     
    }
    with open(json_output, "w") as f:
        json.dump(output_data, f, indent=4)
    
    print(f"处理完成，输出视频已保存至: {output_video}")
    print(f"帧数据已保存至: {json_output}")



if __name__ == "__main__":
    generate_video("video8.mp4", "output_test8_.mp4")
