import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import json
from ultralytics import YOLO
from tqdm import tqdm
from itertools import combinations
import math

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

def draw_text(image, text, position=(25, 100), font_scale=1.25, color=(0, 0, 255)):
    """简化文本绘制函数"""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

def detect_people(image):
    """YOLO 检测人数"""
    results = model(image, verbose=False)
    return sum(int(box.cls) == 0 for box in results[0].boxes)

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
    max_distance = math.sqrt(person_width ** 2 + person_height ** 2)
    person_size = int(person_width * person_height)
    
    return cx_list, cy_list, max_distance, person_size

def classify_posture(image, cx_list, cy_list, max_distance):
    """判断站立/仰卧/其他姿态，并返回姿态类型"""
    scalar = 0.1
    best_shoulder = 11 if cy_list[11] < cy_list[12] else 12
    best_hip = 23 if cy_list[23] < cy_list[24] else 24
    best_knee = 25 if cy_list[25] < cy_list[26] else 26
    best_foot = 27 if cy_list[27] < cy_list[28] else 28
    
    head_y, shoulder_y, hip_y, knee_y, foot_y = cy_list[0], cy_list[best_shoulder], cy_list[best_hip], cy_list[best_knee], cy_list[best_foot]
    head_x, shoulder_x, hip_x, knee_x, foot_x = cx_list[0], cx_list[best_shoulder], cx_list[best_hip], cx_list[best_knee], cx_list[best_foot]

    # 计算 x 轴的最大、最小值
    min_x = min(head_x, shoulder_x, hip_x, knee_x, foot_y)
    max_x = max(head_x, shoulder_x, hip_x, knee_x, foot_y)

    posture = "Not classified"
    
    #头 > 肩 > 髋 > 膝盖 > 脚
    if head_y < shoulder_y < hip_y < knee_y < foot_y and (max_x - min_x) < 0.5 * max_distance:
        posture = "Standing"
    
    # 膝盖 > 肩, 然后比脚尖和脚跟 
    elif knee_y < shoulder_y:
    # 如果膝盖比头还高，直接判断为仰卧（Supine）
        if knee_y < head_y:
            posture = "Supine"
        else:
            # 通过脚后跟 vs 脚趾 的 y 轴坐标来判断朝向
            posture = "Prone" if cy_list[30] < cy_list[32] and cy_list[29] < cy_list[31] else "Supine"

    # 手和脚尖垂直距离足够小, 然后先比手和脚的垂直距离，再比髋比膝盖的垂直。
    elif abs(cy_list[15] - cy_list[27]) < scalar * max_distance or abs(cy_list[16] - cy_list[28]) < scalar * max_distance:
    # 如果手和脚的水平距离也足够小，则认为是坐姿
        if abs(cx_list[15] - cx_list[27]) < 3 * scalar * max_distance or abs(cx_list[16] - cx_list[28]) < 3 * scalar * max_distance:
            posture = "Sitting"
        # 如果髋部（hip）低于膝盖，说明是仰卧
        elif hip_y < knee_y:
            posture = "Supine"
        # 其他情况判断为俯卧
        else:
            posture = "Prone"

    # 脚尖与髋部垂直距离足够小
    elif abs(cy_list[32] - hip_y) < scalar * max_distance or abs(cy_list[31] - hip_y) < scalar * max_distance:
    # 判断脚掌朝上还是朝下
        if cy_list[30] > cy_list[32] and cy_list[29] > cy_list[31]:  # 脚掌朝上（heel > toes）
            # 如果肩膀和脚的水平距离足够小，则是坐姿
            if abs(cx_list[11] - cx_list[27]) < 4 * scalar * max_distance and abs(cx_list[12] - cx_list[28]) < 4 * scalar * max_distance:
                posture = "Sitting"
            else:
                posture = "Supine"  # 躺着
        else:
            posture = "Prone"  # 脚掌朝下，趴着

    elif abs(cy_list[32] - knee_y) < scalar * max_distance or abs(cy_list[31] - knee_y) < scalar * max_distance:
        posture = "Prone"  # 如果脚尖和膝盖的垂直距离足够小，直接判定为趴着

    draw_text(image, posture, (25, 200))
    return posture

def process_frame(image):
    """处理单帧图像，并返回 JSON 记录"""
    if image is None:
        return None, None
    
    people_count = detect_people(image)
    draw_text(image, f"YOLO: People: {people_count}")

    pose_data = process_pose(image)
    
    frame_data = {
        "people_count": people_count,
        "person_size": None,
        "max_distance": None,
        "posture": "No Person"
    }

    if pose_data:
        cx_list, cy_list, max_distance, person_size = pose_data
        posture = classify_posture(image, cx_list, cy_list, max_distance)

        frame_data.update({"person_size": person_size, "max_distance": max_distance, "posture": classify_posture(image, cx_list, cy_list, max_distance)})

        draw_text(image, f'Person Size: {person_size}', (25, 300))
        draw_text(image, f'Max Distance: {max_distance}', (25, 400))
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
    json_output = "output_data6.json"
    with open(json_output, "w") as f:
        json.dump(frame_data_list, f, indent=4)
    
    print(f"处理完成，输出视频已保存至: {output_video}")
    print(f"帧数据已保存至: {json_output}")

if __name__ == "__main__":
    generate_video("Video6.mp4", "output_test6.mp4")
