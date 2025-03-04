import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm


# 初始化YOLO和MediaPipe
model = YOLO("yolov8n.pt")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
pose = mp_pose.Pose(
    static_image_mode=False,  # 处理连续视频帧
    model_complexity=2,       # 选择模型复杂度（2最高精度）
    smooth_landmarks=True,    # 平滑关键点
    min_detection_confidence=0.5,  # 置信度阈值
    min_tracking_confidence=0.5    # 追踪阈值
)


def process_frame(image):
    """
    处理视频中的每一帧，检测人数并标注人体姿态。
    """
    if image is None:
        return None  # 防止空帧报错
    
    h, w = image.shape[:2]

    # YOLO 目标检测
    results_YOLO = model(image, verbose=False)
    people_count = sum(1 for box in results_YOLO[0].boxes if int(box.cls) == 0)

    # 显示检测到的人数
    text = f"YOLO: People: {people_count}"
    cv2.putText(image, text, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2)

    # 处理人体姿态检测
    img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(img_RGB)

    if results_pose.pose_landmarks:  # 检测到人体关键点

        mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        scalar = 0.1  # 可调节缩放系数

        # 提取所有 33 个关键点的坐标
        cx_list, cy_list, cz_list = [], [], []
        for i, landmark in enumerate(results_pose.pose_landmarks.landmark):
            cx_list.append(int(landmark.x * w))
            cy_list.append(int(landmark.y * h))
            cz_list.append(landmark.z)
        
        min_x = min(cx_list)
        max_x = max(cx_list)
        min_y = min(cy_list)
        max_y = max(cy_list)
        person_width = max_x - min_x
        person_height = max_y - min_y
        person_size = int (person_width * person_height)  # 面积
        cv2.putText(image, f'Person Size: {int(person_size)}', (25, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2)

        
        # 计算可见点之间的最大距离
        visible_points = [(cx_list[i], cy_list[i]) for i in range(33) if results_pose.pose_landmarks.landmark[i].visibility > 0.5]
        max_distance = max([((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 for x1, y1 in visible_points for x2, y2 in visible_points], default=0)
                
        # 选择 visibility 最高的关键点进行站立判断
        best_shoulder = 11 if results_pose.pose_landmarks.landmark[11].visibility > results_pose.pose_landmarks.landmark[12].visibility else 12
        best_hip = 23 if results_pose.pose_landmarks.landmark[23].visibility > results_pose.pose_landmarks.landmark[24].visibility else 24
        best_knee = 25 if results_pose.pose_landmarks.landmark[25].visibility > results_pose.pose_landmarks.landmark[26].visibility else 26
        best_foot = 27 if results_pose.pose_landmarks.landmark[27].visibility > results_pose.pose_landmarks.landmark[28].visibility else 28

        head_y = cy_list[0]  # 头部（最上方）
        shoulder_y = cy_list[best_shoulder]  # 可见度最高的肩膀
        hip_y = cy_list[best_hip]  # 可见度最高的髋部
        knee_y = cy_list[best_knee]  # 可见度最高的膝盖
        foot_y = cy_list[best_foot]  # 可见度最高的脚

        # 判断是否为站立状态
        if head_y < shoulder_y < hip_y < knee_y < foot_y :  
            cv2.putText(image, 'Standing',(25, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2)

        # 判断是否为仰卧状态（髋部点高于肩部）
        elif knee_y < shoulder_y:
            if cy_list[32] > cy_list[28] or cy_list[31] > cy_list[27]:
                cv2.putText(image, 'Prone', (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2)
            else:
                cv2.putText(image, 'Supine', (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2)
        
        elif abs(cy_list[15] - cy_list[27]) < scalar * max_distance or abs(cy_list[16] - cy_list[28]) < scalar * max_distance:
            if hip_y < knee_y:
                cv2.putText(image, 'Prone', (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2)
            else:
                cv2.putText(image, 'Supine', (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2)
        
        elif abs(cy_list[32] - hip_y)< scalar * max_distance or abs(cy_list[31] - hip_y)< scalar * max_distance:
            cv2.putText(image, 'Supine', (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2)

        elif abs(cy_list[32] - knee_y)< scalar * max_distance or abs(cy_list[31] - knee_y)< scalar * max_distance:
            cv2.putText(image, 'Supine', (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2)
        
        else:
            cv2.putText(image, 'Not classified', (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2)
    
    else:
        # 如果未检测到人体
        cv2.putText(image, 'No Person', (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2)
    return image


def generate_video(input_video, output_video):
    """
    读取视频，处理每一帧，并保存为新视频。
    """
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("错误: 无法打开视频文件！")
        return

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'视频信息: {total_frames} 帧, {fps} FPS, 分辨率 {width}x{height}')

    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 进度条
    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame is None:
                break

            try:
                processed_frame = process_frame(frame)
                if processed_frame is None:
                    processed_frame = frame  # 发生错误时仍然写入原始帧，避免跳帧
                out.write(processed_frame)
            except Exception as e:
                print(f"处理帧时出错: {e}")
                out.write(frame)  # 发生异常时也写入原始帧

            pbar.update(1)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"处理完成，输出视频已保存至: {output_video}")


# 运行视频处理
if __name__ == "__main__":
    input_video = "Video2-intro.mp4"  # 输入视频路径
    output_video = "output_test.mp4"  # 输出视频路径
    generate_video(input_video, output_video)
