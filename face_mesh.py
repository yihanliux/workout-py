import cv2
import torch
import numpy as np
from ultralytics import YOLO

# 加载 YOLOv8-Face 模型
model = YOLO("yolov8n-face.pt")

# 读取视频
cap = cv2.VideoCapture("Pre-video1.mp4")

# 获取视频属性
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 初始化视频写入器
output_filename = "output_Pre-video1.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8-Face 检测
    results = model(frame)
    
    face_detected = False
    face_orientation = "Unknown"

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            face_detected = True

            # 计算面部关键点
            face_center = np.array([(x1 + x2) // 2, (y1 + y2) // 2])  # 面部中心
            nose = np.array([face_center[0], y1 + (y2 - y1) * 0.2])  # 近似鼻子位置
            chin = np.array([face_center[0], y1 + (y2 - y1) * 0.9])  # 近似下巴位置

            # 计算头部方向向量
            head_direction = chin - nose  # 头部方向

            # **分类面部方向**
            if abs(head_direction[0]) > abs(head_direction[1]):  # 水平方向占主导
                if head_direction[0] < 0:
                    face_orientation = "Left"
                else:
                    face_orientation = "Right"
            else:  # 垂直方向占主导
                if head_direction[1] < 0:  # 头部朝上
                    face_orientation = "Up"
                elif head_direction[1] > 0:  # 头部朝下
                    face_orientation = "Down"
                else:
                    face_orientation = "Forward"  # 头部正对镜头

            # 画框并标注方向
            color = (255, 255, 255)  # 默认白色
            if face_orientation == "Left":
                color = (0, 255, 0)  # 绿色
            elif face_orientation == "Right":
                color = (0, 0, 255)  # 红色
            elif face_orientation == "Up":
                color = (255, 255, 0)  # 黄色
            elif face_orientation == "Down":
                color = (255, 0, 255)  # 紫色
            elif face_orientation == "Forward":
                color = (0, 255, 255)  # 青色

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Face: {face_orientation}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 如果没有检测到人脸，显示提示
    if not face_detected:
        cv2.putText(frame, "Face Not Detected", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 写入视频
    out.write(frame)

    # 显示视频
    cv2.imshow("Face Orientation Detection", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"处理后的视频已保存到：{output_filename}")
