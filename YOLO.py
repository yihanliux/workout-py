import cv2
from ultralytics import YOLO
from tqdm import tqdm

# 加载 YOLOv8 目标检测模型
model = YOLO("yolov8n.pt")  # 选择轻量版 YOLOv8 nano

def process_frame(image):
    """
    处理每一帧，检测视频中的人数，并标注人数
    """
    results = model(image)  # 运行目标检测
    people_count = sum(1 for box in results[0].boxes if int(box.cls) == 0)  # 统计 "person" 类别
    
    # 在画面上打印人数
    text = f"People: {people_count}"
    color = (0, 0, 255)  # 红色字体
    image = cv2.putText(image, text, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.25, color, 3)
    
    return image

def generate_video(input_video, output_video):
    """
    读取视频，处理每一帧，生成新视频
    """
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("无法打开视频文件！")
        return

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 直接获取帧数

    print(f'视频信息: {total_frames} 帧, {fps} FPS, 分辨率 {width}x{height}')

    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 进度条
    with tqdm(total=total_frames, desc="Processing Video") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            try:
                frame = process_frame(frame)
            except Exception as e:
                print(f"处理帧时出错: {e}")
                continue  # 跳过当前帧

            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"处理完成，输出视频已保存至: {output_video}")

# 运行视频处理
if __name__ == "__main__":
    input_video = "Video2-intro.mp4"  # 输入视频路径
    output_video = "output_YOLO_Video2.mp4"  # 输出视频路径
    generate_video(input_video, output_video)
