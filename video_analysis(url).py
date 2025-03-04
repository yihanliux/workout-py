

import time
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
from io import BytesIO



mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
pose = mp_pose.Pose(static_image_mode=False,        # 是静态图片还是连续视频帧
                    model_complexity=2,            # 选择人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于两者之间
                    smooth_landmarks=True,         # 是否平滑关键点
                    min_detection_confidence=0.5,  # 置信度阈值
                    min_tracking_confidence=0.5)   # 追踪阈值



def process_frame(img):
    
    # 记录该帧开始处理的时间
    start_time = time.time()
    
    # 获取图像宽高
    h, w = img.shape[0], img.shape[1]

    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型，获取预测结果
    results = pose.process(img_RGB)

    if results.pose_landmarks: # 若检测出人体关键点

        # 可视化关键点及骨架连线
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for i in range(33): # 遍历所有33个关键点，可视化

            # 获取该关键点的三维坐标
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            cz = results.pose_landmarks.landmark[i].z

            radius = 10

            if i == 0: # 鼻尖
                img = cv2.circle(img,(cx,cy), radius, (0,0,255), -1)
            elif i in [11,12]: # 肩膀
                img = cv2.circle(img,(cx,cy), radius, (223,155,6), -1)
            elif i in [23,24]: # 髋关节
                img = cv2.circle(img,(cx,cy), radius, (1,240,255), -1)
            elif i in [13,14]: # 胳膊肘
                img = cv2.circle(img,(cx,cy), radius, (140,47,240), -1)
            elif i in [25,26]: # 膝盖
                img = cv2.circle(img,(cx,cy), radius, (0,0,255), -1)
            elif i in [15,16,27,28]: # 手腕和脚腕
                img = cv2.circle(img,(cx,cy), radius, (223,155,60), -1)
            elif i in [17,19,21]: # 左手
                img = cv2.circle(img,(cx,cy), radius, (94,218,121), -1)
            elif i in [18,20,22]: # 右手
                img = cv2.circle(img,(cx,cy), radius, (16,144,247), -1)
            elif i in [27,29,31]: # 左脚
                img = cv2.circle(img,(cx,cy), radius, (29,123,243), -1)
            elif i in [28,30,32]: # 右脚
                img = cv2.circle(img,(cx,cy), radius, (193,182,255), -1)
            elif i in [9,10]: # 嘴
                img = cv2.circle(img,(cx,cy), radius, (205,235,255), -1)
            elif i in [1,2,3,4,5,6,7,8]: # 眼及脸颊
                img = cv2.circle(img,(cx,cy), radius, (94,218,121), -1)
            else: # 其它关键点
                img = cv2.circle(img,(cx,cy), radius, (0,255,0), -1)

        # 展示图片
        # look_img(img)

    else:
        scaler = 1
        failure_str = 'No Person'
        img = cv2.putText(img, failure_str, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
        # print('从图像中未检测出人体关键点，报错。')
        
    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1/(end_time - start_time)
    
    scaler = 1
    # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
    img = cv2.putText(img, 'FPS  '+str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
    return img




def generate_video(input_url):

    print(f"视频开始处理 {input_url}")

    # 从 URL 获取视频流
    response = requests.get(input_url, stream=True)

    if response.status_code != 200:
        print(f"无法下载视频，状态码：{response.status_code}")
        return
    
    video_bytes = BytesIO(response.content)
    cap = cv2.VideoCapture(video_bytes)

    if not cap.isOpened():
        print("无法打开视频流")
        return
    
    # 获取视频信息
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数为 {frame_count}")
    
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 输出格式
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = "url-out-video.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    # 进度条绑定视频总帧数
    with tqdm(total=frame_count) as pbar:
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # 处理帧
                try:
                    frame = process_frame(frame)
                except Exception as e:
                    print(f"处理帧时发生错误: {e}")
                    continue

                # 写入输出视频
                out.write(frame)
                pbar.update(1)

        except Exception as e:
            print(f"视频处理发生错误: {e}")

    # 释放资源
    cap.release()
    out.release()
    print(f"视频已保存 {output_path}")

generate_video(https://www.youtube.com/watch?v=KPG1tJW8dwQ)