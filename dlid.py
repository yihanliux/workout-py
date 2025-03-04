import cv2
import dlib
import numpy as np

# 1️⃣ 加载 Dlib 预训练模型（用于人脸检测 + 68 关键点提取）
detector = dlib.get_frontal_face_detector()  # 人脸检测器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 关键点检测模型

# 2️⃣ 定义 3D 参考点（用于头部姿态估计）
MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),  # 鼻尖
    (-30.0, -70.0, -50.0),  # 左眼角
    (30.0, -70.0, -50.0),  # 右眼角
    (-60.0, -40.0, -50.0),  # 左嘴角
    (60.0, -40.0, -50.0),  # 右嘴角
    (0.0, -110.0, -65.0),  # 下巴
    (0.0, 40.0, -60.0)  # 额头（眉心）
], dtype=np.float32)


def process_frame(image):
    """
    处理单帧图像，检测人脸并估算头部姿态。
    :param image: 输入图像（numpy 数组）
    :return: 处理后的图像（带检测结果）
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图，提高检测精度
    faces = detector(gray)  # 检测所有人脸

    if len(faces) == 0:
        # ❌ 没有检测到人脸，在画面上显示 "No Face Detected"
        image = cv2.putText(image, "No Face Detected", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
        return image  # 直接返回图像

    for face in faces:
        landmarks = predictor(gray, face)  # 提取 68 个人脸关键点

        # 提取 2D 关键点（用于计算头部姿态）
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # 鼻尖
            (landmarks.part(36).x, landmarks.part(36).y),  # 左眼角
            (landmarks.part(45).x, landmarks.part(45).y),  # 右眼角
            (landmarks.part(48).x, landmarks.part(48).y),  # 左嘴角
            (landmarks.part(54).x, landmarks.part(54).y),  # 右嘴角
            (landmarks.part(8).x, landmarks.part(8).y),  # 下巴
            (landmarks.part(27).x, landmarks.part(27).y)  # 额头（眉心）
        ], dtype=np.float32)

        # 计算 3D 头部姿态
        _, rotation_vec, _ = cv2.solvePnP(MODEL_3D_POINTS, image_points, np.eye(3), None, flags=cv2.SOLVEPNP_EPNP)

        # 提取 Yaw 角度（偏航角，用于判断是否直面摄像头）
        yaw = rotation_vec[1] * (180 / np.pi)  # 转换为角度制

        # 调整阈值，使得检测更稳定
        if -20 < yaw < 20:  # 放宽阈值，避免误判
            text = "Facing Camera"
            color = (0, 255, 0)  # 绿色
        else:
            text = "Not Facing Camera"
            color = (0, 0, 255)  # 红色

        # 在图像上绘制检测结果
        image = cv2.putText(image, text, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.25, color, 3)

    return image  # 返回处理后的图像


def generate_video(input_video_path, output_video_path):
    """
    处理整个视频，并保存带检测结果的视频。
    :param input_video_path: 输入视频路径
    :param output_video_path: 输出视频路径
    """
    cap = cv2.VideoCapture(input_video_path)  # 打开视频文件
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {input_video_path}")
        return

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度

    # 配置输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 指定编码格式
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"📹 处理视频: {input_video_path} -> {output_video_path}")

    # 逐帧处理视频
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束

        # 处理当前帧
        processed_frame = process_frame(frame)

        # 保存结果
        out.write(processed_frame)
        cv2.imshow("Head Pose Estimation", processed_frame)  # 显示实时检测结果

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 退出
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 处理完成，视频已保存: {output_video_path}")


# 示例调用（替换为你的视频路径）
if __name__ == "__main__":
    input_video = "Pre-video1.mp4"  # 输入视频路径
    output_video = "output_dlid.mp4"  # 输出视频路径
    generate_video(input_video, output_video)
