from moviepy.video.fx import all as vfx
import os
import cv2
from moviepy.editor import VideoFileClip
import numpy as np
import shutil

# === 설정 ===
CLASS_NAME = "비수어"  # 비수어 클래스 폴더명
INPUT_DIR = os.path.join("videos", CLASS_NAME)
OUTPUT_DIR = os.path.join("videos_augmented", CLASS_NAME)

# 증강 옵션
SPEEDS = [0.8, 1.2]  # 느리게, 빠르게
SHIFTS = [(-10, 0), (10, 0)]  # 좌우 이동
SCALES = [1.1]  # 확대

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def change_speed(input_path, output_path, speed):
    clip = VideoFileClip(input_path)
    new_clip = clip.fx(vfx.speedx, speed)
    new_clip.write_videofile(output_path, audio=False, codec="libx264", verbose=False, logger=None)

def shift_frame(frame, dx, dy):
    h, w = frame.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def scale_frame(frame, scale):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 0, scale)
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def augment_video_cv2(input_path, output_path, augment_func):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        aug_frame = augment_func(frame)
        out.write(aug_frame)
    cap.release()
    out.release()

def augment_non_sign_class():
    ensure_dir(OUTPUT_DIR)

    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(('.mp4', '.avi', '.mov')):
            continue
        name, ext = os.path.splitext(fname)
        input_path = os.path.join(INPUT_DIR, fname)

        # 1. 속도 증강
        for spd in SPEEDS:
            output_path = os.path.join(OUTPUT_DIR, f"{name}_speed{spd}{ext}")
            change_speed(input_path, output_path, spd)

        # 2. 이동 증강
        for dx, dy in SHIFTS:
            output_path = os.path.join(OUTPUT_DIR, f"{name}_shift{dx}_{dy}{ext}")
            augment_video_cv2(input_path, output_path, lambda f: shift_frame(f, dx, dy))

        # 3. 확대 증강
        for scale in SCALES:
            output_path = os.path.join(OUTPUT_DIR, f"{name}_scale{scale}{ext}")
            augment_video_cv2(input_path, output_path, lambda f: scale_frame(f, scale))

        # 4. 원본 복사
        shutil.copy(input_path, os.path.join(OUTPUT_DIR, fname))

if __name__ == "__main__":
    augment_non_sign_class()
    print("✅ 비수어(정지) 클래스 증강 완료!")
