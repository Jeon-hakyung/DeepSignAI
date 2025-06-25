import cv2
import os
import csv
import mediapipe as mp

# === 설정 ===
VIDEO_ROOT = "videos_augmented"  # 입력 폴더
OUTPUT_ROOT = "coords"           # 출력 폴더
POSE_IDX = [11, 12, 13, 14, 15, 16, 23, 24]  # 상체 (8개)
FACE_IDX = [61, 291, 0, 13, 14, 17, 87, 178]  # 얼굴 입 중심 (8개)

# MediaPipe Holistic 초기화
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

def extract_coords(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    row = []

    # 왼손 (21개)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])
    else:
        row.extend([0.0] * 63)

    # 오른손 (21개)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])
    else:
        row.extend([0.0] * 63)

    # 상체 포즈 (8개)
    if results.pose_landmarks:
        for idx in POSE_IDX:
            lm = results.pose_landmarks.landmark[idx]
            row.extend([lm.x, lm.y, lm.z])
    else:
        row.extend([0.0] * 3 * len(POSE_IDX))

    # 얼굴 (8개)
    if results.face_landmarks:
        for idx in FACE_IDX:
            lm = results.face_landmarks.landmark[idx]
            row.extend([lm.x, lm.y, lm.z])
    else:
        row.extend([0.0] * 3 * len(FACE_IDX))

    return row  # 총 58 × 3 = 174차원

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    with open(output_path, "w", newline='') as f:
        writer = csv.writer(f)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            coords = extract_coords(frame)
            writer.writerow(coords)
    cap.release()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    for class_name in os.listdir(VIDEO_ROOT):
        class_dir = os.path.join(VIDEO_ROOT, class_name)
        if not os.path.isdir(class_dir):
            continue

        output_class_dir = os.path.join(OUTPUT_ROOT, class_name)
        ensure_dir(output_class_dir)

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.mp4', '.avi', '.mov')):
                continue

            video_path = os.path.join(class_dir, fname)
            name, _ = os.path.splitext(fname)
            output_path = os.path.join(output_class_dir, name + ".csv")
            process_video(video_path, output_path)
            print(f"✅ {class_name}/{fname} → {name}.csv 완료")

    print("\n🎉 모든 CSV 추출 완료!")

if __name__ == "__main__":
    main()
