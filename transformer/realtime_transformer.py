# realtime_transformer.py

import cv2
import torch
import numpy as np
from collections import deque
import mediapipe as mp
from transformer_model import SignTransformer

# === 설정 ===
SEQ_LEN = 30
INPUT_DIM = 174
NUM_CLASS = 3  # 클래스 수 맞게 수정
MODEL_PATH = "best_transformer_sign_model.pt"
MEAN_PATH = "mean.npy"
STD_PATH = "std.npy"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 정규화 값 불러오기 ===
mean = np.load(MEAN_PATH)  # shape: (174,)
std = np.load(STD_PATH)    # shape: (174,)
std[std == 0] = 1e-6        # divide-by-zero 방지

# === 모델 로드 ===
model = SignTransformer(input_dim=INPUT_DIM, num_classes=NUM_CLASS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === MediaPipe 초기화 ===
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_drawing = mp.solutions.drawing_utils

# === 좌표 시퀀스 버퍼 초기화 ===
seq = deque(maxlen=SEQ_LEN)

# === 클래스 이름 (수어 클래스 명칭에 따라 수정) ===
class_names = ["thx", "call", "none"]

def extract_keypoints(results):
    lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))

    pose_landmark_idx = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    pose = []
    if results.pose_landmarks and len(results.pose_landmarks.landmark) >= max(pose_landmark_idx) + 1:
        for i in pose_landmark_idx:
            lm = results.pose_landmarks.landmark[i]
            pose.append([lm.x, lm.y, lm.z])
        pose = np.array(pose)
    else:
        pose = np.zeros((16, 3))

    keypoints = np.concatenate([lh, rh, pose], axis=0)  # (58, 3)
    return keypoints.flatten()  # → (174,)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    print("현재 시퀀스 길이:", len(seq))

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    keypoints = extract_keypoints(results)
    if keypoints.shape[0] == INPUT_DIM:
        seq.append(keypoints)

    if len(seq) == SEQ_LEN:
        print("예측 시작!")
        input_seq = np.array(seq)  # (30, 174)
        input_seq = (input_seq - mean) / std
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, 30, 174)

        with torch.no_grad():
            out = model(input_tensor)
            pred = out.argmax(dim=1).item()
            label = class_names[pred]
            prob = torch.softmax(out, dim=1)[0][pred].item()

        cv2.putText(frame, f"{label} ({prob:.2f})", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    cv2.imshow("Sign Prediction (Transformer)", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
