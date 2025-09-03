import cv2
import torch
import numpy as np
from collections import deque
import mediapipe as mp
from transformer_ctc_model import TransformerCTC

# === 설정 ===
SEQ_LEN = 30
INPUT_DIM = 244  # 244차원으로 변경
NUM_CLASS = 13  # 13개 클래스로 변경
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === confidence threshold 및 smoothing 설정 ===
CONF_THRESHOLD = 0.7
SMOOTHING_WINDOW = 5
recent_preds = deque(maxlen=SMOOTHING_WINDOW)

print("🎯 244차원 실시간 손동작 인식 시작")
print(f"🚀 Device: {DEVICE}")

# === 모델 로드 ===
model = TransformerCTC(input_dim=INPUT_DIM, num_classes=NUM_CLASS).to(DEVICE)
# === 모델 로드 ===
try:
    checkpoint = torch.load("best_ctc_transformer_244.pt", map_location=DEVICE)

    # 체크포인트에서 모델 설정 읽기
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        model = TransformerCTC(
            input_dim=config.get('input_dim', 244),
            num_classes=config.get('num_classes', 13),
            model_dim=config.get('model_dim', 256),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 6),
            dropout=0.1
        ).to(DEVICE)
        print(f"✅ 체크포인트 설정으로 모델 생성: {config}")
    else:
        # 기본 설정 사용
        model = TransformerCTC(
            input_dim=244,
            num_classes=13,
            model_dim=256,
            num_heads=8,
            num_layers=6,
            dropout=0.1
        ).to(DEVICE)
        print("⚠️ 기본 설정으로 모델 생성")

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 모델 로드 성공 (최고 정확도: {checkpoint.get('best_val_acc', 'N/A'):.2f}%)")
    else:
        model.load_state_dict(checkpoint)
        print("✅ 모델 로드 성공")

except FileNotFoundError:
    print("❌ 모델 파일을 찾을 수 없습니다: best_ctc_transformer_244.pt")
    print("💡 먼저 학습을 완료해주세요.")
    exit(1)
except Exception as e:
    print(f"❌ 모델 로드 오류: {e}")
    print("💡 모델 파일이 손상되었거나 호환되지 않습니다.")
    exit(1)

model.eval()

# === MediaPipe 초기화 ===
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === 버퍼 초기화 ===
seq = deque(maxlen=SEQ_LEN)

# 13개 클래스명 (실제 사용하는 클래스명으로 수정하세요)
class_names = [
    "감사합니다",  # 0
    "공부하다",  # 1
    "괜찮습니다",  # 2
    "나",  # 3
    "당신",  # 4
    "도착하다",  # 5
    "반갑습니다",  # 6
    "비수어",  # 7
    "싫다",  # 8
    "안녕하세요",  # 9
    "연락해주세요",  # 10
    "좋다멋지다",  # 11
    "죄송합니다"  # 12
]
blank_idx = NUM_CLASS  # CTC blank index (13)


# === 유틸 함수들 ===
def _safe_norm(v):
    """벡터 크기 안전 계산"""
    return float(np.linalg.norm(v))


def calculate_joint_angle(l1, l2, l3):
    """3점으로 관절 각도 계산"""
    v1 = l2 - l1
    v2 = l3 - l2
    n1, n2 = _safe_norm(v1), _safe_norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    v1 = v1 / n1
    v2 = v2 / n2
    dot_product = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return float(angle_deg)


def get_finger_states(hand_pts, eps=1e-6):
    """손가락 폄 상태 계산"""
    if hand_pts is None:
        return np.zeros(4, dtype=np.float32)

    finger_indices = [
        [5, 6, 7, 8],  # Index
        [9, 10, 11, 12],  # Middle
        [13, 14, 15, 16],  # Ring
        [17, 18, 19, 20]  # Pinky
    ]

    states = []
    for finger in finger_indices:
        v1 = hand_pts[finger[1]] - hand_pts[finger[0]]
        v2 = hand_pts[finger[2]] - hand_pts[finger[1]]

        n1, n2 = _safe_norm(v1), _safe_norm(v2)
        if n1 < eps or n2 < eps:
            states.append(0.0)
            continue

        cos = np.dot(v1, v2) / (n1 * n2)
        cos = float(np.clip(cos, -1.0, 1.0))
        state = (cos + 1.0) / 2.0
        states.append(state)

    return np.array(states, dtype=np.float32)


def get_palm_normal(hand_pts, eps=1e-6):
    """손바닥 법선벡터 계산"""
    if hand_pts is None:
        return np.zeros(3, dtype=np.float32)

    p0 = hand_pts[0]  # wrist
    p1 = hand_pts[5]  # index MCP
    p2 = hand_pts[17]  # pinky MCP

    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)
    n = _safe_norm(normal)
    if n < eps:
        return np.zeros(3, dtype=np.float32)
    return (normal / n).astype(np.float32)


def extract_landmarks(landmarks, indices):
    """특정 인덱스의 랜드마크 좌표 추출"""
    coords = []
    flag = 1
    if landmarks:
        for idx in indices:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                coords.extend([lm.x, lm.y, lm.z])
            else:
                flag = 0
                break
        if not flag:
            coords = [0.0] * (3 * len(indices))
    else:
        coords = [0.0] * (3 * len(indices))
        flag = 0
    return coords, flag


def extract_unified_features(results, previous_frame=None):
    """244차원 통합 특징 추출"""
    # === 1. 손 좌표 (126개) ===
    lh_coords, lh_flag = extract_landmarks(results.left_hand_landmarks, range(21))
    rh_coords, rh_flag = extract_landmarks(results.right_hand_landmarks, range(21))

    # 3D 배열로 변환
    lh_3d = np.array(lh_coords).reshape(21, 3) if lh_flag else None
    rh_3d = np.array(rh_coords).reshape(21, 3) if rh_flag else None

    # === 2. 상체 포즈 (27개) ===
    your_pose_idx = [11, 12, 13, 14, 15, 16, 23, 24]
    your_pose, pose_flag1 = extract_landmarks(results.pose_landmarks, your_pose_idx)

    # 힙센터 계산
    if results.pose_landmarks and len(results.pose_landmarks.landmark) >= 25:
        left_hip = results.pose_landmarks.landmark[23]
        right_hip = results.pose_landmarks.landmark[24]
        hip_center = [(left_hip.x + right_hip.x) / 2,
                      (left_hip.y + right_hip.y) / 2,
                      (left_hip.z + right_hip.z) / 2]
    else:
        hip_center = [0.5, 1.0, 0.0]

    pose_combined = your_pose + hip_center
    pose_flag = pose_flag1

    # === 3. 얼굴 좌표 (63개) ===
    your_face_idx = [61, 0, 13, 14, 17, 70, 107, 336, 296, 33, 133, 362, 263, 1, 2, 98, 327, 152, 234, 454]
    friend_face_idx = [1, 33, 263, 61, 291]

    overlapping = {1, 33, 263, 61}
    your_unique = [idx for idx in your_face_idx if idx not in overlapping]
    friend_unique = [idx for idx in friend_face_idx if idx not in overlapping]

    unified_face_idx = list(overlapping) + your_unique + friend_unique
    face_coords, face_flag = extract_landmarks(results.face_landmarks, unified_face_idx)

    # === 4. 관절 각도 (8개) ===
    if results.pose_landmarks and pose_flag:
        pose_lm = results.pose_landmarks.landmark
        l_shoulder = np.array([pose_lm[11].x, pose_lm[11].y, pose_lm[11].z])
        l_elbow = np.array([pose_lm[13].x, pose_lm[13].y, pose_lm[13].z])
        l_wrist = np.array([pose_lm[15].x, pose_lm[15].y, pose_lm[15].z])
        r_shoulder = np.array([pose_lm[12].x, pose_lm[12].y, pose_lm[12].z])
        r_elbow = np.array([pose_lm[14].x, pose_lm[14].y, pose_lm[14].z])
        r_wrist = np.array([pose_lm[16].x, pose_lm[16].y, pose_lm[16].z])
        hip_center_3d = np.array(hip_center)

        angle_l1 = calculate_joint_angle(l_shoulder, l_elbow, l_wrist)
        angle_r1 = calculate_joint_angle(r_shoulder, r_elbow, r_wrist)
        angle_l2 = calculate_joint_angle(l_elbow, l_shoulder, hip_center_3d)
        angle_r2 = calculate_joint_angle(r_elbow, r_shoulder, hip_center_3d)

        angles = [angle_l1, angle_l2, angle_r1, angle_r2]
        angles_rad = np.radians(angles)
        angle_features = []
        for angle in angles_rad:
            angle_features.extend([np.cos(angle), np.sin(angle)])
    else:
        angle_features = [0.0] * 8

    # === 5. 손가락 상태 (8개) ===
    left_finger_states = get_finger_states(lh_3d)
    right_finger_states = get_finger_states(rh_3d)
    finger_states = list(left_finger_states) + list(right_finger_states)

    # === 6. 손바닥 법선벡터 (6개) ===
    left_palm_normal = get_palm_normal(lh_3d)
    right_palm_normal = get_palm_normal(rh_3d)
    palm_normals = list(left_palm_normal) + list(right_palm_normal)

    # === 7. 속도 계산 (2개) ===
    speed_your = 0.0
    speed_friend = 0.0

    if previous_frame is not None:
        current_all = np.array(lh_coords + rh_coords + pose_combined + face_coords)
        prev_all = np.array(previous_frame[:len(current_all)])
        if len(prev_all) == len(current_all):
            speed_your = np.linalg.norm(current_all - prev_all) / len(current_all)

        current_hands = np.array(lh_coords + rh_coords).reshape(-1, 3)
        prev_hands = np.array(previous_frame[:126]).reshape(-1, 3)
        if len(prev_hands) == len(current_hands):
            speed_friend = float(np.linalg.norm(current_hands - prev_hands, axis=1).mean())

    # === 8. 마스크/플래그 (4개) ===
    flags = [lh_flag, rh_flag, pose_flag, face_flag]

    # === 최종 결합 ===
    unified_features = (
            lh_coords +  # 63개
            rh_coords +  # 63개
            pose_combined +  # 27개
            face_coords +  # 63개
            angle_features +  # 8개
            finger_states +  # 8개
            palm_normals +  # 6개
            [speed_your, speed_friend] +  # 2개
            flags  # 4개
    )

    return unified_features


# === Greedy decoding ===
def greedy_decode_ctc(logits, blank=blank_idx):
    pred = logits.argmax(dim=2)  # (B, T)
    decoded = []
    for p in pred:
        prev = blank
        seq = []
        for label in p:
            label = label.item()
            if label != blank and label != prev:
                seq.append(label)
            prev = label
        decoded.append(seq[0] if len(seq) > 0 else blank)
    return decoded


# === 실시간 예측 루프 ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다!")
    exit()

print("📹 카메라 시작... (종료: 'q' 키)")

previous_features = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 읽기 실패")
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True

    # 244차원 특징 추출
    features = extract_unified_features(results, previous_features)
    previous_features = features.copy()

    if len(features) == 244:
        seq.append(features)

    # 버퍼 상태 표시
    buffer_status = f"Buffer: {len(seq)}/{SEQ_LEN}"
    cv2.putText(frame, buffer_status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # MediaPipe 랜드마크 그리기
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # 예측 수행
    if len(seq) == SEQ_LEN:
        try:
            input_seq = np.array(seq)  # (30, 244)
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(input_tensor)
                log_probs = torch.softmax(logits, dim=2)
                preds = greedy_decode_ctc(log_probs, blank=blank_idx)

                label_idx = preds[0] if preds[0] != blank_idx else -1

                if 0 <= label_idx < len(class_names):
                    label = class_names[label_idx]
                    confidence = torch.max(log_probs).item()

                    recent_preds.append(label)
                    smoothed_label = max(set(recent_preds), key=recent_preds.count)

                    # 결과 표시
                    result_text = f"{smoothed_label} ({confidence:.2f})"
                    cv2.putText(frame, result_text, (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                    print(f"예측: {label} (신뢰도: {confidence:.3f})")
                else:
                    cv2.putText(frame, "None", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        except Exception as e:
            print(f"⚠️ 예측 오류: {e}")
            # 오류 시 버퍼 일부만 제거
            if len(seq) > 15:
                for _ in range(5):
                    seq.popleft()

    cv2.imshow("244D Sign Language Recognition (CTC)", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🎉 프로그램 종료")