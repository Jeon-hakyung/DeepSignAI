import numpy as np
import cv2
import os
import csv
import mediapipe as mp

# === 설정 ===
VIDEO_ROOT = "videos_augmented"  # 입력 폴더
OUTPUT_ROOT = "coords_unified"  # 출력 폴더
TARGET_CLASS = "안녕하세요"

print("💻 CPU 최적화 버전으로 실행합니다.")
print("🚀 손동작 데이터는 CPU가 더 효율적입니다!")

# MediaPipe Holistic 초기화
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)


# ==================== 유틸 함수들 ====================

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
    """손가락 폄 상태 계산 (검지~새끼 4개)"""
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


# ==================== 랜드마크 추출 ====================

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


def extract_unified_features(frame, previous_frame=None):
    """통합 특징 추출 (244차원) - CPU 최적화"""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    # === 1. 손 좌표 (126개) ===
    lh_coords, lh_flag = extract_landmarks(results.left_hand_landmarks, range(21))
    rh_coords, rh_flag = extract_landmarks(results.right_hand_landmarks, range(21))

    # 3D 배열로 변환 (손가락 상태/법선 계산용)
    lh_3d = np.array(lh_coords).reshape(21, 3) if lh_flag else None
    rh_3d = np.array(rh_coords).reshape(21, 3) if rh_flag else None

    # === 2. 상체 포즈 (27개) ===
    # 당신 방식: 8개 포인트
    your_pose_idx = [11, 12, 13, 14, 15, 16, 23, 24]
    your_pose, pose_flag1 = extract_landmarks(results.pose_landmarks, your_pose_idx)

    # 친구 방식: 6개 포인트 + 힙센터 (플래그 계산용만)
    friend_pose_idx = [11, 13, 15, 12, 14, 16]
    friend_pose, pose_flag2 = extract_landmarks(results.pose_landmarks, friend_pose_idx)

    # 힙센터 계산
    if results.pose_landmarks and len(results.pose_landmarks.landmark) >= 25:
        left_hip = results.pose_landmarks.landmark[23]
        right_hip = results.pose_landmarks.landmark[24]
        hip_center = [(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2, (left_hip.z + right_hip.z) / 2]
    else:
        hip_center = [0.5, 1.0, 0.0]

    pose_combined = your_pose + hip_center  # 8*3 + 3 = 27개
    pose_flag = pose_flag1 or pose_flag2

    # === 3. 얼굴 좌표 (63개) - 중복 제거 ===
    # 당신 방식: 20개 포인트
    your_face_idx = [61, 0, 13, 14, 17, 70, 107, 336, 296, 33, 133, 362, 263, 1, 2, 98, 327, 152, 234, 454]

    # 친구 방식: 5개 포인트
    friend_face_idx = [1, 33, 263, 61, 291]

    # 중복 제거: 겹치는 부분 [1, 33, 263, 61] 제거
    overlapping = {1, 33, 263, 61}
    your_unique = [idx for idx in your_face_idx if idx not in overlapping]  # 16개
    friend_unique = [idx for idx in friend_face_idx if idx not in overlapping]  # 1개 (291)

    # 통합 인덱스: 겹치는 부분 + 당신 고유 + 친구 고유
    unified_face_idx = list(overlapping) + your_unique + friend_unique  # 4 + 16 + 1 = 21개

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

        # sin/cos 변환
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
    speed_your = 0.0  # 당신 방식
    speed_friend = 0.0  # 친구 방식

    if previous_frame is not None:
        # 당신 방식: 전체 좌표 변화량
        current_all = np.array(lh_coords + rh_coords + pose_combined + face_coords)
        prev_all = np.array(previous_frame[:len(current_all)])
        if len(prev_all) == len(current_all):
            speed_your = np.linalg.norm(current_all - prev_all) / len(current_all)

        # 친구 방식: 손 좌표 평균 변화량
        current_hands = np.array(lh_coords + rh_coords).reshape(-1, 3)
        prev_hands = np.array(previous_frame[:126]).reshape(-1, 3)
        if len(prev_hands) == len(current_hands):
            speed_friend = float(np.linalg.norm(current_hands - prev_hands, axis=1).mean())

    # === 8. 마스크/플래그 (4개) ===
    flags = [lh_flag, rh_flag, pose_flag, face_flag]

    # === 최종 결합 ===
    unified_features = (
            lh_coords +  # 63개 (손)
            rh_coords +  # 63개 (손)
            pose_combined +  # 27개 (포즈)
            face_coords +  # 63개 (얼굴)
            angle_features +  # 8개 (각도)
            finger_states +  # 8개 (손가락 상태)
            palm_normals +  # 6개 (손바닥 법선)
            [speed_your, speed_friend] +  # 2개 (속도)
            flags  # 4개 (플래그)
    )

    return unified_features


def process_video(input_path, output_path):
    """비디오 처리하여 CSV 저장 - CPU 최적화"""
    cap = cv2.VideoCapture(input_path)
    previous_frame = None

    with open(output_path, "w", newline='') as f:
        writer = csv.writer(f)

        # 헤더 작성
        header = []
        # 손 좌표
        for hand in ['left', 'right']:
            for i in range(21):
                header.extend([f'{hand}_hand_{i}_x', f'{hand}_hand_{i}_y', f'{hand}_hand_{i}_z'])

        # 포즈 좌표
        pose_names = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hip', 'r_hip',
                      'hip_center']
        for name in pose_names:
            header.extend([f'{name}_x', f'{name}_y', f'{name}_z'])

        # 얼굴 좌표 (중복 제거된 21개)
        for i in range(21):
            header.extend([f'face_{i}_x', f'face_{i}_y', f'face_{i}_z'])

        # 각도 (sin/cos)
        angle_names = ['l_elbow', 'l_shoulder', 'r_elbow', 'r_shoulder']
        for name in angle_names:
            header.extend([f'{name}_cos', f'{name}_sin'])

        # 손가락 상태
        for hand in ['left', 'right']:
            for finger in ['index', 'middle', 'ring', 'pinky']:
                header.append(f'{hand}_{finger}_state')

        # 손바닥 법선
        for hand in ['left', 'right']:
            header.extend([f'{hand}_palm_nx', f'{hand}_palm_ny', f'{hand}_palm_nz'])

        # 속도 & 플래그
        header.extend(['speed_your', 'speed_friend', 'lh_flag', 'rh_flag', 'pose_flag', 'face_flag'])

        writer.writerow(header)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            features = extract_unified_features(frame, previous_frame)
            writer.writerow(features)
            previous_frame = features.copy()
            frame_count += 1

            if frame_count % 100 == 0:
                print(f"  처리된 프레임: {frame_count}")

    cap.release()
    print(f"  총 {frame_count}프레임 처리 완료")


def main():
    """메인 실행 함수"""
    print("🎯 CPU 최적화 손동작 특징 추출기")
    print("=" * 50)

    # 출력 폴더 생성
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 지정된 클래스 폴더 확인
    class_dir = os.path.join(VIDEO_ROOT, TARGET_CLASS)
    if not os.path.isdir(class_dir):
        print(f"🚫 {TARGET_CLASS} 폴더가 존재하지 않습니다.")
        return

    output_class_dir = os.path.join(OUTPUT_ROOT, TARGET_CLASS)
    os.makedirs(output_class_dir, exist_ok=True)

    # 비디오 파일 처리
    video_files = [f for f in os.listdir(class_dir)
                   if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    print(f"📹 {len(video_files)}개 비디오 파일 발견")
    print("🚀 CPU 최적화 통합 특징 추출 시작...\n")

    for i, fname in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] 처리 중: {fname}")
        video_path = os.path.join(class_dir, fname)
        name, _ = os.path.splitext(fname)
        output_path = os.path.join(output_class_dir, name + ".csv")

        try:
            process_video(video_path, output_path)
            print(f"✅ {name}.csv 완료 (244차원)\n")
        except Exception as e:
            print(f"❌ {fname} 처리 실패: {e}\n")

    print("🎉 CPU 최적화 통합 특징 추출 완료!")
    print(f"📊 최종 차원: 244개")
    print(f"📁 저장 위치: {output_class_dir}")


if __name__ == "__main__":
    main()