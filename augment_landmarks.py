import numpy as np
from scipy.ndimage import zoom

# === 설정 ===
INPUT_FILE_X = "X_unified_244.npy"
INPUT_FILE_Y = "y_unified_244.npy"
OUTPUT_FILE_X = "X_unified_244_aug.npy"
OUTPUT_FILE_Y = "y_unified_244_aug.npy"

JITTER_STD = 0.05
TIME_WARP_RATE = [0.9, 1.1]

# 라벨별 증강 배수 (클래스 불균형 해결)
AUG_CONFIG = {
    0: 1,  # 감사합니다
    1: 1,  # 공부하다
    2: 4,  # 괜찮습니다
    3: 4,  # 나
    4: 4,  # 당신
    5: 4,  # 도착하다
    6: 5,  # 반갑습니다
    7: 2,  # 비수어
    8: 2,  # 싫다
    9: 4,  # 안녕하세요
    10: 4,  # 연락해주세요
    11: 2,  # 좋다멋지다
    12: 4  # 죄송합니다
}

print("🎯 244차원 통합 특징 데이터 증강기")
print("=" * 50)

# === 데이터 로드 ===
try:
    X = np.load(INPUT_FILE_X).astype(np.float32)  # (N, 30, 244)
    y = np.load(INPUT_FILE_Y)
    print(f"✅ 데이터 로드 완료: X={X.shape}, y={y.shape}")

    if X.shape[2] != 244:
        print(f"⚠️ 경고: 예상 244차원이지만 실제 {X.shape[2]}차원입니다.")
except FileNotFoundError:
    print("❌ 파일을 찾을 수 없습니다. 먼저 sequence_converter_244.py를 실행하세요.")
    exit(1)

# 클래스별 개수 확인
unique, counts = np.unique(y, return_counts=True)
print(f"\n📊 클래스별 원본 데이터:")
for label, count in zip(unique, counts):
    print(f"  클래스 {label}: {count}개")


# === 증강 함수 ===

def jitter(seq, std=JITTER_STD):
    """노이즈 추가 (좌표 부분만)"""
    noise = np.random.normal(0, std, seq.shape).astype(np.float32)
    # 플래그 부분은 노이즈 제외 (마지막 4차원)
    noise[:, 240:244] = 0.0
    return (seq + noise).astype(np.float32)


def time_warp(seq, rate):
    """시간 축 변형"""
    warped = zoom(seq, (rate, 1), order=1).astype(np.float32)
    if len(warped) > seq.shape[0]:
        return warped[:seq.shape[0]]
    else:
        pad = np.zeros((seq.shape[0] - len(warped), seq.shape[1]), dtype=np.float32)
        return np.vstack([warped, pad])


def horizontal_flip(seq):
    """좌우 반전 (손 좌표만)"""
    seq = seq.copy()

    # 손 좌표 x값만 반전 (0-125: 손 좌표, 3개씩 x,y,z)
    for i in range(0, 126, 3):  # 손 좌표 126개 중 x좌표만
        seq[:, i] = 1.0 - seq[:, i]

    # 포즈 x좌표 반전 (126-152: 포즈 좌표)
    for i in range(126, 153, 3):  # 포즈 좌표 27개 중 x좌표만
        seq[:, i] = 1.0 - seq[:, i]

    # 얼굴 x좌표 반전 (153-215: 얼굴 좌표)
    for i in range(153, 216, 3):  # 얼굴 좌표 63개 중 x좌표만
        seq[:, i] = 1.0 - seq[:, i]

    return seq.astype(np.float32)


def random_frame_drop(seq, drop_prob=0.2):
    """랜덤 프레임 드롭"""
    seq = seq.copy()
    for i in range(seq.shape[0]):
        if np.random.rand() < drop_prob:
            # 좌표는 0으로, 플래그는 유지
            seq[i, :240] = 0.0
    return seq.astype(np.float32)


def speed_variation(seq, speed_factor=None):
    """속도 변화 증강"""
    if speed_factor is None:
        speed_factor = np.random.uniform(0.8, 1.2)

    # 속도 관련 특징 (236-237번째) 수정
    seq = seq.copy()
    seq[:, 236:238] *= speed_factor  # speed_your, speed_friend
    return seq.astype(np.float32)


# === 증강 수행 ===
aug_X, aug_y = [], []

print("\n🚀 데이터 증강 시작...")

for i in range(len(X)):
    seq, label = X[i], y[i]

    # 원본 추가
    aug_X.append(seq.astype(np.float32))
    aug_y.append(label)

    # 증강 수행
    aug_count = AUG_CONFIG.get(label, 0)
    for aug_idx in range(aug_count):
        if label == 7:  # 비수어 특화 증강
            new_seq = random_frame_drop(seq, drop_prob=0.3)
            if np.random.rand() < 0.5:
                new_seq = horizontal_flip(new_seq)
        else:
            # 일반 증강
            new_seq = seq.copy()

            # 50% 확률로 각 증강 적용
            if np.random.rand() < 0.7:
                new_seq = jitter(new_seq)

            if np.random.rand() < 0.5:
                new_seq = time_warp(new_seq, np.random.uniform(*TIME_WARP_RATE))

            if np.random.rand() < 0.4:
                new_seq = horizontal_flip(new_seq)

            if np.random.rand() < 0.3:
                new_seq = speed_variation(new_seq)

        aug_X.append(new_seq.astype(np.float32))
        aug_y.append(label)

    if (i + 1) % 100 == 0:
        print(f"  처리 완료: {i + 1}/{len(X)}")

# === 배열화 및 저장 ===
aug_X = np.array(aug_X, dtype=np.float32)
aug_y = np.array(aug_y)

print(f"\n📊 증강 결과:")
print(f"원본: X={X.shape}, y={y.shape}")
print(f"증강: X={aug_X.shape}, y={aug_y.shape}")
print(f"증가율: {len(aug_X) / len(X):.1f}배")

# 클래스별 증강 결과
unique_aug, counts_aug = np.unique(aug_y, return_counts=True)
print(f"\n📈 클래스별 증강 후 데이터:")
for label, count in zip(unique_aug, counts_aug):
    original_count = counts[unique == label][0] if label in unique else 0
    increase = count / original_count if original_count > 0 else 0
    print(f"  클래스 {label}: {original_count} → {count} ({increase:.1f}배)")

# 저장
np.save(OUTPUT_FILE_X, aug_X)
np.save(OUTPUT_FILE_Y, aug_y)

print(f"\n🎉 저장 완료!")
print(f"📄 {OUTPUT_FILE_X}")
print(f"📄 {OUTPUT_FILE_Y}")
print(f"\n💡 이제 이 파일들을 학습에 사용할 수 있습니다!")