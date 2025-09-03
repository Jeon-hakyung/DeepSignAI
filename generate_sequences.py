import os
import csv
import numpy as np

# === 설정 ===
CSV_ROOT = "coords_unified"  # 새로운 통합 특징 폴더
SEQ_LEN = 30
OVERLAP = 10

LABEL_MAP = {
    "감사합니다": 0,
    "공부하다": 1,
    "괜찮습니다": 2,
    "나": 3,
    "당신": 4,
    "도착하다": 5,
    "반갑습니다": 6,
    "비수어": 7,
    "싫다": 8,
    "안녕하세요": 9,
    "연락해주세요": 10,
    "좋다멋지다": 11,
    "죄송합니다": 12
}


def load_csv_244(filepath):
    """244차원 통합 특징 CSV 로드"""
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # 헤더 스킵
        data = []
        total = 0
        skipped = 0

        for row in reader:
            total += 1
            if len(row) == 244:  # 244차원 통합 특징
                # 숫자 변환 (플래그는 정수로)
                converted_row = []
                for i, val in enumerate(row):
                    try:
                        # 마지막 4개는 플래그 (정수)
                        if i >= 240:  # lh_flag, rh_flag, pose_flag, face_flag
                            converted_row.append(int(float(val)))
                        else:
                            converted_row.append(float(val))
                    except:
                        converted_row.append(0.0)

                data.append(converted_row)
            else:
                skipped += 1

        if skipped > 0:
            print(f"⚠️ {filepath}: {skipped}/{total} 프레임 스킵됨 ({skipped / total:.1%})")
        return np.array(data)


def csv_to_sequences_244(csv_array):
    """244차원 특징을 시퀀스로 변환"""
    sequences = []
    total_frames = len(csv_array)
    step = SEQ_LEN - OVERLAP

    for start in range(0, total_frames - SEQ_LEN + 1, step):
        clip = csv_array[start:start + SEQ_LEN]  # (30, 244)
        # 244차원을 그대로 유지하거나 필요시 reshape
        sequences.append(clip)

    return sequences


def main():
    print("🎯 244차원 통합 특징 시퀀스 변환기")
    print("=" * 50)

    X_data = []
    y_data = []

    for class_name in os.listdir(CSV_ROOT):
        class_path = os.path.join(CSV_ROOT, class_name)
        if not os.path.isdir(class_path):
            continue

        label = LABEL_MAP.get(class_name)
        if label is None:
            print(f"🚫 {class_name} 라벨 매핑 없음 → 스킵")
            continue

        print(f"\n📁 처리 중: {class_name} (라벨: {label})")
        class_sequences = 0

        for fname in os.listdir(class_path):
            if fname.endswith(".csv"):
                filepath = os.path.join(class_path, fname)
                try:
                    csv_array = load_csv_244(filepath)
                    if len(csv_array) >= SEQ_LEN:  # 최소 길이 체크
                        seqs = csv_to_sequences_244(csv_array)
                        X_data.extend(seqs)
                        y_data.extend([label] * len(seqs))
                        class_sequences += len(seqs)
                        print(f"  ✅ {fname} → {len(seqs)} 시퀀스")
                    else:
                        print(f"  ⚠️ {fname} → 너무 짧음 ({len(csv_array)} < {SEQ_LEN})")
                except Exception as e:
                    print(f"  ❌ {fname} → 오류: {e}")

        print(f"📊 {class_name}: 총 {class_sequences}개 시퀀스")

    if len(X_data) == 0:
        print("❌ 생성된 시퀀스가 없습니다!")
        return

    X_data = np.array(X_data)  # (N, 30, 244)
    y_data = np.array(y_data)  # (N,)

    print(f"\n🎯 최종 결과:")
    print(f"📊 전체 시퀀스 수: {len(X_data)}")
    print(f"📏 X shape: {X_data.shape}")
    print(f"📏 y 형태: {y_data.shape}")
    print(f"🏷️ 클래스별 분포:")

    # 클래스별 개수 출력
    unique, counts = np.unique(y_data, return_counts=True)
    for class_id, count in zip(unique, counts):
        class_name = [k for k, v in LABEL_MAP.items() if v == class_id][0]
        print(f"   {class_name}: {count}개")

    # 저장
    np.save("X_unified_244.npy", X_data)
    np.save("y_unified_244.npy", y_data)
    print("\n✅ 시퀀스 저장 완료:")
    print("   📄 X_unified_244.npy")
    print("   📄 y_unified_244.npy")


if __name__ == "__main__":
    main()