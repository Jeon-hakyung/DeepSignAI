import os
import csv
import numpy as np

# === 설정 ===
CSV_ROOT = "coords"  # 클래스별 폴더 (예: coords/감사합니다/*.csv)
SEQ_LEN = 30
OVERLAP = 5

LABEL_MAP = {
    "감사합니다": 0,
    "연락해주세요": 1,
    "비수어": 2,  # 필요 시 추가
}

def load_csv(filepath):
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        data = []
        total = 0
        skipped = 0
        for row in reader:
            total += 1
            if len(row) == 174:  # 58 관절 x (x, y, z)
                data.append(list(map(float, row)))
            else:
                skipped += 1

        if skipped > 0:
            print(f"⚠️ {filepath}: {skipped}/{total} 프레임 스킵됨 ({skipped / total:.1%})")
        return np.array(data)

def csv_to_sequences(csv_array):
    sequences = []
    total_frames = len(csv_array)
    step = SEQ_LEN - OVERLAP
    for start in range(0, total_frames - SEQ_LEN + 1, step):
        clip = csv_array[start:start + SEQ_LEN]          # (30, 174)
        clip = clip.reshape(SEQ_LEN, 58, 3)              # (30, 58, 3)
        clip = clip.transpose(2, 0, 1)                   # → (3, 30, 58)
        sequences.append(clip)
    return sequences

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

    for fname in os.listdir(class_path):
        if fname.endswith(".csv"):
            filepath = os.path.join(class_path, fname)
            csv_array = load_csv(filepath)
            seqs = csv_to_sequences(csv_array)
            X_data.extend(seqs)
            y_data.extend([label] * len(seqs))
            print(f"✅ {fname} → {len(seqs)} 시퀀스 생성")

X_data = np.array(X_data)  # (N, 3, 30, 58)
y_data = np.array(y_data)  # (N,)

print(f"\n🎯 전체 시퀀스 수: {len(X_data)}")
np.save("X_holistic_58.npy", X_data)
np.save("y_holistic_58.npy", y_data)
print("✅ 시퀀스 저장 완료: X_holistic_58.npy, y_holistic_58.npy")
