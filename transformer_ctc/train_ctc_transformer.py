import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

from transformer_ctc_model import TransformerCTC

# === 하이퍼파라미터 ===
BATCH_SIZE = 16
EPOCHS = 30
LR = 0.0001
NUM_CLASS = 13  # 실제 클래스 개수 (blank 제외) - 13개로 변경
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🎯 244차원 CTC Transformer 학습 시작")
print("=" * 50)
print("🚀 Device 설정:", DEVICE)

if DEVICE.type == "cuda":
    print("✅ GPU 사용 중:", torch.cuda.get_device_name(DEVICE))
    print("   할당된 메모리:", round(torch.cuda.memory_allocated(DEVICE) / 1024 ** 2, 2), "MB")
else:
    print("⚠️ GPU를 사용하지 않고 CPU만 사용 중입니다.")

# === 데이터 로드 ===
print("\n📂 데이터 로딩...")
X = np.load("X_unified_244_aug.npy")  # (N, T, 244) - 파일명 변경
y = np.load("y_unified_244_aug.npy")  # (N,) - 파일명 변경

print(f"✅ 데이터 로드 완료: X={X.shape}, y={y.shape}")
print(f"📊 시퀀스 길이: {X.shape[1]}")
print(f"📏 특징 차원: {X.shape[2]}")

# 클래스별 분포 확인
unique, counts = np.unique(y, return_counts=True)
print(f"\n📈 클래스별 데이터 분포:")
for class_id, count in zip(unique, counts):
    print(f"  클래스 {class_id}: {count}개")

seq_len = X.shape[1]

# === CTC용 라벨 처리 ===
targets = [torch.tensor([label], dtype=torch.long) for label in y]  # 각 샘플당 1개의 클래스
target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

X = torch.tensor(X, dtype=torch.float32)
y = torch.cat(targets)  # (N,)

# === 데이터 분할 ===
print("\n🔄 데이터 분할 (Train:Val = 8:2)")
X_train, X_val, y_train, y_val, tgt_len_train, tgt_len_val = train_test_split(
    X, y, target_lengths, test_size=0.2, stratify=y, random_state=42
)

print(f"📊 학습 데이터: {len(X_train)}개")
print(f"📊 검증 데이터: {len(X_val)}개")

train_dataset = TensorDataset(X_train, y_train, tgt_len_train)
val_dataset = TensorDataset(X_val, y_val, tgt_len_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"🔢 배치 수 - 학습: {len(train_loader)}, 검증: {len(val_loader)}")

# === 모델 초기화 ===
print("\n🧠 모델 초기화...")
model = TransformerCTC(
    input_dim=244,  # 244차원으로 변경
    num_classes=NUM_CLASS,  # 13개 클래스
    model_dim=256,
    num_heads=8,
    num_layers=6,
    dropout=0.1
).to(DEVICE)

criterion = nn.CTCLoss(blank=NUM_CLASS, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

# 학습률 스케줄러 추가
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

best_val_loss = float("inf")
best_val_acc = 0.0

print(f"\n📋 학습 설정:")
print(f"  배치 크기: {BATCH_SIZE}")
print(f"  에포크: {EPOCHS}")
print(f"  학습률: {LR}")
print(f"  클래스 수: {NUM_CLASS}")


# === Greedy decoding 함수 ===
def greedy_decode(logits, blank=NUM_CLASS):
    """
    logits: (B, T, C+1)
    return: (B,) 예측된 class
    """
    pred = logits.argmax(dim=2)  # (B, T)
    decoded = []
    for p in pred:
        # CTC 규칙: blank 제거, 연속된 같은 라벨 합치기
        prev = blank
        seq = []
        for label in p:
            label = label.item()
            if label != blank and label != prev:
                seq.append(label)
            prev = label
        # 이번 경우는 라벨 하나만 남기므로 첫번째 라벨 사용
        decoded.append(seq[0] if len(seq) > 0 else blank)
    return decoded


# === 학습 루프 ===
print(f"\n🚀 학습 시작!")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):
    # === 학습 단계 ===
    model.train()
    train_loss, train_correct = 0.0, 0

    for batch_idx, (x_batch, y_batch, tgt_lens) in enumerate(train_loader):
        x_batch, y_batch, tgt_lens = x_batch.to(DEVICE), y_batch.to(DEVICE), tgt_lens.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x_batch)  # (B, T, C+1)
        log_probs = nn.functional.log_softmax(logits, dim=2)
        input_lens = torch.full((logits.size(0),), logits.size(1), dtype=torch.long).to(DEVICE)

        loss = criterion(log_probs.permute(1, 0, 2), y_batch, input_lens, tgt_lens)

        # gradient clipping 추가
        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # === 정확도 계산 ===
        with torch.no_grad():
            preds = greedy_decode(log_probs, blank=NUM_CLASS)
            train_correct += (torch.tensor(preds).to(DEVICE) == y_batch).sum().item()

    # === 검증 단계 ===
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for x_batch, y_batch, tgt_lens in val_loader:
            x_batch, y_batch, tgt_lens = x_batch.to(DEVICE), y_batch.to(DEVICE), tgt_lens.to(DEVICE)

            logits = model(x_batch)
            log_probs = nn.functional.log_softmax(logits, dim=2)
            input_lens = torch.full((logits.size(0),), logits.size(1), dtype=torch.long).to(DEVICE)

            loss = criterion(log_probs.permute(1, 0, 2), y_batch, input_lens, tgt_lens)
            if not torch.isnan(loss) and not torch.isinf(loss):
                val_loss += loss.item()

            # === 정확도 계산 ===
            preds = greedy_decode(log_probs, blank=NUM_CLASS)
            val_correct += (torch.tensor(preds).to(DEVICE) == y_batch).sum().item()

    # === 메트릭 계산 ===
    avg_train_loss = train_loss / len(train_loader) if train_loss > 0 else float('inf')
    avg_val_loss = val_loss / len(val_loader) if val_loss > 0 else float('inf')
    train_acc = train_correct / len(train_dataset) * 100
    val_acc = val_correct / len(val_dataset) * 100

    # 학습률 스케줄러 업데이트
    scheduler.step(avg_val_loss)

    print(f"[Epoch {epoch:2d}/{EPOCHS}] "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:5.2f}% | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:5.2f}%")

    # === 최고 모델 저장 ===
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'model_config': {
                'input_dim': 244,
                'num_classes': NUM_CLASS,
                'model_dim': 256,
                'num_heads': 8,
                'num_layers': 6
            }
        }, "best_ctc_transformer_244.pt")
        print(f"✅ Best model 업데이트! (Val Acc: {val_acc:.2f}%)")

    # GPU 메모리 정리
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

print("\n🎉 CTC Transformer 학습 완료!")
print("=" * 50)
print(f"🏆 최고 검증 정확도: {best_val_acc:.2f}%")
print(f"📄 모델 저장됨: best_ctc_transformer_244.pt")

# === 최종 모델 정보 출력 ===
print(f"\n📊 최종 모델 정보:")
model_info = model.get_model_info()
for key, value in model_info.items():
    if isinstance(value, int):
        print(f"  {key}: {value:,}")
    else:
        print(f"  {key}: {value}")