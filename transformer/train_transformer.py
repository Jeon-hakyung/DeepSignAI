# train_transformer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformer_model import SignTransformer
from sklearn.model_selection import train_test_split
import numpy as np

# === 하이퍼파라미터 ===
BATCH_SIZE = 16
EPOCHS = 30
LR = 0.0005
NUM_CLASS = 3  # 클래스 개수에 따라 조정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 데이터 로딩 및 분할 ===
X = np.load("../X_holistic_58.npy")  # (N, 3, 30, 58)
y = np.load("../y_holistic_58.npy")  # (N,)

# === 정규화 처리 ===
# (N, 3, 30, 58) → (N, 174) 기준 평균, 표준편차 계산
X_reshaped = X.transpose(0, 2, 1, 3).reshape(-1, 174)  # (N*T, 174)
mean = X_reshaped.mean(axis=0)
std = X_reshaped.std(axis=0) + 1e-6  # 분모 0 방지용 작은 값 추가
X = (X.transpose(0, 2, 1, 3).reshape(-1, 174) - mean) / std
X = X.reshape(-1, 30, 3, 58).transpose(0, 2, 1, 3)  # 다시 (N, 3, 30, 58)

# === 정규화 값 저장 ===
np.save("mean.npy", mean)
np.save("std.npy", std)
print("✅ mean.npy, std.npy 저장 완료")


X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === 모델 구성 ===
model = SignTransformer(input_dim=174, num_classes=NUM_CLASS).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0

# === 학습 루프 ===
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        train_correct += (outputs.argmax(1) == y).sum().item()

    # === Validation ===
    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)

            val_loss += loss.item() * x.size(0)
            val_correct += (outputs.argmax(1) == y).sum().item()

    train_acc = train_correct / len(train_dataset) * 100
    val_acc = val_correct / len(val_dataset) * 100

    print(f"[Epoch {epoch}/{EPOCHS}] Train Loss: {train_loss / len(train_dataset):.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss / len(val_dataset):.4f}, Val Acc: {val_acc:.2f}%")

    # best model 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_transformer_sign_model.pt")

print("✅ Transformer 모델 학습 완료 및 best 모델 저장!")
