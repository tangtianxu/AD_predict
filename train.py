import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from MLP_model import MLP
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
# 加载数据
data = pd.read_csv('data/preprocessed_data.csv')

# 分离特征与标签
X = data.drop(columns=["DX"])
y = data["DX"]

# 切分训练集、验证集和测试集，比例为3:1:1
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 将数据转换为Tensor并创建DataLoader
train_data = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32),
                           torch.tensor(y_train.values, dtype=torch.long))
val_data = TensorDataset(torch.tensor(X_val.values, dtype=torch.float32),
                         torch.tensor(y_val.values, dtype=torch.long))
test_data = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32),
                          torch.tensor(y_test.values, dtype=torch.long))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 定义模型、损失函数和优化器
model = MLP(input_dim=X_train.shape[1], output_dim=5)
# 计算类的权重
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32)
# 将类权重传递给损失函数
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 200
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
best_val_accuracy = 0.0  # 初始验证准确率设为 0
best_model_wts = None  # 用于保存最好的模型权重
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        # print(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = correct / total * 100
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")
    train_losses.append(running_loss)
    train_accuracies.append(train_accuracy)
    # 验证集
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = val_correct / val_total * 100
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%")
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # 如果当前验证准确率更好，则保存模型权重
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_wts = model.state_dict()  # 保存当前最好的模型权重
# 恢复最好的模型权重
model.load_state_dict(best_model_wts)

# 评估测试集性能
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

# 计算准确率、精确率、召回率、F1分数
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# 打印结果
print(f"Test Accuracy: {accuracy:.2f}")
print(f"Test Precision: {precision:.2f}")
print(f"Test Recall: {recall:.2f}")
print(f"Test F1 Score: {f1:.2f}")

# 计算并显示混淆矩阵
conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
epochs_range = list(range(1, epochs + 1))
# 创建一个 1x2 的子图，左边绘制 Loss 曲线，右边绘制 Accuracy 曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# 绘制训练和验证损失
ax1.plot(epochs_range, train_losses, label='Training Loss', color='blue')
ax1.plot(epochs_range, val_losses, label='Validation Loss', color='red')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)
# 绘制训练和验证正确率
ax2.plot(epochs_range, train_accuracies, label='Training Accuracy', color='blue')
ax2.plot(epochs_range, val_accuracies, label='Validation Accuracy', color='red')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)
# 显示图形
plt.tight_layout()
plt.show()

# 保存模型
torch.save(model.state_dict(), "model/MLP_model.pth")
print("Model saved to MLP_model.pth")
