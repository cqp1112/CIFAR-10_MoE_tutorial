import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# =========================
# 1. 基本配置
# =========================
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = 256
learning_rate = 1e-3
num_epochs = 20

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# =========================
# 2. 数据预处理
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),                       # [0,255] -> [0,1]
    transforms.Normalize((0.5,), (0.5,))        # 简单标准化到大致 [-1,1]
])


# =========================
# 3. 下载数据集
# =========================
train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))


# =========================
# 4. 看几张样本图，帮助理解数据
# =========================
def show_samples(dataset, class_names, num_images=8):
    plt.figure(figsize=(12, 3))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.squeeze(0)   # [1,28,28] -> [28,28]

        plt.subplot(1, num_images, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(class_names[label], fontsize=9)
        plt.axis("off")

    plt.suptitle("FashionMNIST Samples")
    plt.tight_layout()
    plt.show()


show_samples(train_dataset, class_names)


# =========================
# 5. 定义一个小 CNN
# =========================
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
model = SmallCNN().to(device)
print(model)


# =========================
# 6. 损失函数和优化器
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# =========================
# 7. 训练函数
# =========================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


# =========================
# 8. 测试函数
# =========================
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


# =========================
# 9. 正式训练
# =========================
train_losses = []
train_accs = []
test_losses = []
test_accs = []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
    )


# =========================
# 10. 画训练曲线
# =========================
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.title("FashionMNIST Baseline - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(train_accs, label="train acc")
plt.plot(test_accs, label="test acc")
plt.title("FashionMNIST Baseline - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# 11. 看几张测试集预测结果
# =========================
def show_predictions(model, dataset, class_names, device, num_images=8):
    model.eval()

    plt.figure(figsize=(12, 3))

    with torch.no_grad():
        for i in range(num_images):
            image, label = dataset[i]
            input_tensor = image.unsqueeze(0).to(device)   # [1,1,28,28]

            output = model(input_tensor)
            pred = output.argmax(dim=1).item()

            image_show = image.squeeze(0).cpu().numpy()

            plt.subplot(1, num_images, i + 1)
            plt.imshow(image_show, cmap="gray")
            plt.title(f"P:{class_names[pred]}\nT:{class_names[label]}", fontsize=8)
            plt.axis("off")

    plt.suptitle("Baseline Predictions (P=Pred, T=True)")
    plt.tight_layout()
    plt.show()


show_predictions(model, test_dataset, class_names, device)