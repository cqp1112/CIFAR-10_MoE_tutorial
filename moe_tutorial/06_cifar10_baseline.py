import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# =========================
# 0. 路径配置：当前脚本目录/output
# =========================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_ROOT = "/mnt/data"

print("当前工作目录 BASE_DIR:", BASE_DIR)
print("输出目录 OUTPUT_DIR:", OUTPUT_DIR)
print("数据目录 DATA_ROOT:", DATA_ROOT)


# =========================
# 1. 基本配置
# =========================
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = 128
learning_rate = 3e-3
num_epochs = 60

num_classes = 10
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print("exists DATA_ROOT:", os.path.exists(DATA_ROOT))
if os.path.exists(DATA_ROOT):
    try:
        print("files under DATA_ROOT:", os.listdir(DATA_ROOT)[:30])
    except Exception as e:
        print("读取 DATA_ROOT 目录失败：", e)


# =========================
# 2. 数据增强与预处理
# =========================
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# =========================
# 3. 数据集与 DataLoader
#    不再联网下载，要求数据已存在
# =========================
train_dataset = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=False,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root=DATA_ROOT,
    train=False,
    download=False,
    transform=test_transform
)

# 单独准备一个“只用于展示原图”的 dataset
show_dataset = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))


# =========================
# 4. 看几张样本图
# =========================
def show_samples(dataset, class_names, num_images=8, save_path=None):
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "cifar10_baseline_samples.png")

    plt.figure(figsize=(12, 3))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0).cpu().numpy()  # [3,32,32] -> [32,32,3]

        plt.subplot(1, num_images, i + 1)
        plt.imshow(image)
        plt.title(class_names[label], fontsize=9)
        plt.axis("off")

    plt.suptitle("CIFAR-10 Baseline Samples")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"saved sample image to: {save_path}")


show_samples(show_dataset, class_names)


# =========================
# 5. 定义 baseline CNN
# =========================
class CIFAR10BaselineCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   # [B,64,16,16]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)    # [B,128,8,8]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CIFAR10BaselineCNN(num_classes=num_classes).to(device)
print(model)


# =========================
# 6. 损失函数、优化器、调度器
# =========================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=12,
    gamma=0.5
)


# =========================
# 7. 训练函数
# =========================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

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

best_test_acc = 0.0
best_epoch = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch + 1

        best_model_path = os.path.join(OUTPUT_DIR, "cifar10_baseline_best_cnn.pth")
        torch.save(model.state_dict(), best_model_path)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"LR: {current_lr:.6f} | "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
    )

print("\nBest Test Accuracy:", round(best_test_acc, 4))
print("Best Epoch:", best_epoch)


# =========================
# 10. 画训练曲线
# =========================
loss_curve_path = os.path.join(OUTPUT_DIR, "cifar10_baseline_loss_curve.png")
acc_curve_path = os.path.join(OUTPUT_DIR, "cifar10_baseline_accuracy_curve.png")

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.title("CIFAR-10 Baseline - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(loss_curve_path, dpi=200, bbox_inches="tight")
plt.show()
plt.close()
print(f"saved loss curve to: {loss_curve_path}")

plt.figure(figsize=(8, 5))
plt.plot(train_accs, label="train acc")
plt.plot(test_accs, label="test acc")
plt.title("CIFAR-10 Baseline - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(acc_curve_path, dpi=200, bbox_inches="tight")
plt.show()
plt.close()
print(f"saved acc curve to: {acc_curve_path}")


# =========================
# 11. 看几张测试集预测
# =========================
show_test_dataset = datasets.CIFAR10(
    root=DATA_ROOT,
    train=False,
    download=False,
    transform=transforms.ToTensor()
)

def show_predictions(model, dataset, class_names, device, num_images=8, save_path=None):
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "cifar10_baseline_predictions.png")

    model.eval()
    plt.figure(figsize=(12, 3))

    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )

    with torch.no_grad():
        for i in range(num_images):
            image, label = dataset[i]
            input_tensor = normalize(image).unsqueeze(0).to(device)

            output = model(input_tensor)
            pred = output.argmax(dim=1).item()

            image_show = image.permute(1, 2, 0).cpu().numpy()

            plt.subplot(1, num_images, i + 1)
            plt.imshow(image_show)
            plt.title(f"P:{class_names[pred]}\nT:{class_names[label]}", fontsize=8)
            plt.axis("off")

    plt.suptitle("CIFAR-10 Baseline Predictions")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"saved prediction image to: {save_path}")


show_predictions(model, show_test_dataset, class_names, device)


# =========================
# 12. 保存最终模型与训练记录
# =========================
final_model_path = os.path.join(OUTPUT_DIR, "cifar10_baseline_final_cnn.pth")
torch.save(model.state_dict(), final_model_path)
print(f"saved final model to: {final_model_path}")

metrics_path = os.path.join(OUTPUT_DIR, "cifar10_baseline_training_metrics.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(f"Best Test Accuracy: {best_test_acc:.6f}\n")
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write("\nEpoch-wise metrics:\n")
    for i in range(num_epochs):
        f.write(
            f"Epoch {i+1:03d} | "
            f"Train Loss: {train_losses[i]:.6f} | "
            f"Train Acc: {train_accs[i]:.6f} | "
            f"Test Loss: {test_losses[i]:.6f} | "
            f"Test Acc: {test_accs[i]:.6f}\n"
        )
print(f"saved training metrics to: {metrics_path}")


# =========================
# 13. 结束提示
# =========================
print("\nAll done.")
print("请到下面目录查看输出文件：")
print(OUTPUT_DIR)
