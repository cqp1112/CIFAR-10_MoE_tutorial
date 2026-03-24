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
learning_rate = 3e-3
num_epochs = 20

num_classes = 10
num_experts = 4
top_k = 2
expert_hidden_dim = 256
gate_temperature = 1.0
balance_loss_weight = 0.2
gate_noise_std = 0.1

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# =========================
# 2. 数据预处理
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# =========================
# 3. 数据集与 DataLoader
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
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))


# =========================
# 4. 看几张样本图
# =========================
def show_samples(dataset, class_names, num_images=8):
    plt.figure(figsize=(12, 3))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.squeeze(0)

        plt.subplot(1, num_images, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(class_names[label], fontsize=9)
        plt.axis("off")

    plt.suptitle("FashionMNIST Samples")
    plt.tight_layout()
    plt.show()


show_samples(train_dataset, class_names)


# =========================
# 5. 定义专家：输出 hidden
# =========================
class HiddenExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.net(x)   # [B, hidden_dim]


# =========================
# 6. Top-2 HiddenMix MoE 头（带 gate 噪声）
# =========================
class TopKHiddenMixMoEHead(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        num_experts,
        top_k=2,
        temperature=1.0,
        gate_noise_std=0.0
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.gate_noise_std = gate_noise_std

        self.gate = nn.Linear(input_dim, num_experts)

        self.experts = nn.ModuleList([
            HiddenExpert(input_dim, hidden_dim)
            for _ in range(num_experts)
        ])

        self.shared_classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: [B, input_dim]

        返回:
            logits: [B, num_classes]
            sparse_gate_probs: [B, num_experts]
            mixed_hidden: [B, hidden_dim]
            full_gate_probs: [B, num_experts]
        """
        gate_logits = self.gate(x)  # [B, E]

        # 训练时加入少量噪声，防止过早路由锁死
        if self.training and self.gate_noise_std > 0:
            gate_logits = gate_logits + self.gate_noise_std * torch.randn_like(gate_logits)

        # 完整 softmax 分布，仅用于观测/可选分析
        full_gate_probs = torch.softmax(gate_logits / self.temperature, dim=1)  # [B, E]

        # top-k 路由
        topk_vals, topk_idx = torch.topk(full_gate_probs, k=self.top_k, dim=1)  # [B, K], [B, K]

        # 对 top-k 再归一化
        topk_vals = topk_vals / topk_vals.sum(dim=1, keepdim=True)

        # 构造稀疏 gate
        sparse_gate_probs = torch.zeros_like(full_gate_probs)
        sparse_gate_probs.scatter_(1, topk_idx, topk_vals)

        # 所有专家先算 hidden（教学上更简单）
        expert_hiddens = []
        for expert in self.experts:
            h = expert(x)  # [B, H]
            expert_hiddens.append(h)

        expert_hiddens = torch.stack(expert_hiddens, dim=1)  # [B, E, H]

        # 稀疏混合 hidden
        sparse_gate_expanded = sparse_gate_probs.unsqueeze(-1)  # [B, E, 1]
        mixed_hidden = (sparse_gate_expanded * expert_hiddens).sum(dim=1)  # [B, H]

        logits = self.shared_classifier(mixed_hidden)  # [B, C]

        return logits, sparse_gate_probs, mixed_hidden, full_gate_probs


# =========================
# 7. 整体模型：CNN + Top-2 HiddenMix MoE
# =========================
class CNNTopKHiddenMoE(nn.Module):
    def __init__(
        self,
        num_classes=10,
        num_experts=4,
        top_k=2,
        expert_hidden_dim=256,
        gate_temperature=1.0,
        gate_noise_std=0.0
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # [B, 32, 14, 14]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)    # [B, 64, 7, 7]
        )

        self.flatten = nn.Flatten()

        self.moe_head = TopKHiddenMixMoEHead(
            input_dim=64 * 7 * 7,
            hidden_dim=expert_hidden_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            top_k=top_k,
            temperature=gate_temperature,
            gate_noise_std=gate_noise_std
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        logits, gate_probs, mixed_hidden, full_gate_probs = self.moe_head(x)
        return logits, gate_probs, mixed_hidden, full_gate_probs


model = CNNTopKHiddenMoE(
    num_classes=num_classes,
    num_experts=num_experts,
    top_k=top_k,
    expert_hidden_dim=expert_hidden_dim,
    gate_temperature=gate_temperature,
    gate_noise_std=gate_noise_std
).to(device)

print(model)


# =========================
# 8. 损失函数、优化器、调度器
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# =========================
# 9. 训练一个 epoch
# =========================
def train_one_epoch(model, dataloader, criterion, optimizer, device, num_experts, balance_loss_weight):
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_balance_loss = 0.0
    total_correct = 0
    total_samples = 0

    sparse_gate_sum = torch.zeros(num_experts, device=device)
    full_gate_sum = torch.zeros(num_experts, device=device)

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, gate_probs, _, full_gate_probs = model(images)

        # 分类损失
        cls_loss = criterion(logits, labels)

        # 负载均衡损失：
        # 用完整 gate 分布做均衡约束，比直接对稀疏 top-k 分布约束更温和
        mean_gate = full_gate_probs.mean(dim=0)  # [E]
        uniform_target = torch.full_like(mean_gate, 1.0 / num_experts)
        balance_loss = ((mean_gate - uniform_target) ** 2).mean()

        loss = cls_loss + balance_loss_weight * balance_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_cls_loss += cls_loss.item() * images.size(0)
        total_balance_loss += balance_loss.item() * images.size(0)

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        sparse_gate_sum += gate_probs.sum(dim=0)
        full_gate_sum += full_gate_probs.sum(dim=0)

    avg_loss = total_loss / total_samples
    avg_cls_loss = total_cls_loss / total_samples
    avg_balance_loss = total_balance_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_sparse_gate = sparse_gate_sum / total_samples
    avg_full_gate = full_gate_sum / total_samples

    return (
        avg_loss,
        avg_cls_loss,
        avg_balance_loss,
        avg_acc,
        avg_sparse_gate.detach().cpu(),
        avg_full_gate.detach().cpu()
    )


# =========================
# 10. 测试
# =========================
def evaluate(model, dataloader, criterion, device, num_experts):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    sparse_gate_sum = torch.zeros(num_experts, device=device)
    full_gate_sum = torch.zeros(num_experts, device=device)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, gate_probs, _, full_gate_probs = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            sparse_gate_sum += gate_probs.sum(dim=0)
            full_gate_sum += full_gate_probs.sum(dim=0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_sparse_gate = sparse_gate_sum / total_samples
    avg_full_gate = full_gate_sum / total_samples

    return avg_loss, avg_acc, avg_sparse_gate.detach().cpu(), avg_full_gate.detach().cpu()


# =========================
# 11. 训练主循环
# =========================
train_losses = []
train_accs = []
test_losses = []
test_accs = []

best_test_acc = 0.0
best_epoch = 0

for epoch in range(num_epochs):
    (
        train_loss,
        train_cls_loss,
        train_balance_loss,
        train_acc,
        train_sparse_gate,
        train_full_gate
    ) = train_one_epoch(
        model, train_loader, criterion, optimizer, device, num_experts, balance_loss_weight
    )

    test_loss, test_acc, test_sparse_gate, test_full_gate = evaluate(
        model, test_loader, criterion, device, num_experts
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

    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    print(f"LR:                     {current_lr:.6f}")
    print(f"Train Total Loss:       {train_loss:.4f}")
    print(f"Train Class Loss:       {train_cls_loss:.4f}")
    print(f"Train Balance Loss:     {train_balance_loss:.6f}")
    print(f"Train Acc:              {train_acc:.4f}")
    print(f"Test  Loss:             {test_loss:.4f}")
    print(f"Test  Acc:              {test_acc:.4f}")
    print("Train sparse gate avg:", [round(x, 4) for x in train_sparse_gate.tolist()])
    print("Test  sparse gate avg:", [round(x, 4) for x in test_sparse_gate.tolist()])
    print("Train full   gate avg:", [round(x, 4) for x in train_full_gate.tolist()])
    print("Test  full   gate avg:", [round(x, 4) for x in test_full_gate.tolist()])

print("\nBest Test Accuracy:", round(best_test_acc, 4))
print("Best Epoch:", best_epoch)


# =========================
# 12. 画训练曲线
# =========================
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="train total loss")
plt.plot(test_losses, label="test loss")
plt.title("FashionMNIST MoE v5b Top-2 HiddenMix - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(train_accs, label="train acc")
plt.plot(test_accs, label="test acc")
plt.title("FashionMNIST MoE v5b Top-2 HiddenMix - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# 13. 看几张测试集预测
# =========================
def show_predictions(model, dataset, class_names, device, num_images=8):
    model.eval()
    plt.figure(figsize=(12, 3))

    with torch.no_grad():
        for i in range(num_images):
            image, label = dataset[i]
            input_tensor = image.unsqueeze(0).to(device)

            logits, gate_probs, _, full_gate_probs = model(input_tensor)
            pred = logits.argmax(dim=1).item()

            top2 = torch.topk(gate_probs, k=2, dim=1)
            top2_idx = top2.indices.squeeze(0).tolist()
            top2_val = top2.values.squeeze(0).tolist()

            image_show = image.squeeze(0).cpu().numpy()

            plt.subplot(1, num_images, i + 1)
            plt.imshow(image_show, cmap="gray")
            plt.title(
                f"P:{class_names[pred]}\nT:{class_names[label]}\nE:{top2_idx}\nW:{[round(v, 2) for v in top2_val]}",
                fontsize=7
            )
            plt.axis("off")

    plt.suptitle("MoE v5b Top-2 Predictions")
    plt.tight_layout()
    plt.show()


show_predictions(model, test_dataset, class_names, device)


# =========================
# 14. 统计：每个类别偏向哪些专家
# =========================
def analyze_class_expert_preference(model, dataloader, class_names, device, num_experts):
    model.eval()

    class_gate_sum = torch.zeros(len(class_names), num_experts, device=device)
    class_count = torch.zeros(len(class_names), device=device)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            _, gate_probs, _, _ = model(images)

            for c in range(len(class_names)):
                mask = (labels == c)
                if mask.any():
                    class_gate_sum[c] += gate_probs[mask].sum(dim=0)
                    class_count[c] += mask.sum()

    class_avg_gate = class_gate_sum / class_count.unsqueeze(1)

    print("\n=== Class -> Expert Preference (sparse top-2 gate) ===")
    for c, class_name in enumerate(class_names):
        probs = class_avg_gate[c].detach().cpu().tolist()
        best_experts = torch.topk(class_avg_gate[c], k=2).indices.tolist()
        print(
            f"{class_name:12s} -> "
            f"{[round(x, 4) for x in probs]}   "
            f"top-2 experts = {best_experts}"
        )


analyze_class_expert_preference(model, test_loader, class_names, device, num_experts)