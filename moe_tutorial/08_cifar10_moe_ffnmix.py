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
learning_rate = 0.01
num_epochs = 60

num_classes = 10
num_experts = 4
top_k = 2
expert_hidden_dim = 256

# 中和版超参
gate_temperature = 1.2
balance_loss_weight = 0.25
gate_lr_scale = 0.35

# gate noise：前期略强保护，后期更稳
initial_gate_noise_std = 0.12
later_gate_noise_std = 0.03
noise_transition_epoch = 28

# warmup 略缩短，但保留
warmup_epochs = 10

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
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),  # CIFAR-10的均值(R, G, B)
        (0.2470, 0.2435, 0.2616)   # CIFAR-10的标准差(R, G, B)
    ),
])

# 用于 clean train eval（不做随机增强）
train_eval_transform = transforms.Compose([
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
# =========================
train_dataset = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=False,
    transform=train_transform
)

train_eval_dataset = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=False,
    transform=train_eval_transform
)

test_dataset = datasets.CIFAR10(
    root=DATA_ROOT,
    train=False,
    download=False,
    transform=test_transform
)

show_dataset = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

show_test_dataset = datasets.CIFAR10(
    root=DATA_ROOT,
    train=False,
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

train_eval_loader = DataLoader(
    train_eval_dataset,
    batch_size=batch_size,
    shuffle=False,
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
        save_path = os.path.join(OUTPUT_DIR, "cifar10_moe_samples.png")

    plt.figure(figsize=(12, 3))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0).cpu().numpy()

        plt.subplot(1, num_images, i + 1)
        plt.imshow(image)
        plt.title(class_names[label], fontsize=9)
        plt.axis("off")

    plt.suptitle("CIFAR-10 FFN-MoE Samples")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"saved sample image to: {save_path}")


show_samples(show_dataset, class_names)


# =========================
# 5. 定义专家：完整 FFN，直接输出 logits
# =========================
class FFNExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 6. 可切换“均匀路由 / top-k 路由”的 FFN-MoE 头
# =========================
class WarmupTopKFFNMixMoEHead(nn.Module):
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
        self.num_classes = num_classes
        self.temperature = temperature
        self.gate_noise_std = gate_noise_std

        self.gate = nn.Linear(input_dim, num_experts)

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        self.experts = nn.ModuleList([
            FFNExpert(input_dim, hidden_dim, num_classes)
            for _ in range(num_experts)
        ])

    def forward(self, x, mode="uniform"):
        batch_size = x.size(0)

        if mode == "uniform":
            full_gate_probs = torch.full(
                (batch_size, self.num_experts),
                1.0 / self.num_experts,
                device=x.device,
                dtype=x.dtype
            )
            route_probs = full_gate_probs

        elif mode == "topk":
            gate_logits = self.gate(x)

            if self.training and self.gate_noise_std > 0:
                gate_logits = gate_logits + self.gate_noise_std * torch.randn_like(gate_logits)

            full_gate_probs = torch.softmax(gate_logits / self.temperature, dim=1)

            topk_vals, topk_idx = torch.topk(full_gate_probs, k=self.top_k, dim=1)
            topk_vals = topk_vals / topk_vals.sum(dim=1, keepdim=True)

            route_probs = torch.zeros_like(full_gate_probs)
            route_probs.scatter_(1, topk_idx, topk_vals)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        expert_logits = []
        for expert in self.experts:
            logits_i = expert(x)
            expert_logits.append(logits_i)

        expert_logits = torch.stack(expert_logits, dim=1)     # [B, E, C]
        route_probs_expanded = route_probs.unsqueeze(-1)      # [B, E, 1]
        mixed_logits = (route_probs_expanded * expert_logits).sum(dim=1)

        return mixed_logits, route_probs, expert_logits, full_gate_probs


# =========================
# 7. 整体模型
# =========================
class CIFAR10MoECNN(nn.Module):
    def __init__(
        self,
        num_classes=10,
        num_experts=4,
        top_k=2,
        expert_hidden_dim=320,
        gate_temperature=1.0,
        gate_noise_std=0.0
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)
        )

        self.flatten = nn.Flatten()

        self.moe_head = WarmupTopKFFNMixMoEHead(
            input_dim=128 * 8 * 8,
            hidden_dim=expert_hidden_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            top_k=top_k,
            temperature=gate_temperature,
            gate_noise_std=gate_noise_std
        )

    def forward(self, x, mode="uniform"):
        x = self.features(x)
        x = self.flatten(x)
        logits, gate_probs, expert_logits, full_gate_probs = self.moe_head(x, mode=mode)
        return logits, gate_probs, expert_logits, full_gate_probs


model = CIFAR10MoECNN(
    num_classes=num_classes,
    num_experts=num_experts,
    top_k=top_k,
    expert_hidden_dim=expert_hidden_dim,
    gate_temperature=gate_temperature,
    gate_noise_std=initial_gate_noise_std
).to(device)

print(model)


# =========================
# 8. 损失函数、优化器、调度器
# =========================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# gate 单独更小学习率
gate_params = list(model.moe_head.gate.parameters())
gate_param_ids = set(id(p) for p in gate_params)

base_params = [p for p in model.parameters() if id(p) not in gate_param_ids]

optimizer = optim.Adam(
    [
        {"params": base_params, "lr": learning_rate},
        {"params": gate_params, "lr": learning_rate * gate_lr_scale},
    ],
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[learning_rate, learning_rate * gate_lr_scale],
    epochs=num_epochs,
    steps_per_epoch=len(train_loader)
)


# =========================
# 9. 冻结 / 解冻 gate
# =========================
def set_gate_trainable(model, trainable: bool):
    for param in model.moe_head.gate.parameters():
        param.requires_grad = trainable


# =========================
# 10. 训练一个 epoch
# =========================
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, num_experts, balance_loss_weight, mode):
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_balance_loss = 0.0
    total_correct = 0
    total_samples = 0

    route_gate_sum = torch.zeros(num_experts, device=device)
    full_gate_sum = torch.zeros(num_experts, device=device)

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, gate_probs, _, full_gate_probs = model(images, mode=mode)
        cls_loss = criterion(logits, labels)

        if mode == "uniform":
            balance_loss = torch.tensor(0.0, device=device)
        else:
            mean_gate = full_gate_probs.mean(dim=0)
            uniform_target = torch.full_like(mean_gate, 1.0 / num_experts)
            balance_loss = ((mean_gate - uniform_target) ** 2).mean()

        loss = cls_loss + balance_loss_weight * balance_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        total_cls_loss += cls_loss.item() * images.size(0)
        total_balance_loss += balance_loss.item() * images.size(0)

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        route_gate_sum += gate_probs.sum(dim=0)
        full_gate_sum += full_gate_probs.sum(dim=0)

    avg_loss = total_loss / total_samples
    avg_cls_loss = total_cls_loss / total_samples
    avg_balance_loss = total_balance_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_route_gate = route_gate_sum / total_samples
    avg_full_gate = full_gate_sum / total_samples

    return (
        avg_loss,
        avg_cls_loss,
        avg_balance_loss,
        avg_acc,
        avg_route_gate.detach().cpu(),
        avg_full_gate.detach().cpu()
    )


# =========================
# 11. 评估
# =========================
def evaluate(model, dataloader, criterion, device, num_experts, mode):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    route_gate_sum = torch.zeros(num_experts, device=device)
    full_gate_sum = torch.zeros(num_experts, device=device)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, gate_probs, _, full_gate_probs = model(images, mode=mode)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            route_gate_sum += gate_probs.sum(dim=0)
            full_gate_sum += full_gate_probs.sum(dim=0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_route_gate = route_gate_sum / total_samples
    avg_full_gate = full_gate_sum / total_samples

    return avg_loss, avg_acc, avg_route_gate.detach().cpu(), avg_full_gate.detach().cpu()


def evaluate_clean_train(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, _, _, _ = model(images, mode="topk")
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


# =========================
# 12. 正式训练
# =========================
train_losses = []
train_accs = []
test_losses = []
test_accs = []
clean_train_accs = []

best_test_acc = 0.0
best_epoch = 0

for epoch in range(num_epochs):
    use_uniform = (epoch + 1 <= warmup_epochs)
    mode = "uniform" if use_uniform else "topk"

    set_gate_trainable(model, trainable=(not use_uniform))

    if use_uniform:
        current_gate_noise_std = initial_gate_noise_std
    else:
        if (epoch + 1) <= noise_transition_epoch:
            current_gate_noise_std = initial_gate_noise_std
        else:
            current_gate_noise_std = later_gate_noise_std

    model.moe_head.gate_noise_std = current_gate_noise_std

    (
        train_loss,
        train_cls_loss,
        train_balance_loss,
        train_acc,
        train_route_gate,
        train_full_gate
    ) = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_experts,
        balance_loss_weight,
        mode=mode
    )

    test_loss, test_acc, test_route_gate, test_full_gate = evaluate(
        model,
        test_loader,
        criterion,
        device,
        num_experts,
        mode=mode
    )

    clean_train_loss, clean_train_acc = evaluate_clean_train(
        model,
        train_eval_loader,
        criterion,
        device
    )

    current_lr_backbone = optimizer.param_groups[0]["lr"]
    current_lr_gate = optimizer.param_groups[1]["lr"]

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    clean_train_accs.append(clean_train_acc)

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch + 1

        best_model_path = os.path.join(OUTPUT_DIR, "cifar10_moe_ffnmix_best_model.pth")
        torch.save(model.state_dict(), best_model_path)

    mode_name = "UNIFORM-WARMUP" if use_uniform else "TOP-2"

    print(f"\nEpoch [{epoch+1}/{num_epochs}]  Mode: {mode_name}")
    print(f"Backbone/Expert LR:     {current_lr_backbone:.6f}")
    print(f"Gate LR:                {current_lr_gate:.6f}")
    print(f"Gate Noise Std:         {current_gate_noise_std:.4f}")
    print(f"Train Total Loss:       {train_loss:.4f}")
    print(f"Train Class Loss:       {train_cls_loss:.4f}")
    print(f"Train Balance Loss:     {train_balance_loss:.6f}")
    print(f"Train Acc:              {train_acc:.4f}")
    print(f"Clean Train Acc:        {clean_train_acc:.4f}")
    print(f"Test  Loss:             {test_loss:.4f}")
    print(f"Test  Acc:              {test_acc:.4f}")
    print("Train route gate avg:", [round(x, 4) for x in train_route_gate.tolist()])
    print("Test  route gate avg:", [round(x, 4) for x in test_route_gate.tolist()])
    print("Train full  gate avg:", [round(x, 4) for x in train_full_gate.tolist()])
    print("Test  full  gate avg:", [round(x, 4) for x in test_full_gate.tolist()])

print("\nBest Test Accuracy:", round(best_test_acc, 4))
print("Best Epoch:", best_epoch)


# =========================
# 13. 画训练曲线
# =========================
loss_curve_path = os.path.join(OUTPUT_DIR, "cifar10_moe_ffnmix_loss_curve.png")
acc_curve_path = os.path.join(OUTPUT_DIR, "cifar10_moe_ffnmix_accuracy_curve.png")

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="train total loss")
plt.plot(test_losses, label="test loss")
plt.title("CIFAR-10 FFN-MoE - Loss")
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
plt.plot(clean_train_accs, label="clean train acc")
plt.plot(test_accs, label="test acc")
plt.title("CIFAR-10 FFN-MoE - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(acc_curve_path, dpi=200, bbox_inches="tight")
plt.show()
plt.close()
print(f"saved accuracy curve to: {acc_curve_path}")


# =========================
# 14. 看几张测试集预测
# =========================
normalize = transforms.Normalize(
    (0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2616)
)

def show_predictions(model, dataset, class_names, device, num_images=8, save_path=None):
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "cifar10_moe_ffnmix_predictions.png")

    model.eval()
    plt.figure(figsize=(12, 3))

    with torch.no_grad():
        for i in range(num_images):
            image, label = dataset[i]
            input_tensor = normalize(image).unsqueeze(0).to(device)

            logits, gate_probs, _, _ = model(input_tensor, mode="topk")
            pred = logits.argmax(dim=1).item()

            top2 = torch.topk(gate_probs, k=2, dim=1)
            top2_idx = top2.indices.squeeze(0).tolist()
            top2_val = top2.values.squeeze(0).tolist()

            image_show = image.permute(1, 2, 0).cpu().numpy()

            plt.subplot(1, num_images, i + 1)
            plt.imshow(image_show)
            plt.title(
                f"P:{class_names[pred]}\nT:{class_names[label]}\nE:{top2_idx}\nW:{[round(v, 2) for v in top2_val]}",
                fontsize=7
            )
            plt.axis("off")

    plt.suptitle("CIFAR-10 FFN-MoE Predictions")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"saved prediction image to: {save_path}")


show_predictions(model, show_test_dataset, class_names, device)


# =========================
# 15. 统计：每个类别偏向哪些专家
# =========================
def analyze_class_expert_preference(model, dataloader, class_names, device, num_experts, save_path=None):
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "cifar10_moe_ffnmix_class_expert_preference.txt")

    model.eval()

    class_gate_sum = torch.zeros(len(class_names), num_experts, device=device)
    class_count = torch.zeros(len(class_names), device=device)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            _, gate_probs, _, _ = model(images, mode="topk")

            for c in range(len(class_names)):
                mask = (labels == c)
                if mask.any():
                    class_gate_sum[c] += gate_probs[mask].sum(dim=0)
                    class_count[c] += mask.sum()

    class_avg_gate = class_gate_sum / class_count.unsqueeze(1)

    print("\n=== Class -> Expert Preference (Top-2 after uniform warmup) ===")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== Class -> Expert Preference (Top-2 after uniform warmup) ===\n")
        for c, class_name in enumerate(class_names):
            probs = class_avg_gate[c].detach().cpu().tolist()
            best_experts = torch.topk(class_avg_gate[c], k=2).indices.tolist()

            line = (
                f"{class_name:12s} -> "
                f"{[round(x, 4) for x in probs]}   "
                f"top-2 experts = {best_experts}"
            )
            print(line)
            f.write(line + "\n")

    print(f"saved class expert preference to: {save_path}")


analyze_class_expert_preference(model, test_loader, class_names, device, num_experts)


# =========================
# 16. 保存最终模型与训练记录
# =========================
final_model_path = os.path.join(OUTPUT_DIR, "cifar10_moe_ffnmix_final_model.pth")
torch.save(model.state_dict(), final_model_path)
print(f"saved final model to: {final_model_path}")

metrics_path = os.path.join(OUTPUT_DIR, "cifar10_moe_ffnmix_training_metrics.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(f"Best Test Accuracy: {best_test_acc:.6f}\n")
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Warmup Epochs: {warmup_epochs}\n")
    f.write(f"Num Experts: {num_experts}\n")
    f.write(f"Top-k: {top_k}\n")
    f.write(f"Expert Hidden Dim: {expert_hidden_dim}\n")
    f.write(f"Gate Temperature: {gate_temperature}\n")
    f.write(f"Balance Loss Weight: {balance_loss_weight}\n")
    f.write(f"Gate LR Scale: {gate_lr_scale}\n")
    f.write(f"Initial Gate Noise Std: {initial_gate_noise_std}\n")
    f.write(f"Later Gate Noise Std: {later_gate_noise_std}\n")
    f.write(f"Noise Transition Epoch: {noise_transition_epoch}\n")
    f.write("\nEpoch-wise metrics:\n")
    for i in range(num_epochs):
        f.write(
            f"Epoch {i+1:03d} | "
            f"Train Loss: {train_losses[i]:.6f} | "
            f"Train Acc: {train_accs[i]:.6f} | "
            f"Clean Train Acc: {clean_train_accs[i]:.6f} | "
            f"Test Loss: {test_losses[i]:.6f} | "
            f"Test Acc: {test_accs[i]:.6f}\n"
        )
print(f"saved training metrics to: {metrics_path}")


# =========================
# 17. 结束提示
# =========================
print("\nAll done.")
print("请到下面目录查看输出文件：")
print(OUTPUT_DIR)