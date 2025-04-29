import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from data_loader import get_loaders, get_class_info
import os

if __name__ == '__main__':
    # 检查GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 获取数据加载器和类别信息
    train_loader, val_loader, test_loader = get_loaders()
    class_names, num_classes = get_class_info()

    # 加载预训练模型
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # 如果需要继续训练，加载之前保存的权重
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model_epoch10.pth')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"已加载权重: {checkpoint_path}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    # 训练参数
    num_epochs = 10  # 继续训练10个epoch

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1} 训练损失: {epoch_loss:.4f} 训练准确率: {epoch_acc:.4f}")

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss /= val_total
        val_acc = val_correct / val_total
        print(f"验证损失: {val_loss:.4f} 验证准确率: {val_acc:.4f}")

        # 保存模型
        save_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_epoch_continue_{epoch+1}.pth"))

    print("继续训练完成！")