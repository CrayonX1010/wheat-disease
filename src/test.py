import torch
import torch.nn as nn
from torchvision import models
from data_loader import get_loaders, get_class_info
import os

if __name__ == '__main__':
    # --- 配置 ---
    MODEL_PATH = r'd:\\prog2\\agriculture\\models\\best_model_epoch_continue_10.pth' # <--- 修改为你效果最好的模型文件路径
    # -----------

    # 检查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 获取数据加载器和类别信息
    _, _, test_loader = get_loaders() # 只需要测试加载器
    class_names, num_classes = get_class_info()

    # 加载模型结构 (与训练时相同)
    model = models.resnet18(weights=None) # 不加载预训练权重，因为我们要加载自己训练的权重
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 加载训练好的模型权重
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 {MODEL_PATH}")
        exit()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"成功加载模型: {MODEL_PATH}")
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        exit()

    model = model.to(device)
    model.eval() # 设置为评估模式

    # 损失函数 (用于计算损失，虽然测试时主要关注准确率)
    criterion = nn.CrossEntropyLoss()

    # 在测试集上评估
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad(): # 测试时不需要计算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # 计算并打印测试结果
    final_loss = test_loss / test_total
    final_acc = test_correct / test_total

    print(f"\n测试集评估结果:")
    print(f"测试损失: {final_loss:.4f}")
    print(f"测试准确率: {final_acc:.4f} ({test_correct}/{test_total})")

    print("\n测试完成！")