import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --- 配置参数 ---
IMG_SIZE = 224 # 图像统一调整后的大小，可以根据需要调整
BATCH_SIZE = 32 # 每个批次加载的图像数量

# --- 数据路径 ---
base_dir = r'd:\prog2\agriculture\data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# --- 数据预处理和增强 ---
# 定义训练集的数据变换：随机裁剪、随机水平翻转、转换为Tensor、归一化
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet 均值和标准差
])

# 定义验证集和测试集的数据变换：调整大小、中心裁剪、转换为Tensor、归一化
val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 创建 Dataset ---
# 使用 ImageFolder 自动从文件夹结构加载数据和标签
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
validation_dataset = datasets.ImageFolder(validation_dir, transform=val_test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

# --- 创建 DataLoader ---
# DataLoader 用于批量加载数据
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) # num_workers 可根据你的 CPU 核心数调整
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# --- 获取类别信息 ---
class_names = train_dataset.classes
num_classes = len(class_names)
print(f"数据集类别: {class_names}")
print(f"类别数量: {num_classes}")
print(f"训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(validation_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")

# --- 导出信息供其他脚本使用 ---
def get_loaders():
    return train_loader, validation_loader, test_loader

def get_class_info():
    return class_names, num_classes

if __name__ == '__main__':
    # 测试数据加载器
    print("\n测试加载一个批次的训练数据...")
    try:
        # 获取一个批次的数据
        inputs, classes = next(iter(train_loader))
        print(f"图像批次形状: {inputs.shape}") # 应为 [BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE]
        print(f"标签批次形状: {classes.shape}") # 应为 [BATCH_SIZE]
        print("数据加载器工作正常。")
    except Exception as e:
        print(f"加载数据时出错: {e}")
        print("请检查数据集路径、文件完整性以及依赖库是否正确安装。")