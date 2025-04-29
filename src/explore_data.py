import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# 定义数据集路径
base_dir = r'd:\prog2\agriculture\data' # 使用原始字符串避免转义问题
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

print("数据探索：")

# 获取训练集中的类别
categories = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
print(f"发现 {len(categories)} 个类别: {categories}")

# 打印每个类别的训练图像数量
print("\n训练集中每个类别的图像数量:")
for category in categories:
    category_path = os.path.join(train_dir, category)
    num_images = len(os.listdir(category_path))
    print(f"- {category}: {num_images} 张图像")

# 随机显示一个训练图像样本
try:
    random_category = random.choice(categories)
    random_category_path = os.path.join(train_dir, random_category)
    random_image_name = random.choice(os.listdir(random_category_path))
    random_image_path = os.path.join(random_category_path, random_image_name)

    img = mpimg.imread(random_image_path)
    plt.imshow(img)
    plt.title(f"样本图像: {random_category} / {random_image_name}")
    plt.axis('off') # 不显示坐标轴
    plt.show()
except Exception as e:
    print(f"\n无法显示样本图像: {e}")
    print("请确保 matplotlib 已正确安装，并且数据集路径和结构正确。")

print("\n数据探索完成。")