import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置matplotlib中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 配置参数
IMG_SIZE = 224
MODEL_PATH = r'd:\prog2\agriculture\models\best_model_epoch_continue_10.pth'  # 使用您最好的模型

# 加载类别信息 - 英文名和中文名对照
class_names = ['Aphid', 'Black Rust', 'Blast', 'Brown Rust', 'Common Root Rot', 
               'Fusarium Head Blight', 'Healthy', 'Leaf Blight', 'Mildew', 
               'Mite', 'Septoria', 'Smut', 'Stem fly', 'Tan spot', 'Yellow Rust']

# 中文名称对照表
chinese_names = {
    'Aphid': '蚜虫',
    'Black Rust': '黑锈病',
    'Blast': '稻瘟病',
    'Brown Rust': '褐锈病',
    'Common Root Rot': '根腐病',
    'Fusarium Head Blight': '赤霉病',
    'Healthy': '健康',
    'Leaf Blight': '叶枯病',
    'Mildew': '霉病',
    'Mite': '螨虫',
    'Septoria': '小麦纹枯病',
    'Smut': '黑穗病',
    'Stem fly': '茎蝇',
    'Tan spot': '褐斑病',
    'Yellow Rust': '黄锈病'
}

num_classes = len(class_names)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载模型
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        return model, device
    else:
        st.error(f"模型文件未找到: {MODEL_PATH}")
        return None, None

# 预测函数
def predict_image(image, model, device):
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        
    # 获取所有类别的概率
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    top_5_prob, top_5_idx = torch.topk(probabilities, 5)
    
    return top_5_prob.cpu().numpy(), [class_names[idx] for idx in top_5_idx.cpu().numpy()]

# 自定义CSS样式
def local_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        margin-bottom: 30px;
        padding-bottom: 10px;
        border-bottom: 2px solid #3498db;
    }
    h3 {
        color: #2980b9;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: 500;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .upload-box {
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-box {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .highlight {
        font-weight: bold;
        color: #e74c3c;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# 主函数
def main():
    st.set_page_config(
        page_title="小麦病虫害识别系统",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # 应用自定义CSS
    local_css()
    
    st.title("🌱 小麦病虫害识别系统")
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>上传一张农作物图片，系统将识别可能的病虫害类型</p>", unsafe_allow_html=True)
    
    # 加载模型
    model, device = load_model()
    
    if model is None:
        st.stop()
    
    # 创建两列布局
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("<h3>📷 上传图片</h3>", unsafe_allow_html=True)
        
        # 创建上传区域
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="上传的图片", use_container_width=True)
            
            # 添加预测按钮
            if st.button("🔍 开始识别"):
                with st.spinner("正在分析图片..."):
                    # 预测
                    probs, classes = predict_image(image, model, device)
                    
                    # 在第二列显示结果
                    with col2:
                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.markdown("<h3>🔍 识别结果</h3>", unsafe_allow_html=True)
                        
                        # 显示主要预测结果
                        main_class = classes[0]
                        main_prob = probs[0] * 100
                        
                        st.markdown(f"""
                        <div style='text-align: center; margin: 20px 0;'>
                            <h2>预测结果: <span class='highlight'>{main_class}</span> ({chinese_names[main_class]})</h2>
                            <p style='font-size: 18px;'>置信度: {main_prob:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 显示概率条形图 (只使用英文标签)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        y_pos = np.arange(len(classes))
                        
                        # 水平条形图
                        bars = ax.barh(y_pos, probs * 100, align='center', color='#3498db')
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(classes)  # 只使用英文标签
                        ax.invert_yaxis()  # 标签从上到下
                        ax.set_xlabel('Probability (%)')
                        ax.set_title('Top 5 Predictions')
                        
                        # 在条形上添加百分比标签
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                                    f'{width:.1f}%', ha='left', va='center')
                        
                        # 设置x轴范围，留出标签空间
                        ax.set_xlim(0, 110)
                        
                        # 设置图表样式
                        fig.patch.set_facecolor('#f5f7f9')
                        ax.set_facecolor('#f5f7f9')
                        
                        st.pyplot(fig)
                        
                        # 显示详细结果表格 (包含中英文对照)
                        st.markdown("### 详细预测结果")
                        result_data = {
                            "英文名称": classes,
                            "中文名称": [chinese_names[c] for c in classes],
                            "概率 (%)": [f"{p*100:.2f}%" for p in probs]
                        }
                        st.table(result_data)
                        st.markdown("</div>", unsafe_allow_html=True)
    
    # 如果没有上传图片，在右侧显示说明
    if uploaded_file is None:
        with col2:
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown("<h3>📋 使用说明</h3>", unsafe_allow_html=True)
            st.markdown("""
            <ol>
                <li>在左侧上传一张农作物图片（支持JPG、JPEG、PNG格式）</li>
                <li>点击"开始识别"按钮</li>
                <li>系统将分析图片并显示识别结果</li>
                <li>结果包括预测的病虫害类型及其概率</li>
            </ol>
            <p>本系统可识别的病虫害类型包括：</p>
            <ul style='columns: 2;'>
            """, unsafe_allow_html=True)
            
            for en, cn in chinese_names.items():
                st.markdown(f"<li>{en} ({cn})</li>", unsafe_allow_html=True)
            
            st.markdown("""
            </ul>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()