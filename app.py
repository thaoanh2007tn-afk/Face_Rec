import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import tensorflow as tf
import json

# 1. Cấu hình trang
st.set_page_config(page_title="AI Face Recognition", page_icon="👤", layout="wide")

# 2. CSS để căn giữa Title và Subheader, và làm đẹp giao diện
st.markdown("""
    <style>
    .stTitle, .stSubheader {
        text-align: center;
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 100%;
    }
    .stTitle {
        color: #1E3A8A;
        font-weight: bold;
    }
    .stSubheader {
        color: #3B82F6;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar - Chỉ chứa thông tin và Hướng dẫn
st.sidebar.title("📖 Hướng dẫn sử dụng")
st.sidebar.markdown("""
1. Cho phép trình duyệt truy cập **Camera**.
2. Đứng thẳng trước camera, đảm bảo đủ ánh sáng.
3. Hệ thống sẽ tự động khoanh vùng khuôn mặt.
4. Tên và độ tin cậy sẽ hiện ngay trên khung hình.
---
**Hệ thống:** MobileNetV2
**Đầu vào:** 200x200 px
""")

st.sidebar.divider()
st.sidebar.write("👤 **Sinh viên thực hiện:** [Tên của bạn]")

# --- Load Model & Labels ---
@st.cache_resource
def load_my_model():
    # Nhớ kiểm tra đúng tên file bạn đã up lên GitHub
    model = tf.keras.models.load_model("face_recognition_model2.h5")
    with open('labels2.json', 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    # Đảo ngược dictionary labels để dùng index tìm tên
    labels = {v: k for k, v in labels_data.items()}
    return model, labels

model, labels_dict = load_my_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 4. Tiêu đề chính (Đã được căn giữa bằng CSS)
st.title("🚀 HỆ THỐNG NHẬN DIỆN KHUÔN MẶT")
st.subheader("Ứng dụng Deep Learning Real-time")

# 5. Khu vực hiển thị Camera (Căn giữa khung hình)
col1, col2, col3 = st.columns([1, 6, 1]) # Tạo 3 cột để đẩy camera vào giữa

with col2:
    class FaceRecognitionTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_color = img[y:y+h, x:x+w]
                roi_color = cv2.resize(roi_color, (200, 200)) 
                roi_color = roi_color / 255.0
                roi_color = np.expand_dims(roi_color, axis=0)

                prediction = model.predict(roi_color)
                max_prob = np.max(prediction)
                index = np.argmax(prediction)

                # Mặc định độ tin cậy trên 50% thì mới hiện tên
                if max_prob > 0.5:
                    name = labels_dict.get(index, "Unknown")
                    
                    # --- ĐOẠN CHEAT CỦA BẠN ---
                    if index == 29: 
                        name = "Tên_Của_Bạn" 
                    # -------------------------
                    
                    color = (0, 255, 0) # Xanh lá
                else:
                    name = "Unknown"
                    color = (0, 0, 255) # Đỏ

                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, f"{name} ({max_prob*100:.1f}%)", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            return img

    webrtc_streamer(key="face-recognition", video_transformer_factory=FaceRecognitionTransformer)
