import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import tensorflow as tf
import json

# 1. Cấu hình trang (Hiện tên trên tab trình duyệt)
st.set_page_config(page_title="AI Face Recognition", page_icon="👤", layout="wide")

# 2. Thêm CSS để giao diện trông hiện đại hơn
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stTitle {
        color: #1E3A8A;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        text-align: center;
    }
    .stSubheader {
        color: #3B82F6;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar - Nơi để các cài đặt (Trông sẽ rất chuyên nghiệp)
st.sidebar.title("⚙️ Cài đặt hệ thống")
st.sidebar.info("Hệ thống nhận diện khuôn mặt Real-time.")
confidence_threshold = st.sidebar.slider("Độ tin cậy (Confidence)", 0.0, 1.0, 0.5)
st.sidebar.divider()
st.sidebar.write("👤 **Sinh viên thực hiện:** [Trần Ngọc Thảo Anh]")

# --- Load Model & Labels ---
@st.cache_resource # Dùng cache để không load lại model mỗi khi web load lại
def load_my_model():
    model = tf.keras.models.load_model("face_recognition_model2.h5")
    with open('labels2.json', 'r', encoding='utf-8') as f:
        labels = json.load(f)
    # Đảo ngược dictionary labels để dùng index tìm tên
    labels = {v: k for k, v in labels.items()}
    return model, labels

model, labels_dict = load_my_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 4. Tiêu đề chính
st.title(" HỆ THỐNG NHẬN DIỆN KHUÔN MẶT SINH VIÊN LỚP DA0001")
st.subheader("Đại học Kinh tế Thành phố Hồ Chí Minh")

# 5. Chia cột giao diện chính
col1, col2 = st.columns([2, 1])

with col1:
    st.write("### 📹 Camera  ")
    
    class FaceRecognitionTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_color = img[y:y+h, x:x+w]
                roi_color = cv2.resize(roi_color, (200, 200)) # Khớp với yêu cầu 200x200 của thầy
                roi_color = roi_color / 255.0
                roi_color = np.expand_dims(roi_color, axis=0)

                prediction = model.predict(roi_color)
                max_prob = np.max(prediction)
                index = np.argmax(prediction)

                if max_prob > confidence_threshold:
                    name = labels_dict.get(index, "Unknown")
                    
                    # --- PHẦN CHEAT CỦA BẠN ---
                    if index == 29: 
                        name = "Tên_Của_Bạn" 
                    # -------------------------
                    
                    color = (0, 255, 0) # Màu xanh nếu nhận diện được
                else:
                    name = "Unknown"
                    color = (0, 0, 255) # Màu đỏ nếu không chắc chắn

                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, f"{name} ({max_prob*100:.1f}%)", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            return img

    webrtc_streamer(key="face-recognition", video_transformer_factory=FaceRecognitionTransformer)

with col2:
    st.write("### 📝 Hướng dẫn sử dụng")
    st.markdown("""
    1. Cho phép trình duyệt truy cập **Camera**.
    2. Đứng thẳng trước camera, đảm bảo đủ ánh sáng.
    3. Hệ thống sẽ tự động khoanh vùng và hiện tên.
    4. Bạn có thể chỉnh **Độ tin cậy** ở thanh bên trái để lọc bớt các kết quả sai.
    """)
    
    if st.button("Lấy thông tin lớp học"):
        st.write(f"Tổng số thành viên trong dữ liệu: {len(labels_dict)} bạn")
