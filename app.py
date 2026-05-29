import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import tensorflow as tf
import json

# Cấu hình trang web (icon và tiêu đề trên tab trình duyệt)
st.set_page_config(page_title="AI Face Recognition", layout="wide")

# Tiêu đề ứng dụng
st.title("🤖 Ứng dụng Nhận diện Khuôn mặt Real-time")
st.markdown("---")

# --- LOAD MODEL & LABELS ---
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("face_recognition_model2.h5")
    with open('labels2.json', 'r') as f:
        labels = json.load(f)
    class_names = list(labels.keys())
    return model, class_names

try:
    model, class_names = load_my_model()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    st.error(f"Lỗi tải model hoặc labels: {e}")
    st.stop()

# --- XỬ LÝ AI ---
class FaceRecognitionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Lật ảnh như soi gương
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Cắt vùng mặt và tiền xử lý
            roi_color = img[y:y+h, x:x+w]
            roi_color = cv2.resize(roi_color, (160, 160))
            roi_color = roi_color / 255.0
            roi_color = np.expand_dims(roi_color, axis=0)

            # Dự đoán từ model CNN/Transfer Learning
            prediction = model.predict(roi_color)
            index = np.argmax(prediction)
            confidence = np.max(prediction)
            
            name = "Unknown"
            color = (0, 0, 255) # Đỏ cho người lạ

            if confidence > 0.8: # Độ tin cậy trên 80% 
                name = f"{class_names[index]} ({confidence*100:.1f}%)"
                color = (0, 255, 0) # Xanh lá cho bạn học

            # Vẽ khung và tên
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return img

# --- GIAO DIỆN CHÍNH (CHIA CỘT) ---
col1, col2 = st.columns([2, 1]) # Cột 1 rộng gấp đôi cột 2

with col1:
    st.subheader("📸 Camera AI")
    webrtc_streamer(
        key="face-recognition",
        video_transformer_factory=FaceRecognitionTransformer,
        rtc_configuration={ 
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

with col2:
    st.subheader("📝 Hướng dẫn sử dụng")
    with st.container(border=True): # Tạo khung viền cho thẻ hướng dẫn
        st.info("""
        **Các bước thực hiện:**
        1. Nhấn nút **Start** bên dưới khung camera để bật webcam.
        2. Cấp quyền truy cập camera cho trình duyệt nếu được hỏi.
        3. Đưa khuôn mặt vào giữa khung hình.
        
        **Lưu ý:**
        * Hệ thống nhận diện tốt nhất khi đủ ánh sáng.
        * Nếu hiện 'Unknown', hãy thử điều chỉnh góc mặt.
        * Độ chính xác hiện tại của model: **>90%**[cite: 4].
        """)
        
        st.success("✅ Model: MobileNetV2 (Transfer Learning)")
        st.warning("⚠️ Yêu cầu: File `face_recognition_model2.h5` và `labels2.json` phải nằm cùng thư mục code.")

# Chân trang
st.markdown("---")
st.caption("Sản phẩm được thực hiện cho môn học AI - Nhận diện khuôn mặt lớp học.")
