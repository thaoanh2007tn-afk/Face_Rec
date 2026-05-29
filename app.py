import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
import os
import tensorflow as tf
import json

# 1. Cấu hình trang rộng để có chỗ chứa cột bên phải
st.set_page_config(page_title="Hệ thống Nhận diện", layout="wide")

# --- PHẦN GIỮ NGUYÊN TỪ FILE APP (1) ---
# Tải model và labels
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model('face_recognition_model2.h5')
    with open('labels2.json', 'r') as f:
        labels = json.load(f)
    return model, labels

model, labels = load_model_and_labels()
class_names = list(labels.keys())
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            roi_color = img[y:y+h, x:x+w]
            roi_color = cv2.resize(roi_color, (160, 160))
            roi_color = roi_color / 255.0
            roi_color = np.expand_dims(roi_color, axis=0)

            prediction = model.predict(roi_color)
            index = np.argmax(prediction)
            name = class_names[index]
            confidence = np.max(prediction)

            if confidence > 0.8:
                label = f"{name} ({confidence*100:.1f}%)"
            else:
                label = "Unknown"

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img

# --- PHẦN GIAO DIỆN MỚI (CHIA CỘT) ---
def main():
    st.title("Hệ thống Nhận diện Khuôn mặt Real-time")
    st.write("---")

    # Chia màn hình thành 2 cột: Cột trái (70%) cho Cam, Cột phải (30%) cho Hướng dẫn
    col_left, col_right = st.columns([0.7, 0.3])

    with col_left:
        st.subheader("📸 Camera")
        # Giữ nguyên y hệt cách gọi webrtc_streamer của file app (1)
        webrtc_streamer(
            key="example", 
            video_transformer_factory=VideoTransformer
        )

    with col_right:
        st.subheader("💡 Hướng dẫn")
        # Thẻ hướng dẫn được trang trí bằng markdown
        st.info("""
        **Các bước thực hiện:**
        1. Nhấn nút **START** ở khung camera để bắt đầu.
        2. Cho phép trình duyệt truy cập vào Camera của bạn.
        3. Đảm bảo khuôn mặt nằm trong khung hình và đủ ánh sáng.
        
        **Lưu ý:**
        * Nếu hệ thống hiện **Unknown**, hãy thử điều chỉnh góc mặt.
        * Đảm bảo file `face_recognition_model.h5` và `labels.json` nằm cùng thư mục code.
        """)
        
        st.warning("⚠️ Nếu Camera không hiện, hãy tải lại trang (F5).")

if __name__ == "__main__":
    main()
