import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import tensorflow as tf
import json

st.set_page_config(page_title="Face Recognition AI", layout="wide"

model = tf.keras.models.load_model("face_recognition_model2.h5")
with open('labels2.json', 'r') as f:
    labels = json.load(f)
class_names = list(labels.keys())


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class FaceRecognitionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            
            roi_color = img[y:y+h, x:x+w]
            roi_color = cv2.resize(roi_color, (160, 160))
            roi_color = roi_color / 255.0
            roi_color = np.expand_dims(roi_color, axis=0)

            prediction = model.predict(roi_color)
            index = np.argmax(prediction)
            confidence = np.max(prediction)
            
            name = "Unknown"
            if confidence > 0.8: 
                name = class_names[index]

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{name} ({confidence*100:.1f}%)", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img

st.title("Hệ thống Nhận diện Khuôn mặt Lớp DA0001")
st.write("---")


col_cam, col_guide = st.columns([2, 1])

with col_cam:
    st.subheader("📸 Camera")
    
    webrtc_streamer(key="example", video_transformer_factory=FaceRecognitionTransformer)
with col_guide:
    st.subheader("📖 Hướng dẫn sử dụng")
    with st.container(border=True):
        st.markdown("""
        **Các bước thực hiện:**
        1. Nhấn nút **START** để mở camera.
        2. Cho phép trình duyệt truy cập Camera.
        3. Đưa khuôn mặt vào giữa khung hình.
        
        **Lưu ý:**
        * Khoảng cách tốt nhất là từ 0.5m - 1m.
        * Đảm bảo ánh sáng chiếu thẳng vào mặt.
        * Nếu camera không hiện, hãy nhấn Start lại.
        """)
        
        st.info("💡 Hệ thống đang sử dụng model CNN huấn luyện trên tập dữ liệu lớp DA0001.")
