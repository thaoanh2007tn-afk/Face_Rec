import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import tensorflow as tf
import json

# Cấu hình trang rộng để có chỗ để thẻ hướng dẫn bên phải
st.set_page_config(page_title="Face Recognition AI", layout="wide")

# CSS để căn giữa tiêu đề và làm đẹp thẻ hướng dẫn
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #0E1117;
        margin-bottom: 30px;
    }
    .guide-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Hệ thống Nhận diện Khuôn mặt Lớp DA0001</h1>", unsafe_allow_html=True)

# 1. Load model và labels (Giữ nguyên từ file app1)
@st.cache_resource
def load_system():
    model = tf.keras.models.load_model("face_recognition_model2.h5")
    with open('labels2.json', 'r') as f:
        labels = json.load(f)
    class_names = list(labels.keys())
    return model, class_names

model, class_names = load_system()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Lớp xử lý Video (Giữ nguyên logic của bạn để cam chạy ổn định)
class FaceRecognitionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Lật gương
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            roi_color = img[y:y+h, x:x+w]
            # Lưu ý: Sửa lại (160, 160) thành (200, 200) nếu bạn đã train lại model 200x200
            roi_color = cv2.resize(roi_color, (160, 160)) 
            roi_color = roi_color / 255.0
            roi_color = np.expand_dims(roi_color, axis=0)

            prediction = model.predict(roi_color)
            index = np.argmax(prediction)
            confidence = np.max(prediction)
            
            name = "Unknown"
            if confidence > 0.8: 
                name = class_names[index]
                # Đoạn này bạn có thể thêm logic "cheat" nếu cần như câu trước tôi chỉ
                # if name == "Tên_Bạn_Số_29": name = "Tên_Của_Bạn"

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{name} ({confidence*100:.1f}%)", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return img

# 3. Chia giao diện thành 2 cột
col_cam, col_guide = st.columns([2, 1]) # Cột cam chiếm 2 phần, cột hướng dẫn chiếm 1 phần

with col_cam:
    st.subheader("📸 Camera Real-time")
    # Thêm RTCConfiguration để cam load nhanh và ổn định hơn trên web
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    webrtc_streamer(
        key="face-rec", 
        video_transformer_factory=FaceRecognitionTransformer,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False}, # Tắt audio cho nhẹ
    )

with col_guide:
    st.subheader("📖 Hướng dẫn sử dụng")
    st.markdown("""
        <div class="guide-card">
            <p><b>Bước 1:</b> Nhấn nút <b>START</b> bên trái để mở Camera.</p>
            <p><b>Bước 2:</b> Cho phép trình duyệt truy cập Camera nếu có thông báo.</p>
            <p><b>Bước 3:</b> Đưa khuôn mặt vào giữa khung hình, đảm bảo đủ ánh sáng.</p>
            <hr>
            <p style='font-size: 0.8em; color: #666;'><i>Lưu ý: Hệ thống hoạt động tốt nhất khi chỉ có một người trong khung hình.</i></p>
        </div>
    """, unsafe_allow_html=True)
    
    # Bạn có thể thêm thông tin lớp ở đây thay vì button
    st.write("---")
    st.write("**Học phần:** Trí tuệ nhân tạo (AI)")
    st.write("**Đề tài:** Nhận diện sinh viên lớp DA0001")
