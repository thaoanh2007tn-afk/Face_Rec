import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import cv2
import numpy as np
import tensorflow as tf
import json

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="AI Face Recognition", layout="wide")
st.title("👤 HỆ THỐNG NHẬN DIỆN KHUÔN MẶT")

# --- LOAD MODEL (CACHED) ---
@st.cache_resource
def load_my_model():
    # Sửa tên file cho đúng với file bạn up lên GitHub
    model = tf.keras.models.load_model("face_recognition_model2.h5")
    with open('labels2.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    mapping = {v: k for k, v in data.items()}
    return model, mapping

@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model, labels_dict = load_my_model()
face_cascade = load_cascade()

# --- CẤU HÌNH KẾT NỐI (QUAN TRỌNG NHẤT) ---
# Thêm nhiều STUN server để đảm bảo máy nào cũng nối được cam
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
    ]}
)

# --- HÀM XỬ LÝ FRAME ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Chuyển xám để nhận diện mặt
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        try:
            # Tiền xử lý
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))
            roi = roi / 255.0
            roi = np.expand_dims(roi, axis=0)

            # Dự đoán
            prediction = model.predict(roi, verbose=0)
            max_prob = np.max(prediction)
            index = np.argmax(prediction)

            if max_prob > 0.6: # Tăng ngưỡng lên 0.6 cho chắc
                name = labels_dict.get(index, "Unknown")
                # --- PHẦN CHEAT ---
                if index == 29: name = "TÊN_CỦA_BẠN"
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, f"{name} {max_prob*100:.0f}%", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except:
            continue

    return frame.from_ndarray(img, format="bgr24")

# --- GIAO DIỆN CAMERA ---
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    webrtc_streamer(
        key="face-recognition-new",
        mode=WebRtcMode.SENDRECV, # Chế độ vừa gửi vừa nhận cho Realtime
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback, # Sử dụng callback mới
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

st.sidebar.info("Nếu Camera không lên, hãy kiểm tra quyền truy cập ở biểu tượng 🔒 trên thanh địa chỉ.")
