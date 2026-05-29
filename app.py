import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import tensorflow as tf
import json

# Load model và labels
model = tf.keras.models.load_model("face_recognition_model2.h5")
with open('labels2.json', 'r') as f:
    labels = json.load(f)
class_names = list(labels.keys())

# Sử dụng bộ lọc Haar Cascade để phát hiện khuôn mặt nhanh
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class FaceRecognitionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Cắt và xử lý vùng khuôn mặt
            roi_color = img[y:y+h, x:x+w]
            roi_color = cv2.resize(roi_color, (160, 160))
            roi_color = roi_color / 255.0
            roi_color = np.expand_dims(roi_color, axis=0)

            # Dự đoán
            prediction = model.predict(roi_color)
            index = np.argmax(prediction)
            confidence = np.max(prediction)
            
            name = "Unknown"
            if confidence > 0.8: # Chỉ hiện tên nếu độ tin cậy trên 80%
                name = class_names[index]

            # Vẽ khung và tên
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{name} ({confidence*100:.1f}%)", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img

st.title("Hệ thống Nhận diện Khuôn mặt Lớp DA0001")
webrtc_streamer(key="example", video_transformer_factory=FaceRecognitionTransformer)
