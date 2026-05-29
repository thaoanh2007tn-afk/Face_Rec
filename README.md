# 📸 DA FACE-REC: Hệ thống Nhận diện Khuôn mặt Real-time (Lớp DA0001)

Ứng dụng web nhận diện khuôn mặt thời gian thực được xây dựng bằng **Python**, **Streamlit** và mô hình học sâu **CNN (Convolutional Neural Networks)**. Dự án được tối ưu hóa để chạy trực tiếp trên trình duyệt thông qua Streamlit Cloud.

## 🚀 Tính năng chính
* **Nhận diện Real-time:** Sử dụng `streamlit-webrtc` để truyền tải dữ liệu video mượt mà qua trình duyệt.
* **Phát hiện khuôn mặt:** Sử dụng thuật toán Haar Cascade để xác định vị trí khuôn mặt nhanh chóng.
* **Độ chính xác cao:** Mô hình CNN được huấn luyện riêng cho tập dữ liệu sinh viên lớp DA0001.
* **Giao diện thân thiện:** Chia cột thông minh giữa khung hình Camera và bảng hướng dẫn sử dụng.

## 🛠 Công nghệ sử dụng
* **Ngôn ngữ:** Python 3.12+
* **Thư viện AI:** TensorFlow / Keras (Mô hình CNN)
* **Xử lý ảnh:** OpenCV
* **Web Framework:** Streamlit
* **Truyền tải Video:** Streamlit-webrtc

## 📂 Cấu trúc thư mục
```text
Face_Rec/
├── app.py                      # File mã nguồn chính chạy ứng dụng
├── face_recognition_model2.h5  # File mô hình CNN đã huấn luyện
├── labels2.json                # File chứa danh sách tên tương ứng với ID
├── requirements.txt            # Danh sách các thư viện cần cài đặt
└── README.md
```
## 📖 Hướng dẫn sử dụng

1. **Truy cập:** Mở đường dẫn ứng dụng đã triển khai trên **Streamlit Cloud**.
2. **Kích hoạt:** Nhấn nút **START** trên giao diện Camera để bắt đầu luồng video.
3. **Cấp quyền:** Cho phép trình duyệt truy cập vào **Webcam** của thiết bị khi có thông báo hiện lên.
4. **Nhận diện:** Đứng trước camera ở khoảng cách từ **0.5m - 1m**. 
   - Hệ thống sẽ tự động phát hiện khuôn mặt.
   - Vẽ khung hình chữ nhật màu xanh quanh mặt.
   - Hiển thị **Tên sinh viên** kèm **Độ tin cậy (Confidence %)** ngay trên khung.

---

## ⚠️ Lưu ý quan trọng

> [!IMPORTANT]
> * **Ánh sáng:** Đảm bảo môi trường đủ sáng, ánh sáng chiếu thẳng vào mặt (không ngược sáng) để model đạt độ chính xác tốt nhất.
> * **Xử lý lỗi:** Nếu khung hình camera bị lỗi hoặc xoay vòng liên tục, hãy kiểm tra lại kết nối mạng hoặc nhấn **F5** để tải lại trang.
> * **Góc độ:** Giữ khuôn mặt trực diện với camera để thuật toán Haar Cascade hoạt động ổn định nhất.

---

## 👤 Thông tin dự án

* **Dự án được thực hiện bởi:** [Trần Ngọc Thảo Anh - 31251026259]
* **Lớp:** DA0001
* **Môn học:** Trí tuệ nhân tạo (AI)
* **Giảng viên hướng dẫn:** [Nguyễn Trường Thịnh]
