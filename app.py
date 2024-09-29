import streamlit as st
import cv2
from PIL import Image
import numpy as np
import easyocr
from ultralytics import YOLO

# Khởi tạo EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Hàm để xử lý hình ảnh và nhận diện biển số
def process_image(image, model):
    # Chuyển đổi ảnh thành mảng numpy
    img_array = np.array(image)

    # Phát hiện biển số xe sử dụng YOLO
    results = model(img_array)

    # Lấy ra bounding box đầu tiên từ kết quả (giả sử rằng chỉ có một biển số xe trong ảnh)
    bboxes = results[0].boxes.xyxy.numpy().astype(int)

    if len(bboxes) == 0:
        st.write("Không tìm thấy biển số xe trong ảnh.")
        return img_array, []
    
    # Duyệt qua các bounding box và nhận diện ký tự
    recognized_texts = []
    for bbox in bboxes:
        top_left = (bbox[0], bbox[1])
        bottom_right = (bbox[2], bbox[3])

        # Cắt vùng biển số
        plate_img = img_array[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        # Nhận diện ký tự trên biển số
        result = reader.readtext(plate_img)
        for detection in result:
            text = detection[1]
            recognized_texts.append(text)

        # Vẽ khung chữ nhật xung quanh biển số và chèn văn bản
        font = cv2.FONT_HERSHEY_COMPLEX
        img_array = cv2.rectangle(img_array, top_left, bottom_right, (255, 0, 0), 2)
        img_array = cv2.putText(img_array, ' '.join(recognized_texts), (top_left[0], top_left[1] - 10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return img_array, recognized_texts

def main():
    # Streamlit giao diện
    st.title("Nhận diện kí tự biển số xe")
    st.write("Tải lên một hình ảnh để nhận diện kí tự:")

    uploaded_file = st.file_uploader("Chọn một file hình ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Đọc hình ảnh bằng PIL
        image = Image.open(uploaded_file)
        st.image(image, caption='Hình ảnh đã tải lên', use_column_width=True)

        # Khởi tạo mô hình YOLO nếu chưa được khởi tạo
        if 'model' not in st.session_state:
            st.session_state.model = YOLO("Train_data_LT/runs/detect/train/weights/best.pt")

        # Xử lý hình ảnh
        processed_image, recognized_texts = process_image(image, st.session_state.model)

        # Hiển thị hình ảnh đã xử lý
        st.image(processed_image, caption='Hình ảnh đã xử lý', use_column_width=True)

        # In ra các ký tự đã nhận dạng trên cùng một hàng
        st.write("Biển số xe nhận diện được là: ")
        st.write(' '.join(recognized_texts))

if __name__ == "__main__":
    main()
