import cv2
import numpy as np

# 1. Kết nối với Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở Webcam!")
    exit()

print("HƯỚNG DẪN:")
print("- Nhấn 'q' để THOÁT chương trình")
print("- FPS sẽ thấp do GrabCut tính toán rất nặng trên từng khung hình.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Lật ngược ảnh theo chiều ngang để giống soi gương
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # 2. Định nghĩa khung chữ nhật cố định ở giữa màn hình (chiếm 60% chiều rộng, 70% chiều cao)
    rect_w, rect_h = int(w * 0.6), int(h * 0.7)
    rect_x = (w - rect_w) // 2
    rect_y = (h - rect_h) // 2
    rect = (rect_x, rect_y, rect_w, rect_h)

    # 3. Khởi tạo cấu trúc dữ liệu cho GrabCut (Phải tạo mới hoặc reset lại theo từng frame)
    mask = np.zeros(frame.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 4. Chạy GrabCut trực tiếp trên khung hình hiện tại
    # iterCount=2 để cứu vãn FPS không bị tụt quá sâu
    cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, iterCount=2, mode=cv2.GC_INIT_WITH_RECT)

    # 5. Hậu xử lý mặt nạ để lấy vùng tiền cảnh (giá trị 1 và 3)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_obj = frame * mask2[:, :, np.newaxis]

    bg = cv2.GaussianBlur(frame, (15, 15), 0.0)

    cv2.copyTo(frame, segmented_obj, bg)

    # 6. Vẽ khung chữ nhật hướng dẫn lên kết quả cuối cùng để người dùng biết phạm vi quét

    # 7. Hiển thị kết quả liên tục
    cv2.imshow("Mask", segmented_obj)
    cv2.imshow("GrabCut Real-time", bg)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
