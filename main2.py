from ultralytics import YOLO
import cv2

# Wczytaj model YOLOv8
model = YOLO('yolov8n.pt')  # Pobierz wersję Nano dla szybkiego działania

# Ścieżka do wideo
video_path = "path/to/your/video.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Wykonaj detekcję
    results = model(frame)

    # Narysuj detekcje na klatce
    for result in results:
        annotated_frame = result.plot()

    # Wyświetl wynik
    cv2.imshow("YOLOv8", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Wyjście klawiszem ESC
        break

cap.release()
cv2.destroyAllWindows()
