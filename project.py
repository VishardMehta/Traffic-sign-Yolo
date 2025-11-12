from ultralytics import YOLO
import cv2
import time
import numpy as np

# Load your trained YOLO model
model = YOLO("bestyolov11n.pt")  # ensure the file path is correct

# Start webcam
cap = cv2.VideoCapture(0)

# Set webcam resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Start time for FPS calculation
    start_time = time.time()

    # Predict
    results = model(frame, verbose=False)

    # Annotate the frame
    annotated_frame = results[0].plot()

    # Calculate FPS
    fps = 1 / (time.time() - start_time + 1e-6)

    # Compute average confidence of detections (accuracy display)
    confidences = [box.conf[0].item() for box in results[0].boxes] if len(results[0].boxes) > 0 else []
    avg_conf = np.mean(confidences) * 100 if confidences else 0

    # Draw FPS bar
    cv2.rectangle(annotated_frame, (10, 10), (250, 70), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Draw accuracy bar
    cv2.putText(annotated_frame, f"Avg Conf: {avg_conf:.1f}%", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show output
    cv2.imshow("YOLO Webcam Detection", annotated_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
