
# import cv2

# # Load built-in face detector from OpenCV (Haar Cascade)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Start webcam
# cap = cv2.VideoCapture(0)

# print("Press 'q' to quit")

# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # face detector works on grayscale

#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     # Draw rectangles around detected faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, "Human Face", (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     # Show the frame
#     cv2.imshow("Face Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
from deepface import DeepFace

# Choose the backend: 'opencv', 'retinaface', 'mtcnn', 'ssd', etc.
detector_backend = 'mtcnn'  # 'opencv' is fast and avoids RetinaFace error

# Open webcam
cap = cv2.VideoCapture(0)
print("Webcam started. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    if not ret:
        print("Unable to access webcam.")
        break

    try:
        # Detect faces in the current frame
        results = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=detector_backend,
            enforce_detection=False
        )

        # Draw rectangles on detected faces
        for face in results:
            fa = face['facial_area']  # safely access facial_area as dictionary
            x = fa['x']
            y = fa['y']
            w = fa['w']
            h = fa['h']

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Human Face", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    except Exception as e:
        print("Detection error:", str(e))

    # Show the frame
    # frame=cv2.flip(frame,1)
    cv2.imshow("DeepFace Webcam - Face Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting webcam...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
