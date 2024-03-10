import cv2
import random
import time

# Open the STREAM
_capture = cv2.VideoCapture(0)

# Check if the STREAM is opened successfully
if not _capture.isOpened():
    print("Failed to open RTSP stream")
    exit()

last_save_time = time.time()

while True:
    # Read a frame from the STREAM
    _ret, _frame = _capture.read()

    # Check if the frame is read correctly
    if not _ret:
        print("Failed to read frame")
        break

    _grayFrame = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)

    _faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    _faces = _faceCascade.detectMultiScale(_grayFrame, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in _faces:
        cv2.rectangle(_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        current_time = time.time()
        if current_time - last_save_time >= 10:
            face = _frame[y:y+h, x:x+w]
            
            cv2.imwrite("output/Face_" + str(random.randint(15, 2500)) + ".jpg", _grayFrame)
            last_save_time = current_time

    # Display the frame
    cv2.imshow("Camera Stream", _frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the STREAM and close the window
_capture.release()
cv2.destroyAllWindows()