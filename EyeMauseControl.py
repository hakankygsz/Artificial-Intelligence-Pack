import cv2
import mediapipe as mp
import pyautogui

# Open the camera connection
cam = cv2.VideoCapture(0)

# Use the MediaPipe library to detect face features
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get the screen width and height
screen_w, screen_h = pyautogui.size()

while True:
    # Capture a frame from the camera
    _, frame = cam.read()

    # Flip the frame horizontally (like a mirror)
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face features with MediaPipe
    output = face_mesh.process(rgb_frame)

    # Get the points where face features are found
    landmark_points = output.multi_face_landmarks

    # Get the dimensions of the frame
    frame_h, frame_w, _ = frame.shape

    # If there are face points
    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Process the points where the pupils are located
        for id, landmark in enumerate(landmarks[474:478]):  
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))   
            if id == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)

        # Process the left eye area
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        # Check if the left eye area is closed
        if (left[0].y - left[1].y) < 0.001: 
            pyautogui.click()
            pyautogui.sleep(1)
            
    if cv2.waitKey(1) == ord("q"): 
        break

    # Show the window
    cv2.imshow('Eye Controlled Mouse', frame)

    # Close the window when a key is pressed
    cv2.waitKey(1)

# Release the camera
cam.release()
cv2.destroyAllWindows()