import cv2
import mediapipe as mp
import time

# Video yakalama
cap = cv2.VideoCapture(0)

# Mediapipe el modülünü yükleme
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    # Kameradan görüntü al
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    # Eğer eller algılandıysa
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Her bir elin noktalarını ve bağlantılarını çiz
            for id, lm in enumerate(handLms.landmark):
                # Her bir noktayı işle
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Her bir noktayı çember içine al
                cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

            # Ellerin bağlantılarını çiz
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
    # Çıkış için 'q' tuşuna basılıp basılmadığını kontrol et
    if cv2.waitKey(1) == ord("q"):
        break

    # Görüntüyü göster
    cv2.imshow("Image", img)
    cv2.waitKey(1)

# Kamerayı kapat
cap.release()
cv2.destroyAllWindows()