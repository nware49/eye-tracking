# Quick script

# import the package and get the bare minimum working

import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Load the eye detection model
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]
            eye_color = roi_color[ey:ey + eh, ex:ex + ew]

            blurred_eye_gray = cv2.GaussianBlur(eye_gray, (5, 5), 0)

            # Thresholding to isolate the pupil
            # _, thresh = cv2.threshold(eye_gray, 70, 255, cv2.THRESH_BINARY_INV)
            thresh = cv2.adaptiveThreshold(blurred_eye_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

            if contours:
                (cx, cy), radius = cv2.minEnclosingCircle(contours[0])
                cv2.circle(eye_color, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)
                # Assuming cx, cy are the coordinates of the pupil
                #cv2.line(frame, (int(cx), int(cy)), (int(cx) + 50, int(cy)), (0, 255, 0), 2)  # Horizontal line
                #cv2.line(frame, (int(cx), int(cy)), (int(cx), int(cy) + 50), (0, 255, 0), 2)  # Vertical line line

    cv2.imshow('Pupil Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
