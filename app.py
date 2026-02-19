import cv2
import numpy as np
import joblib
import string

# Load trained model
model = joblib.load("sign_model.pkl")

# Create label mapping (remove J and Z)
letters = list(string.ascii_uppercase)
letters.remove('J')
letters.remove('Z')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Removes bg
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    #Find hand and crop
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        hand_img = thresh[y:y+h, x:x+w]

    resized = cv2.resize(gray, (28, 28))

    sample = resized.flatten().reshape(1, -1)

    prediction = model.predict(sample)

    letter = letters[prediction[0]]

    cv2.putText(frame, f"Predicted: {letter}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Sign Detection", frame)
    frame = cv2.flip(frame, 1)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
