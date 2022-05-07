import cv2 as cv

img = cv.imread('/home/meghal/Personal/Konverge_AI/Training/OpenCV Basics/Photos/grp3.jpg')
cv.imshow("Man", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Haarcascading technique
# It not the most used technique for face detection, but in some scenarios it is considered.
haar_cascade = cv.CascadeClassifier('/home/meghal/Personal/Konverge_AI/Training/OpenCV Basics/Face Detection and Recognition/haar_face.xml')

# The more the minNeighbor value the less is the noise that can affect the accuracy of detection.
faces_rectangle = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

print(f"Number of Faces Detected = {len(faces_rectangle)}")

# Drawing Rectanlges where the faces are detected.
for (x,y,w,h) in faces_rectangle:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

cv.imshow("Face Detections", img)

cv.waitKey(0)