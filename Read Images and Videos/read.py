# Import the Libraries
import cv2 as cv

# Display of Images on a new Window
img = cv.imread("Photos/kid.jpg")
cv.imshow("Parrot", img)
cv.waitKey(0)


'''
# Display of Videos frame by frame on a new Window
cap = cv.VideoCapture("Videos/test1.mp4")

while True:
    isTrue, frame = cap.read()

    cv.imshow("Video", frame)

    if cv.waitKey(20) & 0xFF == ord("d"):
        break

cap.release()
cv.destroyAllWindows()
'''