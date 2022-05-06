import cv2 as cv

# Creating a function for re-sizing all types of images and videos.
def rescaled_Frame(frame, scale=0.75):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Creating a function for re-sizing only for Live Videos
def change_Resolution(width, height):
    cap.set(3, width)
    cap.set(4, height)

# Display of Videos frame by frame on a new Window
cap = cv.VideoCapture("Videos/test1.mp4")

while True:
    isTrue, frame = cap.read()

    frame_resized = rescaled_Frame(frame, 0.2)
    
    #cv.imshow("Video", frame)
    cv.imshow("Video", frame_resized)

    if cv.waitKey(20) & 0xFF == ord("d"):
        break

cap.release()
cv.destroyAllWindows()