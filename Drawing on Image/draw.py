import cv2 as cv
import numpy as np

blank  = np.zeros((750,750,3), dtype="uint8")
cv.imshow("Blank", blank)

# Paint the Entire Image with a certain colour
blank[:] = 0,0,255
cv.imshow("Red",blank)

# Paint the Image a certain colour
blank[100:200, 200:300] = 0,0,255
cv.imshow("Red",blank)

# Draw a Rectangle line
cv.rectangle(blank, (0,0), (250,50), (0,255,0), thickness=2)
cv.imshow("Green",blank)

# Draw a Filled Rectangle
cv.rectangle(blank, (0,0), (250,50), (0,255,255), thickness=cv.FILLED)
cv.imshow("Green",blank)

# Draw a Circle line
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2),40, (0,255,255), thickness=2)
cv.imshow("Circle",blank)

# Write a Text
cv.putText(blank, "Hello My Name is Meghal Agarwal !!!", (255,255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,255), thickness=2)
cv.imshow("Text",blank)

cv.waitKey(0)