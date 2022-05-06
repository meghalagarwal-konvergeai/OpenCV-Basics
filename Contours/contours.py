'''
Contours are defined as the line joining all the points along the boundary of an image that are having the same intensity.
Contours come handy in shape analysis,
finding the size of the object of interest,
and object detection.
'''

import cv2 as cv
import numpy as np

# Actual Image
img = cv.imread("Photos/kid.jpg")
#cv.imshow("Kid",img)

# Grey Image
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow("Blue", grey)

# One way of finding Contours is by using "Canny" function
# Blur Image
blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)
#cv.imshow("Blur", blur)

# Edges Image on Actual Image/ Grey Image
# Finding a Contour on an actual image will generate the total number of counters which is hugh and can be slow.
canny = cv.Canny(grey, 125, 175)
cv.imshow("Canny Edges", canny)

# Edges Image on Blurred Actual Image
# Finding a Contour on a blur image will reduce the total number of counters which is better.
blured_canny = cv.Canny(blur, 125, 175)
cv.imshow("Blured Canny Edges", blured_canny)

# -x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-

# Another way of finding a Counter is by using "Threshold" function
# We are setting a Threshold of minimum 125 and maximum 255 and converting an image into Binary
ret, thresh = cv.threshold(grey, 125, 255, cv.THRESH_BINARY)
cv.imshow("Threshold Image", thresh)

# -x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-

# To fetch the number of contours by using "FindContours" method
thresh_contours, thresh_hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

canny_contours, canny_hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

blured_canny_contours, blured_canny_hierarchies = cv.findContours(blured_canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

# Printing the number of contours fetched based on 2 different menthods.
print(f"Total Number of Contours by Canny Method: {len(canny_contours)} contours")
print(f"Total Number of Contours by Blurred Canny Method: {len(blured_canny_contours)} contours")
print(f"Total Number of Contours by Threshold Method: {len(thresh_contours)} contours")

# -x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-

# Drawing the Contours on a Blank Image
# Creating a Blank Image
blank = np.zeros(img.shape, dtype="uint8")

# Drawing Threshold Contour on Blank Image
cv.drawContours(blank, thresh_contours, -1, (0,255,255), 1)
cv.imshow("Creating Contours on Blank Image", blank)

# Drawing Blurred Canny Contour on Blank Image
cv.drawContours(blank, blured_canny_contours, -1, (0,0,255), 1)
cv.imshow("Creating Contours on Blank Image", blank)

# Drawing Canny Contour on Blank Image
cv.drawContours(blank, canny_contours, -1, (255,255,255), 1)
cv.imshow("Creating Contours on Blank Image", blank)

cv.waitKey(0)