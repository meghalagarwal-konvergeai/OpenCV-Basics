import cv2 as cv
import numpy as np

img = cv.imread("Photos/kid.jpg")

cv.imshow("Kid", img)

# Transformation (Rotation)
def transf(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    
    return cv.warpAffine(img, transMat, dimensions)

# -x ---> Shift Left
# x ---> Shift Right
# -y ---> Shift Up
# y ---> Shift Down

translated = transf(img, 100, 100)
cv.imshow("Shifted Image", translated)

# Rotation
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, -45)
cv.imshow("Rotated", rotated)

# To rotate a rotated image again
# If the roated image is rotated again then since in the first rotation image is captured as the source image therefore the image rotated from a rotated image does not appear clear.
rotated_twice = rotate(rotated, -90)
cv.imshow("Rotated Twice", rotated_twice)

# Flipping of Image
flip = cv.flip(img, 0) # To flip horiozontally as a Mirror Image "1" is used in place of "0"
cv.imshow("Flipped Image", flip)

cv.waitKey(0)