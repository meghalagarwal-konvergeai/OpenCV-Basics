import cv2 as cv

img = cv.imread("Photos/raccoon.jpg")
cv.imshow("Raccoon Image", img)

'''
The HSV or Hue, Saturation, and value of a given object is the color space associated with the object in OpenCV.
The Hue in HSV represents the color,
Saturation in HSV represents the greyness, and
Value in HSV represents the brightness.
'''
hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow("HSV Image", hsv_image)

'''
LAB color space is often referred to as L multiplies with the product of A & B ( L*A*B ).
L stands for Luminance dimensions( Intensity )
which a & b are color component dimensions
where 
"a" represents colors from green to magenta,
"b" represents colors from blue to yellow.
'''
lab_img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow("LAB Image", lab_img)

'''
RGB color space is the opposite of BGR color space,
where the objects in the BGR image that are in Blue appear to be Red,
and objects in Red color appear to be in Blue and Green Color remain the same.
'''
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow("RGB Image", rgb_img)

cv.waitKey(0)