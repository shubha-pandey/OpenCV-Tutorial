# pip install opencv-contrib-python                                  includes everything from main module as well as contribution modules provided by community

# pip install caer                                                   helps speed up the workflow                     It is basically a set of utility functions that are super useful in computer vision

import cv2


#  ---  READING IMAGES AND VIDEOS  ---


# IMAGE
img = cv2.imread('IP_Assets/SOC.jpeg')                             # read the image
#cv2.imshow('Image', img)                                           # display the image        

#cv2.waitKey(0)                                                     # to keep the window open otherwise the window containing the image will automatically close as soon as it opens;                          0 --> infinite                    1000 --> 1 sec

'''
# VIDEO
vid = cv2.VideoCapture('IP_Assets/cat.mp4')

while True :
    isTrue, frame = vid.read()
    cv2.imshow('Video', frame)

    if cv2.waitKey(20) & 0xFF == ord('x') :                       # press 'x' to quit
        break

vid.release()
cv2.destroyAllWindows()
'''
# if using a webcam
# cap_vid = cv2.VideoCapture(0)                                     # 0 for device 

'''
Might encouter this error while trying to read a video file -
" error: (-215:Assertion failed) size.width>0 && size.height>0 in function 'cv::imshow' "
The video stops suddenly and there's '-215 assertion failed error'. This means in almost all cases, the OpenCV couldn't find a media file at the specified location. 
It's because the video ran out of frames, OpenCV couldn't find anymore frames after the last frame in the video; so it unexpectedly broke out of the while loo by itself by raising a cv2 error.
Also the same error is shown in case the path to the video is incorrect.
'''



#  --- RESIZING AND RESCALING  ---

# Usually video files and images are resized and or rescaled to prevent computational strain. 
# Large media files tend to store a lot of info in it and displaying it takes up alot of processing needs. By resizing and rescaling we're actually getting rid of some of that information. 
# Rescaling video implies modifying its height & width to a particular height & width. 
# Generally, it's always best practice to downscale or change the width and height of the video files to a smaller value than the original dimeansions. The reason for this is because most cameras, webcam included, don't support going higher than its max capability.


#  _ RESCALE _

def rescaleFrame(frame, scale=0.3) :                                                         # works for images, videos, and live videos                     
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation = cv2.INTER_AREA)


# IMAGE
img_resize = rescaleFrame(img)
cv2.imshow('Resized Image', img_resize) 

# if executed after 'VideoCapture' then it does not work

'''
# VIDEO                                                         
vid = cv2.VideoCapture('IP_Assets/cat.mp4')

while True :
    isTrue, frame = vid.read()
    frame_resize = rescaleFrame(frame, scale = 3)                # can nullify the scale from the function by mentioning a scale value here.
    #cv2.imshow('Resized Video', frame_resize)

    if cv2.waitKey(20) & 0xFF == ord('x') :                       # press 'x' to quit
        break

vid.release()
cv2.destroyAllWindows()
'''
# only one of the video reading method working at a time. 
# The reason only the first video display method works is that OpenCV cannot handle two cv2.VideoCapture() instances for the same video file simultaneously in a single program execution. OpenCV holds onto resources until explicitly released with vid.release().
#  vid.release() and cv2.destroyAllWindows() is used at the end of each section to free up resources, allowing the next section to start with a clean slate.
# But here even that doesn't help. I guess the reason might be that OpenCV might still be holding onto some internal resource connected to the file.


# Anotherway to rescale & resize, only specifically for videos

def resize(width, height) :                                       # works only for live videos
    vid.set(3, width)
    vid.set(4, height)



#  ---  BASIC FUNCTIONS  ---


#  CONVERTING TO GRAYSCALE  

gray_img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
cv2.imshow('GrayScale', gray_img)


#  BLUR                                 

blur_img = cv2.GaussianBlur(img_resize, (5, 5), cv2.BORDER_DEFAULT)                             # cv2.GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None, /)
#cv2.imshow('Blur', blur_img)


#  EDGE CASCADE

ec_img = cv2.Canny(img_resize, 125, 175)                       # cv2.Canny(img, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None, /)
#cv2.imshow('Edge Cascade', ec_img)

smooth_ec_img = cv2.Canny(blur_img, 125, 175)                  # reduce the amount of edges by or get rid of some of the edges
#cv2.imshow('Edge Cascade', smooth_ec_img)


#  DILATING

dilate_img = cv2.dilate(smooth_ec_img, (3,3), iterations=1)   # cv2.dilate(src, kernel, dst=None, anchor = None, borderType = None, borederValue = None, /)
#0cv2.imshow('Dilated', dilate_img)


#  ERODING

erode_img = cv2.erode(dilate_img, (3, 3), iterations=1)           #cv2.erode(src, kernel, dst=None, anchoe=None, iterations=None, )
#cv2.imshow('Erode', erode_img)


#  RESIZE

resize_img = cv2.resize(img, (300, 300), interpolation= cv2.INTER_AREA)
#cv2.imshow('Resize', resize_img)


#  CROP

crop_img = img[50:200, 200:400]
#cv2.imshow('Crop', crop_img)



#  ---  DRAWING SHAPES  ---

''' There are two ways to to draw on images, by actually drawing on standalone images or by creating a dummy image or a blank image to work with '''


import numpy as np                                                 # pip install numpy

#creating a blank image
blank_img = np.zeros((400, 400, 3), dtype='uint8')                    # 'uint8' is basically thr datatype of an image 
#jcv2.imshow('Blank', blank_img)

# PAINT THE IMAGE A CERTAIN COLOR 

#blank_img[:] = 0, 255, 0                                           # colors te entire image green 
#cv2.imshow('Green', blank_img)

blank_img[200:300, 300:400] = 0, 0, 255                             # calling a certain portion of image by giving it a range of pixels 
#cv2.imshow('Green', blank_img)


# LINE

cv2.line(blank_img, (0,0), (blank_img.shape[1]//2, blank_img.shape[0]//2), (255, 255, 255), thickness=2)
#cv2.imshow('Line', blank_img)

cv2.line(blank_img, (100,250), (300, 350), (255, 0, 0), thickness=3) 
#cv2.imshow('Line', blank_img)

# RECTANGLE                                                    cv2.rectangle(img, pt1, pt2. color, thickness, lineType, shift, /)

cv2.rectangle(blank_img, (50,50), (300, 200), (255, 0, 0), thickness=2)
#cv2.imshow('Rectangle', blank_img)

cv2.rectangle(blank_img, (50,50), (300, 200), (255, 0, 0), thickness=cv2.FILLED)                    # cv2.FILLED  or  -1   same action
#cv2.imshow('Rectangle', blank_img)

cv2.rectangle(blank_img, (50,50), (blank_img.shape[1]//2, blank_img.shape[0]//2), (255, 0, 0), thickness=4)
#cv2.imshow('Rectangle', blank_img)


# CIRCLE

cv2.circle(blank_img, (blank_img.shape[1]//2, blank_img.shape[0]//2), 40, (255, 255, 255), thickness = 3)
#cv2.imshow('Circle', blank_img)



#   --- WRITING TEXT  ---


# cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin, /)

cv2.putText(blank_img, ('Writing Text on Images'), (10, 155), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2 )
#cv2.imshow('Text', blank_img)



#  ---  IMAGE TRANSFORMATIONS  ---


# TRANSLATION

def Translate(img, x, y) :                                              # -x  --> Left ,     -y  --> Up ,      x  --> Right ,      y  --> Down
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv2.warpAffine(img, transMat, dimensions)                    # cv2.warpAffine(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None, /)

translate_img = Translate(img_resize, 100, 100)
#cv2.imshow('Translated', translate_img)


# ROTATION

def Rotate(img, angle, rotPoint=None) :
    (height, width) = img.shape[:2]
    if rotPoint is None :
        rotPoint = (width//2, height//2)

    rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)              # cv2.getRotation(center, angle, scale, /)
    dimensions = (width, height)

    return cv2.warpAffine(img, rotMat, dimensions)

rotate_img = Rotate(img_resize, 45)
#cv2.imshow('Rotated', rotate_img)


# RESIZE 

resizing_img = cv2.resize(img, (400, 400), interpolation = cv2.INTER_CUBIC)
#cv2.imshow('Resized', resizing_img)


# FLIP

flip_img = cv2.flip(img_resize, -1)      # cv2.flip(src, flipCode, dst=None, /)            0 --> vertical flip, over x axis          1 --> horizontal flip, over y axis            -1 --> both vertical & horizontal flip 
#cv2.imshow('Flip', flip_img)



#   --- CONTOUR DETECTION  ---


# GRAYSCALE      
# same as done above

# CANNY

canny_img = cv2.Canny(img_resize, 125, 175)
#cv2.imshow('Canny Edges', canny_img)


# FINDING CONTOURS OF AN IMAGE

# This method basically returns two things, contours and higher keys. 
# It essentially looks at the structuring element or the edges found in the image and returns true values. 
# The CONTOURS which is essentially a Python list of all the coordinates of the contours that were found in the image.
# The HIERARCHIES essentially refers to the hierarchical representaion of contours.
# The RETR_LIST essentially is a mod which find the 'findContours' method returns and finds the contours. It essentially returns all the quantities that're found in the image. 
# Other opt   RETR_EXTERNAL --> only external contours,     RETR_TREE  -->  all the hierarchical contours 
# The Contour Approximation Method is basically how we want to approximate the contour.
# CHAIN_APPROX_NONE does nothing, it just returns all of the contours.
# CHAIN_APPROX_SIMPLE --> compresses all the quantities that are returned.  

contours, hierarchies = cv2.findContours(canny_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)                         
#print(f'{len(contours)} contour(s) found !')

# Another way to find contours, instead of using Canny edge detecto, using threshold function
# THRESHOLD essentially looks at an image and tries to binarize that image. Gake an image and convert it into binary form that is either zero or black, or white, or 255.
# If a particular pixel is below 125, if the density of that pixel is below 125, it is going to be set to zero or blank. If it is above 125, it is set to white or 255.

ret, thresh = cv2.threshold(gray_img, 125, 255, cv2.THRESH_BINARY)                  # cv2.threshold(src, thresh, maxval, type, dst = None, /)
#cv2.imshow('Thresh', thresh)

contours, hierarchies = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  
#print(f'{len(contours)} contour(s) found !')


# In OpenCV we can actually visualize the contours found on the image by essentially drawing over the image.

blank = np.zeros(img_resize.shape[:2], dtype='uint8')                               # to visualize
#cv2.imshow('Blank', blank)                                                          # same dimensions as the img_resize

# draw the contours on the blank image to find out what kind of contours the OpenCV found'

cv2.drawContours(blank, contours, -1, (0, 0, 255), 1)         # cv2.drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxlevel=None, offset=None, /)
#cv2.imshow('Contours Drawn', blank)                           # found all the edges of the image and attempted to draw them out on the blank image
# remove [:2] from blkank to run this one


#  --- COLOR SPACES  ---

# Switching bwtween color spaces.
# a COLOR SPACEs are basically a space of colors, a system of representing an array of pixel colors


# BGR to GRAYSCALE
# same as previously done


# BGR to HSV

hsv_img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2HSV)
#cv2.imshow('HSV Image', hsv_img)


# BGR to L*a*b

lab_img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2LAB)                              # more tuned to how how humans percieve color
#cv2.imshow('LAB Image', lab_img)                                                 


''' 
The OpenCV reads an image in BGR format, but that is not the current system that we used to represent.
And that's not the current sys that is used to represented outside OpenCV. Outside of OpenCV , the RGB format is used, which is kind of like the inverse of BGR.
'''

import matplotlib.pyplot as plt

plt.imshow(img_resize)
#plt.show()

# BGR to RGB

rgb_img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
#cv2.imshow('RGB', rgb_img)

plt.imshow(rgb_img)                                                # The matplotib's default is RGB 
#plt.show()


# The inverse of all the above conversion is true. 
# But GRAY to HSV conversion is not possible.  GRAY --> BGR --> HSV



#  --- COLOR CHANNELS  ---

# Splitting and Merging color channels

# A color image basically consists of multiple channels, red, green, and blue.
# All the images around, BGR or RGB, are basically these three channels merged together.
# OpenCV allows us to split an image into its respective color channels.

b, g, r = cv2.split(img_resize)

#cv2.imshow('Blue', b)
#cv2.imshow('Green', g)
#cv2.imshow('Red', r)

#print(img_resize.shape)
#print(b.shape)
#print(g.shape)
#print(r.shape)

# When printing the individual color shapes the tuple does not show any channels because the shape of that component is one(not mentioned). That's why the image is displayed as a grayscale image, because they have a shape of one.

# Merging the color channels together
merge_img = cv2.merge([b, g, r])
#cv2.imshow('Merge', merge_img)


# Reconstructing The Image
# Way of looking the actual color there is in a particular channel. 

blue_img = cv2.merge([b, blank, blank])
#cv2.imshow('Blue Color', blue_img)

green_img = cv2.merge([blank, g, blank])
#cv2.imshow('Green Color', green_img)

red_img = cv2.merge([blank, blank, r])
#cv2.imshow('Red Color', red_img)

# Add [:2] in the blank image to run this 



#  ---  BLURRING TECHNIQUES  ---

# We generally smooth an image when it tends to have a lot of noise.
# Noise that is caused from camera sensors are basically problems in lighting when the image was taken.
# We can essentially smooth out the images or reduce some of the noise by applying some blurring method.

# What actually goes on when we try to apply blur ?
# WINDOW is essentially drawn over a specific portion of an image. This Window has a size called 'Kernel Size'.
# KERNEL size is basically the no of rows and columns.
# Essentially the Blur is applied to the middle pixel as a result of the pixels around it, also called the surrounding pixels.


#  AVERAGING

# Define a kernel window over a specific portion of an image.
# This window will essentially compute the pixel intensity of the middle pixel of the true center as the average of the surrounding pixel intensities.
# This process happens throughout the image.

avg_blur_img = cv2.blur(img_resize, (3, 3))                     # the higher the kernel size is specified the more blur there is
#cv2.imshow('Average Blur', avg_blur_img)


#  GAUSSIAN BLUR

# It basically does the same thing as averaging, except that instead of computing the average of all of this running pixel intensity, each running pixel is given a particular weight.
# And the average of the products of these weights gives the value for the true center.
# Using this method, we tend to get less blurring than compared to the average method, because a certain weight value is added when computing the blur.
# The Gaussian Blur is more natural as compared to averaging.

gauss_blur_img = cv2.GaussianBlur(img_resize, (7,7), 0)
#cv2.imshow('Gauss Blur', gauss_blur_img)


#  MEDIAN BLUR

# It's basically the same thing as averaging, except that instead of finding the average of the surrounding pixels, it finds the median of the surrounding pixels.
# Generally medium blurring tends to be more effective in reducing noise as compared to averaging and even gaussian blur.
# It is pretty good at removing some salt and pepper noise that may exist in the image.
# GEnerslly it is not for high kernel sizes such as 7 or even 5 in some cases and it is more effective in reducing some of the noise in the image.

medium_blur_img = cv2.medianBlur(img_resize, 3)        # The kernel size will not be a tuple of 3x3, but instead, jiust an integer 3, beacause OpenCV automatically assumes that this kernel size will be 3x3 just based off this integer.
#cv2.imshow('Medium Blur', medium_blur_img) 


# Traditional Blurring methods blur the image without looking whether the edges are reduced or not.


#  BILATERAL BLURRING 

# It is the most effective and used in a lot of advanced computer vision projects, essentially because of how it blurs.
# Bilateral blurring applies blurring but retains the edges in the image.
# We get the blurred image but we also get to retain the edges as well.

bilateral_blur_img = cv2.bilateralFilter(img_resize, 10, 25, 35)         # bilateralFilter(src, d, sigmaColor, sigmaSpace, dst = None, borderType = None, /)                d --> diameter
#cv2.imshow('Bilateral Blur', bilateral_blur_img)



#  ---  BITWISE OPERATIONS  ---

# Bitwise Operators operate in a binary manner.
# A pixel is turned off if it has a vlalue of zero(0), and is turned on if it has a value of one(1).

# XOR - OR = AND
# OR - AND = XOR

blank_image = np.zeros((400, 400), dtype='uint8')

rect = cv2.rectangle(blank_image.copy(), (30, 30), (370, 370), 255, -1)
circle = cv2.circle(blank_image.copy(), (200, 200), 200, 255, -1)

#cv2.imshow('Rectangle', rect)
#cv2.imshow('Circle', circle)


#  bitwise AND 

bit_and = cv2.bitwise_and(rect, circle)     # bitwise_and(src1, src2, dst=None, mask=None, /)
##cv2.imshow('AND', bit_and)                  # take two images and return their intersecting region


#  bitwise OR

bit_or = cv2.bitwise_or(rect, circle)        # returns both the intersecting as well as non-intersecting regions    
#cv2.imshow('OR', bit_or)                     


#  bitwise XOR

bit_xor = cv2.bitwise_xor(rect, circle)      # returns the non intersecting regions
#cv2.imshow('XOR', bit_xor)


#  bitwise NOT

bit_not = cv2.bitwise_not(rect)      # it does not returns anything. what it does is it inverts the binary color
#cv2.imshow('NOT', bit_not)



#  --- MASKING  ---

# Masking is essentially performed with the help of bitwise operators.
# MAsking allows us to focus on certain parts of an image that we would like to focus on.

# the dimensions of the mask have to be the same size as that of the image. (keep in mind when creating blank image)

mask = cv2.circle(blank, (img_resize.shape[1]//2, img_resize.shape[0]//2), 100, 255, -1)

masked_img = cv2.bitwise_and(img_resize, img_resize, mask=blank)
#cv2.imshow('Mask', masked_img)

# can obtain different shapes by performing bitwise operations on two shapes and then use it to mask

circle1 = cv2.circle(blank.copy(), (img_resize.shape[1]//2 + 45, img_resize.shape[0]//2), 100, 255, -1)
rect1 = cv2.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)

new_shape = cv2.bitwise_and(circle1, rect1)

masked_image = cv2.bitwise_and(img_resize, img_resize, mask=new_shape)
#cv2.imshow('Mask', masked_image)



#  --- COMPUTING HISTOGRAMS  ---

# HISTOGRAM is basically allow us to visualize the distribution of pixel intensities in an image.
# It is kinda like a graph or a plot that will give a high level initution of the pixel distribution in the image.


#  GRAYSCALE HISTOGRAM

# calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None,/)
# The image is a list so pass it in a list of images.
# The no of channels basically specify the index of the channel we want to compute a histogram for. 
# histSize is the no of bins that we want to use for computing the histogram. The no of bins across the x axis represents the intervals of pixel intensities.
# range is the range of all possible pixel values.


gray_hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

#plt.figure()
#plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
#plt.plot(gray_hist)
#plt.xlim([0, 256])
#plt.show()


# We can create a mask and then computemthe histogram only on that particular mask.
masking = cv2.bitwise_and(gray_img, gray_img, mask=mask)

gray_hist_mask = cv2.calcHist([gray_img], [0], masking, [256], [0, 256])

#plt.plot(gray_hist_mask)
#plt.show()


#  COLOR HISTOGRAM

colors = ('b', 'g', 'r')
for i, col in enumerate(colors) :
    col_hist = cv2.calcHist([img_resize], [i], None, [256], [0, 256])
    #plt.plot(col_hist, color=col)
    plt.xlim([0, 256])

plt.title('Color Histogram')
#plt.show()



#  ---  THRESHOLDING  ----

# THRESHOLDING is a binarization of an image. 
# In general, we want to take an image and convert it to a binary image that is an image where pixels are either 0, or black, or 255, or white.


#  SIMPLE THRESHOLDING 

threshold, thresh1 = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)             # it returns   thresh1 --> thresholding image or the binarized image       threshold -->  the same value that is passes (150)
#cv2.imshow('Simple Threshold', thresh1)

# creating inverse threshold

threshold, thresh1_inv = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY_INV)             # it returns   thresh1 --> thresholding image or the binarized image       threshold -->  the same value that is passes (150)
#cv2.imshow('Simple Threshold Inverse', thresh1_inv)


#  ADAPTIVE THRESHOLDING

# One downside to ths is we have to manually specify the threshold values. 
# What can be donee is to let the computer find the optimal threshold value by itself. And useing that value that refines it binarizes over the image.

# adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C, dst=None, /)
# There is no threshold limit value.
# adaptiveMethod basically tells machine which method to use when computing the optimal threshold value.
# blockSize is the neighborhood size of the kernel size which OpenCV needs to use to compute mean to find optimal threshold value.
# C value is an integer that is subtracted from the mean, allowing uss to essentially fine tune the threshold.

adapt_thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)        
#cv2.imshow('Adaptive Threshold', adapt_thresh)

# Using Gaussian instead of mean, the only difference that Gaussian applied was adding a weight to each pixel value and computing the mean across those pixels. This image is better.



#  ---  EDGE DETECTION  ---

# Gradients can be considered as edge like regions that are present in an image.
# But they are not the same thing. Gradients & Edges are completely different from mathematical point of view.


# CANNY EDGE DETECTOR

# as previously done, an advanced edge detection algorithm.
# A multi-step process.


#  LAPLACION

# It computes the gradients of the image.
# When transitioning from black to ehite and white to black, it's considered a positive and a negative slope.
# Now, images itself can not have negative pixel values. So we compute the absolute value of that image. 
# So, all the pixel values of the image are converted to the absolute values. And then we convert that into uint8, an image specific datatype

# Laplacion(src, ddepth, dst=None, ksize=None, scale=None, delta=None, borderType=None,/ )

lap_img = cv2.Laplacian(gray_img, cv2.CV_64F)
lap_img = np.uint8(np.absolute(lap_img))
#cv2.imshow('Laplacion', lap_img)                                # looks like a lightly smudged pencil shading


#  SOBEL

# Sobel Gradient Magnitude Representation computes the gradients in two directions, the x and y.

# Sobel(src, ddepth, dx, dy, dst=None, ksize=None, scale=None, delta=None, borderType=None, /)

sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1)

#cv2.imshow('Sobel X', sobelx)                         # lots of y specific gradients
#cv2.imshow('Sobel Y', sobely)                         # lots of x specific gradients

combined_sobel = cv2.bitwise_or(sobelx, sobely)
#cv2.imshow('Combined Sobel', combined_sobel)



cv2.waitKey(0)