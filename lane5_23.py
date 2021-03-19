import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)##gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0) ## typically to reduce image noise and reduce detail.
    canny = cv2.Canny(blur,50,150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image) # same dimmension  but totally black
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)#reshape into one dimmensional array
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),5)#   ,coordinates of image, color of line, thickness of the line
    return line_image

# next step limit the region  trace a trangle


def region_of_interest(image):
    height = image.shape[0]  #alternative: height,width,channel = im.shape
    width = image.shape[1]
    polygons = np.array([
    [ (0 , height) , (width , height) , (int(width/2) , int(height/2)) ] #non pi array
    ])# anarray of polygons （in this case only one polygon）

    mask = np.zeros_like(image) # same dimension as the canny image  but have the 0 intensity  question? what if you change the environment? create an array of zeros
    cv2.fillPoly(mask, polygons, 255) # fill the polygon with tranonly accepts array of polygons the contour will be all white
    mask_image = cv2.bitwise_and(image, mask) #  and gate      1111 region_of_interest remain unaffected
    return mask_image


def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2),1)  #return slope and intercept
        slope = parameters[0]
        intercept = parameters[1]
        if slope <0 :
            left_fit.append((slope,intercept))
        if slope > 0 :
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit,axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line,right_line])

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0] #heights
    y2 = int(y1*(3/5))  #lines go 3/5 of the way upward
    x1 = int((y1- intercept)/slope)
    x2 = int((y2- intercept)/slope)
    return np.array([x1,y1,x2,y2])



image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
cropped_image = region_of_interest(canny)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 50, np.array([]),minLineLength=5,maxLineGap = 10)
#(1. the image you want to detect lines, 2(size) 2pixels, 3(size)theita,  4threshold minimem number needed to detect a line 5placeholder array    6 lines having a length of less than 10 are rejected)
average_Lines = average_slope_intercept(lane_image,lines)
hough_image = display_lines(lane_image, lines)
# hough space  y = mx + b   many possible lines could cross
# possible    identify the parameters   candidate
# I can see   given   all -----  for   we are going to
#line of best fit
# what if verticle or horizontal lins     -->  fail
#  p = xcos   +   ysin      polar coordinates   rou the perpendicular distance from origin to the lines
combo_image = cv2.addWeighted(lane_image, 0.8, hough_image, 1,1) #line will be more apparent after blended
new_lane_image = display_lines(lane_image, average_Lines)  # line that is averaged out
new_combo_image = cv2.addWeighted(new_lane_image, 0.8, hough_image, 1,1)
cv2.imshow('result',lane_image)
cv2.imshow('result1',gray)
cv2.imshow('result2',blur)
cv2.imshow('result3',canny)
cv2.imshow('result4',region_of_interest(canny))
cv2.imshow('result5',cropped_image)
cv2.imshow('result6',hough_image)
cv2.imshow('result7',combo_image)
cv2.imshow('result8',new_lane_image)
cv2.imshow('result9',new_combo_image)

plt.imshow(canny) #result 3
plt.show()
cv2.waitKey(0)


# below is for video
cap = cv2.VideoCapture("name")

while(cap.isOpened()):
    _, frame = cap.read #
    canny = canny(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    cropped_image = region_of_interest(canny)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 40, np.array([]),minLineLength=10,maxLineGap = 20)
    average_Lines = average_slope_intercept(frame,lines)
    hough_image = display_lines(frame, lines)
    combo_image = cv2.addWeighted(frame, 0.8, hough_image, 1,1) #line will be more apparent after blended
    new_lane_image = display_lines(frame, average_Lines)  # line that is averaged out
    new_combo_image = cv2.addWeighted(frame, 0.8, hough_image, 1,1)
    cv2.imshow("result", new_combo_image)
    cv2.waitKey(1)
#5.23   binary   why?  255 all 11111111   0: 0000000


#5.28    m默认是单线   但是问题是有两条线  no this is an exception cause its center of the line
