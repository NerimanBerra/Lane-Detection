import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import display
video = cv2.VideoCapture(r'C:\Users\berra\Downloads\LaneDetection.mp4')
# running the loop 
while True: 
  
    # extracting the frames 
    ret, img = video.read() 
      
    # converting to gray-scale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    # displaying the video 
    cv2.imshow("Live", gray) 
  
    # exiting the loop 
    key = cv2.waitKey(1) 
    if key == ord("q"): 
        break
      
# closing the window 
cv2.destroyAllWindows() 
source.release()
blur=cv2.GaussianBlur(gray, (5,5), 0)
def canny(image):
    edges = cv2.Canny(image,50,150)
    return edges
def region(image):
    height, width = image.shape
    triangle = np.array([
                       [(100, height), (475, 325), (width, height)]
                       ])
    
    mask = np.zeros_like(image)
    
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask
lines = cv2.HoughLinesP(isolated, rho=2, theta=np.pi/180, threshold=100, np.array([]), minLineLength=40, maxLineGap=5)
def make_points(image, average): 
 slope, y_int = average 
 y1 = image.shape[0]
 y2 = int(y1 * (3/5))
 x1 = int((y1 — y_int) // slope)
 x2 = int((y2 — y_int) // slope)
 return np.array([x1, y1, x2, y2])
def display_lines(image, lines):
 lines_image = np.zeros_like(image)
 if lines is not None:
   for line in lines:
     x1, y1, x2, y2 = line
     cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
 return lines_image
lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
copy = np.copy(image1)
grey = grey(copy)
gaus = gauss(grey)
edges = canny(gaus,50,150)
isolated = region(edges)lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average(copy, lines)
black_lines = display_lines(copy, averaged_lines)
lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
cv2.imshow("lanes", lanes)
cv2.waitKey(0)
