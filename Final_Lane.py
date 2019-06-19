#
# Author: Arsh
# Created On: 06 May, 2019 at 20:21:56
# Username: arsh_16
#

import cv2, numpy as np

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    canny_image = cv2.Canny(blur, 50, 150)
    return canny_image

def region_of_interest(img):
    height = img.shape[0]
    mask = np.zeros_like(img)
    triangle = np.array([[
        (200, height),
        (550, 250),
        (1100, height)
    ]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])
    y2=  int(y1*3/5)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return [x1, y1, x2, y2]

def average_slope_lines(img, lines):
    left_fit, right_fit = [], []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = fit
            if slope < 0:
                left_fit.append(fit)
            else:
                right_fit.append(fit)
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = right_line = None
    if left_fit:
        left_line = make_points(img, left_fit_average)
    if right_fit:
        right_line = make_points(img, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines

def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            if line is None: continue
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

cap = cv2.VideoCapture('test1.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
while (cap.isOpened()):
    _, frame = cap.read()
    if frame is None: break
    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), 40, 5)
    averaged_lines = average_slope_lines(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    out.write(combo_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()