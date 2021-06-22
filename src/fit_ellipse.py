# fit_ellipse

#%%
import numpy as np
import cv2 as cv
import random as rng

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv.namedWindow("frame", cv.WINDOW_KEEPRATIO | cv.WINDOW_NORMAL)
cv.namedWindow("drawing", cv.WINDOW_KEEPRATIO | cv.WINDOW_NORMAL)

def tuple2ellipse(e):
    cx = int(e[0][0])
    cy = int(e[0][1])
    w  = int(e[1][0]/2)
    h  = int(e[1][1]/2)
    a  = int(e[2])
    return ((cx, cy), (w, h), a)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    smooth = cv.medianBlur(gray, 35)

    _, bin = cv.threshold(smooth, 127, 255, cv.THRESH_OTSU)

    _, contours, hierarchy = cv.findContours(bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)



    drawing = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    for i, contour in enumerate(contours):
        color = (0, 0, 255)
        cv.drawContours(frame, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
        
        if len(contour) >= 5:
            e0 = cv.fitEllipse(contour)  # Ellipse fitting
            e  = tuple2ellipse(e0)
            cv.ellipse(frame, e[0], e[1], e[2], 0, 360, 255, 4)
            print(e)

    print("\n\n\n")

    # cv.ellipse(drawing, e[0], e[1], 0, 0, e[2], 255, -1)

    # Display the resulting frame
    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

# cv.imwrite("frame.png", frame)


# %%
