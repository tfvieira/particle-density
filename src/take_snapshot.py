# take_snapshot.py
# fit_ellipse

#%%
import numpy as np
import cv2 as cv

CAP_NUM_0 = 0
CAP_NUM_1 = 1

cap_0 = cv.VideoCapture(CAP_NUM_0)
cap_1 = cv.VideoCapture(CAP_NUM_1)

if not cap_0.isOpened():
    print("Cannot open camera 0.")
    exit()
else:
    print (f"Frame default resolution: {str(cap_0.get(cv.CAP_PROP_FRAME_WIDTH))}, {str(cap_0.get(cv.CAP_PROP_FRAME_HEIGHT))}")

if not cap_1.isOpened():
    print("Cannot open camera 1.")
    exit()
else:
    print (f"Frame default resolution: {str(cap_1.get(cv.CAP_PROP_FRAME_WIDTH))}, {str(cap_1.get(cv.CAP_PROP_FRAME_HEIGHT))}")


# cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 1024)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 768)

cv.namedWindow("cap_0", cv.WINDOW_KEEPRATIO)
cv.namedWindow("cap_1", cv.WINDOW_KEEPRATIO)

def my_thresh(img):
    _, out = cv.threshold(img, 127, 255, cv.THRESH_OTSU)
    return out

while True:

    # Capture frame-by-frame
    ret_0, frame_0 = cap_0.read()
    ret_1, frame_1 = cap_1.read()

    # if frame is read correctly ret is True
    if not ret_0:
        print("Can't receive frame from cap_0 (stream end?). Exiting ...")
        break
    if not ret_1:
        print("Can't receive frame from cap_1 (stream end?). Exiting ...")
        break


    gray_0 = cv.cvtColor(frame_0, cv.COLOR_BGR2GRAY)
    gray_1 = cv.cvtColor(frame_1, cv.COLOR_BGR2GRAY)

    bin_0 = my_thresh(gray_0)
    bin_1 = my_thresh(gray_1)

    # Display the resulting frame
    cv.imshow('cap_0', frame_0)
    cv.imshow('cap_1', frame_1)

    cv.imshow('bin_0', bin_0)
    cv.imshow('bin_1', bin_1)


    key = cv.waitKey(1)

    if key == ord('s'):
        cv.imwrite("frame_0.tif", frame_0, [cv.IMWRITE_TIFF_RESUNIT, 100])
        cv.imwrite("frame_1.tif", frame_1, [cv.IMWRITE_TIFF_RESUNIT, 100])
        print("Snapshot saved.")

    elif key == ord('q'):
        break

    print (f"cap_0 Resolution: {str(cap_0.get(cv.CAP_PROP_FRAME_WIDTH))}, {str(cap_0.get(cv.CAP_PROP_FRAME_HEIGHT))}")
    print (f"cap_1 Resolution: {str(cap_1.get(cv.CAP_PROP_FRAME_WIDTH))}, {str(cap_1.get(cv.CAP_PROP_FRAME_HEIGHT))}")


# When everything done, release the capture
cap_0.release()
cap_1.release()
cv.destroyAllWindows()

