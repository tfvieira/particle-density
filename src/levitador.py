#%% levitador.py
import numpy as np
import cv2

cap = cv2.VideoCapture('../dataset/levitador/levitador.mov')


cv2.namedWindow("frame", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("diff" , cv2.WINDOW_KEEPRATIO)

frame_number = 0

while 1:

    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame from video capture (stream end?). Exiting ...")
        break
    
    if frame_number % 2 == 0:
        frame_0 = frame.copy()
    if frame_number % 2 == 1:
        frame_1 = frame.copy()
        diff = cv2.subtract(frame_0, frame_1)
    

    # Read key pressed.
    key = 0xFF & cv2.waitKey(33)
    # if key == ord('s'):
    cv2.imwrite(f"frames/frame_{frame_number:08d}.tif", frame)
    if key == ord('q'):
        break

    cv2.putText(frame, f"frame_{frame_number:08d}.tif", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)
    cv2.imshow('frame', frame)
    if 'diff' in locals():
        cv2.imshow('diff', diff)
    frame_number += 1

cv2.destroyAllWindows()
cap.release()
