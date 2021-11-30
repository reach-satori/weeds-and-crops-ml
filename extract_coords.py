import numpy as np
from load_images import load_images
import cv2
# just a little script to pop up an opencv window and print out the coordinates as I clicked,
# so I could get the boxes for the plots

def onclick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)


cv2.namedWindow("disp")
cv2.setMouseCallback("disp", onclick)
for img in load_images():
    print("##############")
    while True:
        img = img[...,:3].copy()
        img = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX) * 255
        cv2.imshow("disp", np.swapaxes(img.astype(np.uint8), 0, 1))
        k = cv2.waitKey()
        if k == ord("n"):
            break
        if k == ord("q"):
            exit()
