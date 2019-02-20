import cv2
import numpy as np


def nothing():
    pass


path_to_im = "pics/im"

cap = cv2.VideoCapture(0)
cv2.namedWindow("Track bars")

cv2.createTrackbar("L - H", "Track bars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Track bars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Track bars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Track bars", 0, 255, nothing)
cv2.createTrackbar("U - S", "Track bars", 0, 255, nothing)
cv2.createTrackbar("U - V", "Track bars", 0, 255, nothing)

sift = cv2.ORB_create(nfeatures=150)
n = 0
for i in range(200):
    n += 1
    ret, frame = cap.read()
    # gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # processed = frame
    # y, x, d = frame.shape
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.pyrDown(frame)
    frame = cv2.pyrDown(frame)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Track bars")
    l_s = cv2.getTrackbarPos("L - S", "Track bars")
    l_v = cv2.getTrackbarPos("L - V", "Track bars")
    u_h = cv2.getTrackbarPos("U - H", "Track bars")
    u_s = cv2.getTrackbarPos("U - S", "Track bars")
    u_v = cv2.getTrackbarPos("U - V", "Track bars")

    # lower_blue = np.array([l_h, l_s, l_v])
    # upper_blue = np.array([u_h, u_s, u_v])
    lower = np.array([0, 134, 0])
    upper = np.array([35, 211, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)

    # frame = cv2.bitwise_and(frame, frame, mask=mask)

    # _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)

    canny = cv2.Canny(frame, 100, 150)
    print(canny.shape)
    ret, canny = cv2.threshold(canny, 127, 255, cv2.THRESH_BINARY)
    print(canny.shape)
    # canny = cv2.bitwise_and(canny, canny, mask=mask)
    # canny = cv2.Canny(canny, 100, 150)
    # keypoints, descriptors = sift.detectAndCompute(canny, None)
    # canny = cv2.drawKeypoints(canny, keypoints, None)
    # print(len(keypoints))

    cv2.imshow("frame", frame)
    # cv2.imshow("HSV frame", mask)
    cv2.imshow("canny", canny)
    # print(mask[1])

    stri = path_to_im + str(n) + ".jpg"
    # print(stri)
    cv2.imwrite(stri, canny)

    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
