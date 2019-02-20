import cv2
import csv
import numpy as np

lines = []

for n in range(1, 200):
    image = cv2.imread("pics/cat5/im" + str(n) + ".jpg", cv2.IMREAD_GRAYSCALE)

    image_array = np.array(image)
    image_array[image_array > 0] = 1
    flatten_array = image_array.ravel()
    flatten_list = list(flatten_array)
    flatten_list.append("ges1")

    lines.append(flatten_list)

    print(len(flatten_list))
    cv2.imshow("hand", image)
    cv2.waitKey(1)

for n in range(1, 200):
    image = cv2.imread("pics/cat6/im" + str(n) + ".jpg", cv2.IMREAD_GRAYSCALE)
    print(image.shape)

    image_array = np.array(image)
    image_array[image_array > 0] = 1
    flatten_array = image_array.ravel()
    flatten_list = list(flatten_array)
    flatten_list.append("ges2")

    lines.append(flatten_list)

    print('dataset width: ', len(flatten_list))
    cv2.imshow("hand", image)
    cv2.waitKey(1)

with open('gestures.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)

cv2.destroyAllWindows()
