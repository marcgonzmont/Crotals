import cv2
import numpy as np
from PIL import Image as im
from matplotlib import pyplot as plt

black_image = np.zeros((100,100), np.uint8)
white_image = np.ones((255,255), np.uint8)*255

# white = (255,255,255)
# black = (0,0,0)
# white_image = im.new("RGB", [255, 255, 3], 'white')#.convert("L")
# black_image = im.new("RGB", [255, 255, 3], 'black')#.convert("L")

# ocv_white = cv2.cvtColor(np.array(white_image), cv2.COLOR_BGR2GRAY)
# ocv_black = cv2.cvtColor(np.array(black_image), cv2.COLOR_BGR2GRAY)

# ocv_white = np.array(white_image)
# ocv_black = np.array(black_image)

cv2.imshow("test", white_image)
cv2.waitKey(500)

cv2.imshow("test", cv2.bitwise_xor(black_image, white_image))
cv2.waitKey(500)

cv2.destroyAllWindows()
# cv2.imshow("test", white_image)
# cv2.waitKey(500)
# cv2.destroyAllWindows()

# fig = plt.figure()
# plt.subplot(121), plt.imshow(ocv_black)
# plt.axis('off')
#
# plt.subplot(122), plt.imshow(ocv_white)
# plt.axis('off')
#
# fig.suptitle("White background", fontsize=14)
# plt.show()