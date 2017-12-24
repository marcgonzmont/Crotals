import cv2
import numpy as np
from myPackage import tools as tl
from matplotlib import pyplot as plt
from os.path import basename
from sklearn.cluster import KMeans


def skewCorrection(nameImage, training):
    image = cv2.imread(nameImage)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    # find the largest contour
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    angle = rect[-1]

    if angle < -45:
        # angle_f = -(90 + angle)
        angle_f = (90 + angle)
        # print("{:.3f} < -45 --> {:.3f}".format(angle, angle_f))

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle_f = angle
        # print("{:.3f} > -45 --> {:.3f}".format(angle, angle_f))
    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_f, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    rotated_copy = rotated.copy()
    rotated_gr_copy = cv2.cvtColor(rotated_copy, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(rotated_gr_copy, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = rotated_gr_copy[y:y + h, x:x + w]

    if training:
        titles = ["original", "rotated", "cropped"]
        images = [image, rotated, cropped]
        title = "Angle correction: {:.3f}º".format(angle_f)
        tl.plotImages(titles, images, title, 1, 3)

        print("--- ROTATION PROCESSING INFORMATION ---\n"
              "Rotation angle of the image '{}': {:.3f}\n"
              "Final angle: {:.3f}\n\n".format(basename(nameImage), angle, angle_f))

    return cropped


def calcHistogram(nameImage, image, k, training):
    # image = cv2.imread(nameImage)
    # image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_array = image.reshape((image.shape[0] * image.shape[1], 1))
    hist, bins = np.histogram(image.ravel(), 256, [0, 256])

    clt = KMeans(n_clusters= k)
    clt.fit(image_array)
    centroids = sorted(clt.cluster_centers_)
    # lvls = [int((centroids[0][0]+centroids[1][0])/2), int((centroids[1][0]+centroids[2][0])/2)]
    lvls = [int(centroids[1][0]-5), int(centroids[1][0]+5)]
    # Threshold the image based on the semi sum of the two last centroids of K-Means
    thresh = cv2.threshold(image, lvls[1], 255, cv2.THRESH_BINARY)[1]
    # thresh = cv2.threshold(image, lvls[0], lvls[1], cv2.THRESH_BINARY)[1]
    # thresh = cv2.adaptiveThreshold(image, lvls[1], cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 13, 8)
    if training:
        print("--- HISTOGRAM PROCESSING INFORMATION ---\n"
              "K-Means centroids for K = {}: {}\n"
              "Threshold levels: {} - {}\n\n".format(k, centroids, lvls[0], lvls[1]))
        # Plot the original image and his histogram
        fig = plt.figure()
        plt.subplot(131), plt.imshow(image)
        plt.axis('off')

        plt.subplot(132), plt.hist(hist, bins, [0, 256], color= 'black'), plt.hist(clt.cluster_centers_, 32, [0,256], color= 'r'),
        plt.hist(lvls, 100, [0, 256], color='b')
        plt.gray()

        plt.subplot(133), plt.imshow(thresh)
        plt.axis('off')

        fig.suptitle(basename(nameImage) + " image and Histogram", fontsize=14)
        plt.show()

    return thresh

def numExtraction(nameImage, bin_img, trainig):
    h, w = bin_img.shape
    print(bin_img.shape)

    hist_y = np.sum(bin_img==0, axis=1)
    # print(len(hist_y))
    clt = KMeans(n_clusters= 4)
    # print(len(hist_y.reshape(-1, 1)))
    clt.fit(hist_y.reshape(-1, 1))
    centroids = sorted(clt.cluster_centers_)
    print(centroids)
    print(len(centroids))
    idx = np.where(np.flip(hist_y, 0) == int(centroids[-2][0]))[0][0]
    roi_num = bin_img[idx-20 : h, 25:w-25]

    if trainig:
        print("Crop image from {}:{}, {}:{}".format(idx, h, 0, w - 1))
        fig = plt.figure()
        plt.subplot(131), plt.imshow(bin_img)
        plt.axis('off')

        plt.subplot(132), plt.hist(hist_y, h, color='black', orientation='horizontal'), \
        plt.hist(clt.cluster_centers_, 40, color='r', orientation='horizontal')
        plt.gca().invert_yaxis()
        plt.gray()

        plt.subplot(133), plt.imshow(roi_num)
        plt.axis('off')

        fig.suptitle(basename(nameImage) + " cropped and Histogram (y)", fontsize=14)
        plt.show()
