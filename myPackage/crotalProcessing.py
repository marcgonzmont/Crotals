import cv2
import numpy as np
from myPackage import tools as tl
from matplotlib import pyplot as plt
from os.path import basename
from sklearn.cluster import KMeans


def validateImage(nameImage, training):
    image = cv2.imread(nameImage)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    tot_px = image.shape[0]*image.shape[0]
    num_black = tot_px - cv2.countNonZero(thresh)
    if training:
        print("\nImage:{}\n"
              "Total pixels: {}\n"
              "Black pixels: {}\n"
              "Threshold: {}\n"
              "Percentage: {:2.3f}\n".format(nameImage, tot_px, num_black, tot_px*0.5, (num_black/tot_px)*100))

    if num_black > tot_px*0.75:
        return False
    else:
        return True


def calcKmeans(image, k):
    if len(image.shape) == 2:
        image_array = image.reshape((image.shape[0] * image.shape[1], 1))
    elif len(image.shape) == 1:
        image_array = image.reshape(-1, 1)
    clt = KMeans(n_clusters=k)
    clt.fit(image_array)

    return clt.cluster_centers_


def cleanImage(bin_img, mode):
    if mode == "hist":
        n = 3
        kernel = (n, n)
        # bin_img = cv2.erode(bin_img, kernel= kernel, iterations= 4)
        # bin_img = cv2.dilate(bin_img, kernel= kernel, iterations= 4)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    elif mode == "extr":
        n = 3
        kernel = (n, n)
        bin_img = cv2.erode(bin_img, kernel= kernel, iterations= 2)


    return bin_img


def skewCorrection(nameImage, training):
    image = cv2.imread(nameImage)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    centroids = sorted(calcKmeans(gray, 3))
    # print(centroids)
    lvls = [int((centroids[0][0]+centroids[1][0])*0.7), int((centroids[1][0]+centroids[2][0])*0.4)]
    # print(lvls)
    thresh = cv2.threshold(gray, lvls[0], lvls[1], cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
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
    thresh = cv2.threshold(rotated_gr_copy, lvls[0], lvls[1], cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    thresh = cleanImage(thresh, "hist")

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]



    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = rotated_copy[y:y + h, x:x + w]

    if training:
        titles = ["original", "rotated", "threshold", "cropped"]
        images = [image, rotated, thresh, cropped]
        title = "Angle correction: {:.3f}º".format(angle_f)
        tl.plotImages(titles, images, title, 2, 2)

        print("--- ROTATION PROCESSING INFORMATION ---\n"
              "Rotation angle of the image '{}': {:.3f}\n"
              "Final angle: {:.3f}\n\n".format(basename(nameImage), angle, angle_f))

    return cropped


def calcHistogram(nameImage, image, k, training):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    centroids = sorted(calcKmeans(img, k))

    hist, bins = np.histogram(img.ravel(), 256, [0, 256])

    lvls = [int((centroids[0][0] + centroids[1][0]) * 0.6), int((centroids[1][0] + centroids[2][0]) * 0.46)]
    # lvls = [int(centroids[1][0]-5), int(centroids[1][0]+5)]
    # Threshold the image based on the semi sum of the two last centroids of K-Means
    # thresh = cv2.threshold(image, lvls[1], 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.threshold(img, lvls[1], 255, cv2.THRESH_BINARY)[1] # lvls[1], 255    lvls[0], lvls[1]
    thresh = cleanImage(thresh, "hist")
    if training:

        # print("--- HISTOGRAM PROCESSING INFORMATION ---\n"
        #       "K-Means centroids for K = {}: {}\n"
        #       "Threshold levels: {} - {}\n\n".format(k, sort_centr, lvls[0], lvls[1]))
        # Plot the original image and his histogram
        fig = plt.figure(figsize=(10,5))
        plt.subplot(131), plt.imshow(img)
        plt.axis('off')

        plt.subplot(132), plt.hist(hist, bins, [0, 256], color= 'black'), plt.hist([centroids], 50, [0,256], color= 'r'),
        plt.hist(lvls, 40, [0, 256], color='b')
        plt.gray()

        plt.subplot(133), plt.imshow(thresh)
        plt.axis('off')

        fig.suptitle(basename(nameImage) + " image and Histogram", fontsize=14)
        plt.show()
    # white_bckg = whiteBckg(thresh)
    return thresh

def numExtraction(nameImage, bin_img, trainig):
    h, w = bin_img.shape
    print(bin_img.shape)

    hist_y = np.sum(bin_img==0, axis=1)
    print(hist_y.shape)

    centroids = sorted(calcKmeans(hist_y, 4))
    # clt = KMeans(n_clusters= 3)
    # print(len(hist_y.reshape(-1, 1)))
    # clt.fit(hist_y.reshape(-1, 1))
    # centroids = sorted(clt.cluster_centers_)
    # bin_img = cleanImage(bin_img, "extr")
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
        plt.hist([centroids], 30, color='r', orientation='horizontal')
        plt.gca().invert_yaxis()
        plt.gray()

        plt.subplot(133), plt.imshow(roi_num)
        plt.axis('off')

        fig.suptitle(basename(nameImage) + " cropped and Histogram (y)", fontsize=14)
        plt.show()
