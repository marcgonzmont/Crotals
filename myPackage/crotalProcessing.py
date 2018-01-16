import cv2
import numpy as np
import pytesseract
from myPackage import tools as tl
from matplotlib import pyplot as plt
from os.path import basename
from os import remove
from PIL import Image
from sklearn.cluster import KMeans


def validateImage(nameImage, training= False):
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
        bin_img = cv2.erode(bin_img, kernel= kernel, iterations= 3)
        # bin_img = cv2.dilate(bin_img, kernel= kernel, iterations= 4)
        # bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
        # bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    elif mode == "extr":
        n = 3
        kernel = (n, n)
        kernel2 = (3*n, 3*n)
        bin_img = cv2.dilate(bin_img, kernel= kernel, iterations= 3)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel2, iterations= 3)

    return bin_img


def skewCorrection(nameImage, training= False):
    image = cv2.imread(nameImage)
    # print(image.shape)
    if image.shape [2]==3:
        # print("COLOR")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    centroids = sorted(calcKmeans(gray, 3))
    # print(centroids)
    lvls = [int((centroids[0][0]+centroids[1][0])*0.7), int((centroids[1][0]+centroids[2][0])*0.4)]
    # lvls = [centroids[1][0] - 20, centroids[1][0] + 20]
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
        titles = ["original", "rotated", "cropped"]
        images = [image, rotated, cropped]
        title = "Angle correction: {:.3f}º".format(angle_f)
        tl.plotImages(titles, images, title, 1, 3)

        print("--- ROTATION PROCESSING INFORMATION ---\n"
              "Rotation angle of the image '{}': {:.3f}\n"
              "Final angle: {:.3f}\n\n".format(basename(nameImage), angle, angle_f))

    return cropped


def calcHistogram(nameImage, image, k, training= False):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    centroids = sorted(calcKmeans(img, k))

    hist, bins = np.histogram(img.ravel(), 256, [0, 256])

    # lvls = [int((centroids[0][0] + centroids[1][0]) * 0.6), int((centroids[1][0] + centroids[2][0]) * 0.46)]
    lvls = [centroids[1][0] - 25, centroids[1][0] + 20]
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
        plt.subplot(221), plt.imshow(img)
        plt.axis('off')
        plt.gray()

        # plt.subplot(132), plt.hist(hist, bins, [0, 256], color= 'black'), plt.hist([centroids], 50, [0,256], color= 'r'),
        # plt.hist(lvls, 50, [0, 256], color='b')

        plt.subplot(122), plt.plot(hist, color= 'black'),
        for xc in centroids:
            plt.axvline(x=xc, color='r')
        for xl in lvls:
            plt.axvline(x=xl, color='b')
        # plt.gray()

        plt.subplot(223), plt.imshow(thresh)
        plt.axis('off')

        # plt.tight_layout()
        fig.suptitle(basename(nameImage) + " image and Histogram", fontsize=14)
        plt.show()
    # white_bckg = whiteBckg(thresh)
    return thresh

def numExtraction(nameImage, bin_img, trainig= False):
    h, w = bin_img.shape
    # print(bin_img.shape)

    hist_y = np.sum(bin_img==0, axis=1)
    # print(hist_y)
    mins_idx = np.where(hist_y == min(hist_y[int(len(hist_y)*0.48) : int(len(hist_y)*0.9)]))
    maxs_idx = np.where(hist_y == max(hist_y[int(len(hist_y)*0.8) : int(len(hist_y))]))
    # print(mins)
    max_idx = maxs_idx[0][-1]
    if max_idx - 100 < mins_idx[0][-1]:
        min_idx = mins_idx[0][0]
    else:
        min_idx = mins_idx[0][-1]

    roi_num = bin_img[min_idx : max_idx-2, 25:w-30]
    roi_num = cleanImage(roi_num, "extr")

    if trainig:
        print("Image: {}\n"
              "Mins: {}"
              "Min value of hist_y and index: {} at {}\n".format(nameImage,mins_idx,min(hist_y), min_idx))
        print("Crop image from {}:{}, {}:{}".format(min_idx, max_idx, 25, w - 25))
        fig = plt.figure()
        plt.subplot(221), plt.imshow(bin_img)
        plt.gray()
        plt.axis('off')

        # # y = np.arange(len(hist_y))
        plt.subplot(122), plt.plot(hist_y, color='black'),
        for m in mins_idx[0]:
            plt.axvline(x = m, color='r')
        for m in maxs_idx[0]:
            plt.axvline(x=m, color='b')
        # plt.gca().invert_yaxis()

        plt.subplot(223), plt.imshow(roi_num)
        plt.gray()
        plt.axis('off')
        # plt.subplot(133), plt.imshow(bin_img)
        # plt.axis('off')
        # plt.gray()
        # plt.tight_layout()
        fig.suptitle(basename(nameImage) + ": ROI", fontsize=14)
        plt.show()

        return roi_num

def evaluate(nameImage, img_num, gt_num):
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
    tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
    filename = 'num_tmp.jpg'
    success = False

    cv2.imwrite(filename, img_num)
    img = Image.open(filename)
    text = pytesseract.image_to_string(img, config=tessdata_dir_config)
    remove(filename)

    if text == gt_num:
        success = True

    print("Image: {}\n"
          "GT_num: {}\n"
          "Result: {}\n"
          "Success: {}".format(nameImage, gt_num, text, success))

    return success