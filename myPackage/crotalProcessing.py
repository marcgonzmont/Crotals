import cv2
import re
import numpy as np
import pytesseract
import tesserocr
from myPackage import tools as tl
from matplotlib import pyplot as plt
from os.path import basename, getsize
from PIL import Image
from sklearn.cluster import KMeans


def validateImage(nameImage, training= False):
    if getsize(nameImage) != 0:
        image = cv2.imread(nameImage)
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
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
    else:
        return False


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
        kernel2 = (11,11)
        # bin_img = cv2.erode(bin_img, kernel= kernel, iterations= 3)
        bin_img = cv2.dilate(bin_img, kernel= kernel, iterations= 4)
        # bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel2, iterations=3)
    elif mode == "extr":
        n = 3
        kernel = (n, n)
        kernel2 = (3*n, 3*n)
        bin_img = cv2.dilate(bin_img, kernel= kernel, iterations= 3)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel2, iterations= 3)

    return bin_img


def skewCorrection(nameImage, training= False):
    image = cv2.imread(nameImage)
    if image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    # find the largest contour
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    angle = rect[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle_f = (90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle_f = angle

    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_f, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    rotated_copy = rotated.copy()
    rotated_gr_copy = cv2.cvtColor(rotated_copy, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(rotated_gr_copy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

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
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cleanImage(thresh, "hist")
    if training:
        fig = plt.figure(figsize=(10,5))
        plt.subplot(221), plt.imshow(img)
        plt.axis('off')
        plt.gray()

        plt.subplot(122), plt.plot(hist, color= 'black')

        plt.subplot(223), plt.imshow(thresh)
        plt.axis('off')

        fig.suptitle(basename(nameImage) + ": Otsu's threshold", fontsize=14)
        plt.show()

    return thresh

def numExtraction(nameImage, bin_img, trainig= False):
    h, w = bin_img.shape
    hist_y = np.sum(bin_img==0, axis=1)
    mins_idx = np.where(hist_y == min(hist_y[int(len(hist_y)*0.48) : int(len(hist_y)*0.9)]))
    maxs_idx = np.where(hist_y == max(hist_y[int(len(hist_y)*0.8) : int(len(hist_y))]))
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

        plt.subplot(122), plt.plot(hist_y, color='black'),
        for m in mins_idx[0]:
            plt.axvline(x = m, color='r')
        for m in maxs_idx[0]:
            plt.axvline(x=m, color='b')

        plt.subplot(223), plt.imshow(roi_num)
        plt.gray()
        plt.axis('off')

        fig.suptitle(basename(nameImage) + ": ROI", fontsize=14)
        plt.show()

    return roi_num

def evaluate(nameImage, img_num, gt_num):
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
    tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata" --psm 6 --eom 3 -c tessedit_char_whitelist=0123456789'
    success = False

    text = pytesseract.image_to_string(Image.fromarray(img_num).convert('L'), config= tessdata_dir_config)
    text = re.sub("[^0-9]{5}$", "", text)

    if len(text) > 5:
        text = text[:5]
    if text == gt_num:
        success = True

    print("Image: {}\n"
          "GT_num: {}\n"
          "Result: {}\n"
          "Success: {}\n".format(basename(nameImage), gt_num, text, success))

    return success