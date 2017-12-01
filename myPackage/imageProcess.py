import cv2
import imutils
from imutils import contours
import numpy as np
from Crotals.myPackage import tools as tl
from matplotlib import pyplot as plt


def crotalContour(nameImage):
    image = cv2.imread(nameImage)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plt.hist(gray.ravel(), 256, [0, 256])
    # plt.show()
    thresh = cv2.threshold(gray, 0, 50, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # thresh_pad = cv2.copyMakeBorder(thresh, 20, 30, 30, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    '''
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        cv2.drawContours(image, [box], 0, (255, 255, 0), 2)
    '''
    # find the largest contour
    rect = cv2.minAreaRect(max(contours, key = cv2.contourArea))
    print("Minimum area: {}".format(cv2.contourArea(max(contours, key = cv2.contourArea))))
    box = cv2.boxPoints(rect)
    # convert all coordinates floating point values to int
    box = np.int0(box)
    # draw a red 'nghien' rectangle
    cv2.drawContours(image, [box], 0, (255, 255, 0), 2)

    # CROP THE IMAGE
    # cv2.drawContours(image, contours, -1, (255, 255, 0), 2)
    # crop = image[y:y + h, x:x + w]
    '''
    thresh_pad = cv2.copyMakeBorder(thresh, 20, 30, 30, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    kernel = np.ones((5, 5), np.uint8)
    iter = 1
    erosion = cv2.erode(thresh_pad, kernel, iterations= iter)
    dilation = cv2.dilate(thresh_pad, kernel, iterations= iter)
    opening = cv2.morphologyEx(thresh_pad, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(thresh_pad, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(thresh_pad, cv2.MORPH_GRADIENT, kernel)
    tophat = cv2.morphologyEx(thresh_pad, cv2.MORPH_TOPHAT, kernel)

    # titles = ["erosion", "dilatation", "opening", "closing", "gradient", "tophat"]
    # images = [erosion, dilation, opening, closing, gradient, tophat]
    # title = "Test"
    # tl.plotImages(titles, images, title, 2, 3)
    
    # find contours and get the external one
    contours = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    # contours = cv2.findContours(thresh_pad, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

    cv2.drawContours(erosion, contours, -1, (0, 0, 255), 2)
    '''
    tl.plt.imshow(thresh, 'gray')
    tl.plt.show()
    # tl.plt.imshow(crop, 'gray')
    # tl.plt.show()
    tl.plt.imshow(image, 'gray')
    tl.plt.show()

def skewCorrection(nameImage):
    image = cv2.imread(nameImage)
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bitwise_not(gray)

    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh_pad = cv2.copyMakeBorder(thresh,10,10,10,10,cv2.BORDER_CONSTANT,value=(0,0,0))

    coords = np.column_stack(np.where(thresh_pad > 0))
    rect = cv2.minAreaRect(coords)
    # contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    # rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    # print("Minimum area: {}".format(cv2.contourArea(max(contours, key=cv2.contourArea))))
    box = cv2.boxPoints(rect)
    # convert all coordinates floating point values to int
    box = np.int0(box)
    # draw a red 'nghien' rectangle
    # cv2.drawContours(image, [box], 0, (255, 255, 0), 2)
    cv2.drawContours(thresh_pad, [box], 0, (255, 255, 255), 2)

    angle = rect[-1]
    print(angle)
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle
    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated_th = cv2.warpAffine(thresh_pad, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    titles = ["original", "gray", "thresh_pad", "rotated", "rotated thresh"]
    images = [image, gray, thresh_pad, rotated, rotated_th]
    title = "Test"
    tl.plotImages(titles,images, title, 2, 3)
    # show the output image
    print("--- PROCESSING INFORMATION ---\n"
          "Rotation angle of the crotal: {:.3f}".format(angle))


    return rotated

def extractDigitsOCR(ocrRef):
    dict_list = []
    # load the reference OCR image from disk, convert it to grayscale,
    # and threshold it, such that the digits appear as *white* on a
    # *black* background
    # and invert it, such that the digits appear as *white* on a *black*
    ref = cv2.imread(ocrRef)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
    # cv2.imshow("OCR-A", ref)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()

    # find contours in the OCR-A image (i.e,. the outlines of the digits)
    # sort them from left to right, and initialize a dictionary to map
    # digit name to the ROI
    refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)[1]
    refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

    digits = {}
    # loop over the OCR-A reference contours
    for (i, c) in enumerate(refCnts):
        # compute the bounding box for the digit, extract it, and resize
        # it to a fixed size
        (x, y, w, h) = cv2.boundingRect(c)
        roi = ref[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))

        # update the digits dictionary, mapping the digit name to the ROI
        digits[i] = roi
    return digits

def processImage(in_image):
    image_rotated = skewCorrection(in_image)

    # initialize a rectangular and square kernels for top-hat morphological and closing
    # operations, respectively
    thKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    image_rotated = imutils.resize(image_rotated, width=300)
    gray = cv2.cvtColor(image_rotated, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("OCR-A", gray)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()