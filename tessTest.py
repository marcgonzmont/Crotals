from Delivery1.myPackage import tools as tl
from os.path import altsep, join
from PIL import Image
import pytesseract
import argparse
import cv2
import os


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True,
                    help="path to input image to be OCR'd")
    args = vars(ap.parse_args())

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
    tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'

    images = tl.natSort(tl.getSamplesTess(args["path"]))
    num = 0
    for img in images:

        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        filename = "tmp{}.png".format(num)

        cv2.imwrite(filename, gray)
        img = Image.open(filename)

        text = pytesseract.image_to_string(img, lang='eng', config=tessdata_dir_config)
        os.remove(filename)
        print(text)
    
        # show the output images
        cv2.imshow("Image", image)
        cv2.imshow("Output", gray)
        cv2.waitKey(1000)
        num += 1
    cv2.destroyAllWindows()
