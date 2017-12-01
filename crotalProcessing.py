import cv2
import argparse
from Crotals.myPackage import tools as tl
from Crotals.myPackage import imageProcess as imp

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-tr", "--training_path", required=True,
                    help="-tr Training path of the samples")
    ap.add_argument("-te", "--test_path", required=True,
                    help="-te Test path of the samples")
    ap.add_argument("-gt", "--gt_file", required=True,
                    help="-gt GT file to measure the performance of the algorithm")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("-rA", "--OCRreferenceA",
                       help="-r Reference OCR-A image")
    group.add_argument("-rB", "--OCRreferenceB",
                       help="-r Reference OCR-B image")
    ap.add_argument("-o", "--output_path", required=False,
                    help="-o Path where the results will be stored")
    args = vars(ap.parse_args())

    # Print information
    tl.printInformation(args)
    # Get the training and test images to process and the GT to evaluate the algorithm
    training_images = tl.natSort(tl.getSamples(args["training_path"]))
    test_images = tl.natSort(tl.getSamples(args["test_path"]))
    gt_dict = tl.getGTcsv(args["gt_file"])

    # Process reference image OCR-A or OCR-B to extract the digits
    reference = args["OCRreferenceA"] if args.get("OCRreferenceA") else args.get("OCRreferenceB")
    ocrDict = imp.extractDigitsOCR(reference)
    # if args.get("referenceA"):
    #     ocrDict= imp.extractDigitsOCR(args["referenceA"])
    # elif args.get("referenceB"):
    #     ocrDict = imp.extractDigitsOCR(args["referenceB"])

    # print(ocrList)
    # cv2.imshow("OCR-A", cv2.imread(ocrList[0]))
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()



    # Loop for extract the images
    # idx = 1
    # image = cv2.imread(training_images[idx], 0)
    for img in training_images:
        # imp.crotalContour(img)
        image_rotated = imp.skewCorrection(img)
    # plt.hist(image_rotated.ravel(), 256, [0, 256])
    # plt.show()
