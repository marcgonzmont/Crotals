import argparse
from myPackage import tools as tl
from myPackage import imageProcess as imp


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-tr", "--training_path", required=True,
                    help="-tr Training path of the samples")
    ap.add_argument("-te", "--test_path", required=True,
                    help="-te Test path of the samples")
    ap.add_argument("-gt", "--gt_file", required=True,
                    help="-gt GT file to measure the performance of the algorithm")
    ap.add_argument("-otr", "--output_train_path", required=False,
                    help="-o Path where the training results will be stored")
    ap.add_argument("-ote", "--output_test_path", required=False,
                    help="-o Path where the test results will be stored")
    args = vars(ap.parse_args())

    # Print information
    tl.printInformation(args)
    # Get the training and test images to process and the GT to evaluate the algorithm
    training_images = tl.natSort(tl.getSamples(args["training_path"]))
    test_images = tl.natSort(tl.getSamples(args["test_path"]))
    gt_dict = tl.getGTcsv(args["gt_file"])
    training = True
    # img = training_images[0]
    # Loop for extract the images
    # idx = 1
    # image = cv2.imread(training_images[idx], 0)
    for img in training_images:
        cropped = imp.skewCorrection(img, training)
        bin_img = imp.calcHistogram(img, cropped, 3, training)
        number = imp.numExtraction(img, bin_img, training)
        # imp.crotalContour(img)
        # image_rotated = imp.skewCorrection(img)

