import sys
import argparse
from myPackage import tools as tl
from myPackage import crotalProcessing as cp


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-tr", "--training_path", required=True,
                    help="-tr Training path of the samples")
    ap.add_argument("-te", "--test_path", required=True,
                    help="-te Test path of the samples")
    ap.add_argument("-gt", "--gt_file", required=True,
                    help="-gt GT file to measure the performance of the algorithm")
    args = vars(ap.parse_args())

    # Print information
    tl.printInformation(args)
    # Get the training and test images to process and the GT to evaluate the algorithm
    training_images = tl.natSort(tl.getSamples(args["training_path"]))
    test_images = tl.natSort(tl.getSamples(args["test_path"]))
    gt_dict = tl.getGTcsv(args["gt_file"])
    gt_numbers = list(gt_dict.values())
 

    training = False
    processed = 0
    results = cp.np.zeros(len(test_images))

    if training:
        for img in training_images:
            accept = cp.validateImage(img, training)
            if accept:
                cropped = cp.skewCorrection(img, training)
                bin_img = cp.calcHistogram(img, cropped, 3, training)
                number = cp.numExtraction(img, bin_img, training)
            else:
                continue
    else:
        for idx, img in enumerate(test_images):
            accept = cp.validateImage(img)
            if accept:
                processed += 1
                cropped = cp.skewCorrection(img)
                bin_img = cp.calcHistogram(img, cropped, 3)
                number = cp.numExtraction(img, bin_img)
                success = cp.evaluate(img, number, gt_numbers[idx])
                if success:
                    results[idx]=1
            else:
                continue
        true = cp.np.count_nonzero(results)
        false = len(results) - true
        print("--- TEST RESULTS ---\n"
              "Number of test images: {}\n"
              "True result: {} ({2.3f}%)\n"
              "False result: {} ({2.3f]%)\n"
              "Rejected examples: {}".format(len(test_images), true, (true/len(test_images))*100, false, (false/len(test_images))*100, len(test_images)-processed))