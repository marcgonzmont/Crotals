import sys
import argparse
import time
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
            accept = cp.validateImage(img)
            if accept:
                cropped = cp.skewCorrection(img)
                bin_img = cp.calcHistogram(img, cropped, 3, training)
                number = cp.numExtraction(img, bin_img, training)
            else:
                continue
    else:
        start = time.time()
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
        end = time.time()
        true = cp.np.count_nonzero(results)
        false = len(results) - true
        rejected = len(test_images) - processed
        total_samples = len(test_images)

        print("--- TEST RESULTS ---\n"
              "Execution time: {:0.3f} seconds\n"
              "Number of test images: {}\n"
              "Success results: {} ({:0.3f}%)\n"
              "Fail results: {} ({:0.3f}%)\n"
              "Rejected examples: {} ({:0.3f}%)".format(end-start, total_samples, true, (true/total_samples)*100, false, (false/total_samples)*100, rejected, (rejected/total_samples)*100))

        names = ('Success', 'Fail', 'Rejected')
        idx = cp.np.arange(len(names))
        values = [true, false, rejected]
        tl.plotResults(names, idx, values)

    sys.exit(0)