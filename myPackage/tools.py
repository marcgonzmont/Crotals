from os import listdir, makedirs, errno, rename
from os.path import isfile, join, altsep
from natsort import natsorted, ns
from matplotlib import pyplot as plt
import csv

def getSamples(path):
    samples = [altsep.join((path, f)) for f in listdir(path)
              if isfile(join(path, f)) and f.endswith('.TIF')]
    return samples

def getSamplesTess(path):
    samples = [altsep.join((path, f)) for f in listdir(path)
              if isfile(join(path, f)) and f.endswith('.jpg')]
    return samples

def makeDir(path):
    '''
    To create output path if doesn't exist
    see: https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
    :param path: path to be created
    :return: none
    '''
    try:
        makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def natSort(list):
    '''
    Sort frames with human method
    see: https://pypi.python.org/pypi/natsort
    :param list: list that will be sorted
    :return: sorted list
    '''
    return natsorted(list, alg=ns.IGNORECASE)

def plotImages(titles, images, title, row, col):
    fig = plt.figure()
    for i in range(len(images)):
        plt.subplot(row, col, i + 1), plt.imshow(images[i], 'gray')
        if len(titles) != 0:
            plt.title(titles[i])
        plt.gray()
        plt.axis('off')
    fig.suptitle(title, fontsize=14)
    plt.show()

def renameFiles(samples_path):
    files = listdir(samples_path)
    i = 0
    for file in files:
        rename(join(samples_path, file), join(samples_path,'img'+str(i)+'.png'))
        i = i+1

def getGTcsv(file):
    with open(file, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader, None)
        mydict = {rows[0]: rows[1] for rows in reader}
        return mydict

def printInformation(args):
    print("--- INFORMATION ---")
    print("Training path: {}\n"
          "Test path: {}\n"
          "GT file: {}".format(args["training_path"], args["test_path"],args["gt_file"]))
    reference = args["OCRreferenceA"] if args.get("OCRreferenceA") else args.get("OCRreferenceB")
    print("OCR reference: {}".format(reference))
    if args.get("output_path"):
        print("Output path: {}\n\n".format(args["output_path"]))