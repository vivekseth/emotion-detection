import cv
import cv2
import time
from PIL import Image
import numpy as np
import csv
import logistic
import mouthdetection as m

WIDTH, HEIGHT = 28, 10 # all mouth images will be resized to the same size
dim = WIDTH * HEIGHT # dimension of feature vector

"""
pop up an image showing the mouth with a blue rectangle
"""
def show(area): 
    cv.Rectangle(img,(area[0][0],area[0][1]),
                     (area[0][0]+area[0][2],area[0][1]+area[0][3]),
                    (255,0,0),2)
    cv.NamedWindow('Face Detection', cv.CV_WINDOW_NORMAL)
    cv.ShowImage('Face Detection', img) 
    cv.WaitKey()

"""
given an area to be cropped, crop() returns a cropped image
"""
def crop(area): 
    crop = img[area[0][1]:area[0][1] + area[0][3], area[0][0]:area[0][0]+area[0][2]] #img[y: y + h, x: x + w]
    return crop

"""
given a jpg image, vectorize the grayscale pixels to 
a (width * height, 1) np array
it is used to preprocess the data and transform it to feature space
"""
def vectorize(filename):
    size = WIDTH, HEIGHT # (width, height)
    im = Image.open(filename) 
    resized_im = im.resize(size, Image.ANTIALIAS) # resize image
    im_grey = resized_im.convert('L') # convert the image to *greyscale*
    im_array = np.array(im_grey) # convert to np array
    oned_array = im_array.reshape(1, size[0] * size[1])
    return oned_array

"""
returns -1 on error. 
returns 0 for no smile
returns 1 for smile
"""
def is_smiling(lr, image):
    mouth_image_name = 'temp-mouth.jpg'
    mouth = m.findmouth(image)
    # show(mouth)
    if mouth != 2: # did not return error
        mouthimg = crop(mouth)
        cv.SaveImage(mouth_image_name, mouthimg)
        # predict the captured emotion
        result = lr.predict(vectorize(mouth_image_name))
        if result == 1:
            return 1
        else:
            return 0
    else:
        return -1;

"""
load training data
"""
def train_recognition():
    # create a list for filenames of smiles pictures
    smilefiles = []
    with open('smiles.csv', 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='\t'):
            smilefiles += rec

    # create a list for filenames of neutral pictures
    neutralfiles = []
    with open('neutral.csv', 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='\t'):
            neutralfiles += rec

    # N x dim matrix to store the vectorized data (aka feature space)       
    phi = np.zeros((len(smilefiles) + len(neutralfiles), dim))
    # 1 x N vector to store binary labels of the data: 1 for smile and 0 for neutral
    labels = []

    # load smile data
    PATH = "../data/smile/"
    for idx, filename in enumerate(smilefiles):
        phi[idx] = vectorize(PATH + filename)
        labels.append(1)

    # load neutral data    
    PATH = "../data/neutral/"
    offset = idx + 1
    for idx, filename in enumerate(neutralfiles):
        phi[idx + offset] = vectorize(PATH + filename)
        labels.append(0)

    """
    training the data with logistic regression
    """
    lr = logistic.Logistic(dim)
    lr.train(phi, labels)

    return lr



if __name__ == '__main__':
    lr = train_recognition()
    

    """
    open webcam and capture images
    """
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    print "\n\n\n\n\npress space to take picture; press ESC to exit"

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(40)
        if key == 27: # exit on ESC
            break
        if key == 32: # press space to save images
            cv.SaveImage("webcam.jpg", cv.fromarray(frame))
            img = cv.LoadImage("webcam.jpg") # input image
            
            res = is_smiling(lr, img)
            if (res == -1):
                print "failed to detect mouth. Try hold your head straight and make sure there is only one face."
            elif (res == 0):
                print "FROWN :("
            else:
                print "SMILE :)"
    
    cv2.destroyWindow("preview")
