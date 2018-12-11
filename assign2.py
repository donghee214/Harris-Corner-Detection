################################################################################
# COMP3317 Computer Vision
# Assignment 2 - Conrner detection
################################################################################
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve1d

################################################################################
#  perform RGB to grayscale conversion
################################################################################
def rgb2gray(img_color) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    # return:
    #    img_gray - a h x w numpy ndarray (dtype = np.float64) holding
    #               the grayscale image

    # print('img start', img_color)
    # print('img end')
    rows = img_color.shape[0]
    cols = img_color.shape[1]

    newImg = np.array(np.zeros((rows, cols)), dtype = np.float64)
    for x in range(0, rows):
        for y in range(0, cols):
            colors = img_color[x, y]
            newImg[x, y] = 0.299*colors[0] + 0.587*colors[1] + 0.114*colors[2]



    return newImg

################################################################################
#  perform 1D smoothing using a 1D horizontal Gaussian filter
################################################################################
def smooth1D(img, sigma) :
    # input :
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the 1D Gaussian function
    # return:
    #    img_smoothed - a h x w numpy ndarry holding the 1D smoothing result


    # TODO: form a 1D horizontal Guassian filter of an appropriate size
    n = 3
    filter = []
    normalization = 0
    for x in range(-int(n/2), int(n/2)+1):
        val = 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-float(x)**2/(2*sigma**2))
        normalization += val
        filter.append(val)

    for i in range(0, len(filter)):
        filter[i] = filter[i]/normalization

    # TODO: convolve the 1D filter with the image;
    rows = img.shape[0]
    cols = img.shape[1]
    newImg = np.array(np.zeros((rows, cols)), dtype=np.float64)
    for x in range(0, rows):
        for y in range(0, cols):
            if y == 0:
                newImg[x, y] = img[x, y]*filter[0] + img[x, y]*filter[1] + img[x, y+1]*filter[2]
            elif y == cols - 1:
                newImg[x, y] = img[x, y-1]*filter[0] + img[x, y]*filter[1] + img[x, y]*filter[2]
            else:
                newImg[x, y] = img[x, y-1]*filter[0] + img[x, y]*filter[1] + img[x, y+1]*filter[2]

    return newImg

################################################################################
#  perform 2D smoothing using 1D convolutions
################################################################################
def smooth2D(img, sigma) :
    # input:
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the Gaussian function
    # return:
    #    img_smoothed - a h x w numpy array holding the 2D smoothing result

    # TODO: smooth the image along the vertical direction
    img = smooth1D(img, sigma)
    img = np.transpose(img)
    img = smooth1D(img, sigma)
    img = np.transpose(img)
    # TODO: smooth the image along the horizontal direction

    return img

################################################################################
#   perform Harris corner detection
################################################################################
def harris(img, sigma, threshold) :
    # input:
    #    img - a h x w numpy ndarry holding the input image
    #    sigma - sigma value of the Gaussian function used in smoothing
    #    threshold - threshold value used for identifying corners
    # return:
    #    corners - a list of tuples (x, y, r) specifying the coordinates
    #              (up to sub-pixel accuracy) and cornerness value of each corner

    # TODO: compute Ix & Iy
    # print('imgage')
    # print(img)
    rows = img.shape[0]
    cols = img.shape[1]


    Ix = np.zeros((rows, cols))

    for row in range(0, rows):
        for col in range(0, cols):
            if col == 0:
                Ix[row, col] = img[row, col+1] - img[row, col]
            elif col == cols - 1:
                Ix[row, col] = img[row, col] - img[row, col - 1]
            else:
                Ix[row, col] = (img[row, col - 1] + img[row, col + 1])/2
    # img = np.transpose(img)

    Iy = np.zeros((rows, cols))

    for row in range(0, rows):
        for col in range(0, cols):
            if row == 0:
                Iy[row, col] = img[row + 1, col] - img[row, col]
            elif row == rows - 1:
                Iy[row, col] = img[row, col] - img[row - 1, col]
            else:
                Iy[row, col] = (img[row - 1, col] + img[row + 1, col])/2
    print('img', img)
    print('Ix', Ix)
    print('Iy', Iy)
    # TODO: compute Ix2, Iy2 and IxIy
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy
    Ix2= gaussian_filter(Ix2, 1)
    Iy2 = gaussian_filter(Iy2, 1)
    IxIy = gaussian_filter(IxIy, 1)
    R = ((Ix2 * Iy2) - (IxIy ** 2)) - 0.04*((Ix2 + Iy2)**2)
    # print(R)
    # TODO: smooth the squared derivatives

    # TODO: compute cornesness functoin R

    # TODO: mark local maxima as corner candidates;
    #       perform quadratic approximation to local corners upto sub-pixel accuracy

    # TODO: perform thresholding and discard weak corners
    cornernessArr = np.zeros((rows, cols))
    corners = []
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            a = (R[row, col - 1] + R[row, col + 1] - (2*R[row, col]))/2
            b = (R[row - 1, col] + R[row + 1, col] - (2*R[row, col]))/2
            c = (R[row, col + 1] - R[row, col - 1])/2
            d = (R[row + 1, col] - R[row - 1, col])/2
            e = R[row, col]
            subPixelX = -c/(2*a)
            subPixelY = -d/(2*b)
            cornernessArr[row, col] = a*subPixelX**2 + b*subPixelY**2 + c*subPixelX + d*subPixelY + e
            if cornernessArr[row, col] > 1*10**7:
                corners.append((col, row, 1))
    # print(cornernessArr)


    return sorted(corners, key = lambda corner : corner[2], reverse = True)

##############################################################################
#   save corners to a file
################################################################################
def save(outputfile, corners) :
    try :
        file = open(outputfile, 'w')
        file.write('%d\n' % len(corners))
        for corner in corners :
            file.write('%.4f %.4f %.4f\n' % corner)
        file.close()
    except :
        print('Error occurs in writting output to \'%s\''  % outputfile)
        sys.exit(1)

################################################################################
#   load corners from a file
################################################################################
def load(inputfile) :
    try :
        file = open(inputfile, 'r')
        line = file.readline()
        nc = int(line.strip())
        print('loading %d corners' % nc)
        corners = list()
        for i in range(nc) :
            line = file.readline()
            (x, y, r) = line.split()
            corners.append((float(x), float(y), float(r)))
        file.close()
        return corners
    except :
        print('Error occurs in writting output to \'%s\''  % outputfile)
        sys.exit(1)

################################################################################
## main
################################################################################
def main() :
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 2')
    parser.add_argument('-i', '--inputfile', type = str, default = 'grid1.jpg', help = 'filename of input image')
    parser.add_argument('-s', '--sigma', type = float, default = 1.0, help = 'sigma value for Gaussain filter')
    parser.add_argument('-t', '--threshold', type = float, default = 1e6, help = 'threshold value for corner detection')
    parser.add_argument('-o', '--outputfile', type = str, help = 'filename for outputting corner detection result')
    args = parser.parse_args()

    print('------------------------------')
    print('COMP3317 Assignment 2')
    print('input file : %s' % args.inputfile)
    print('sigma      : %.2f' % args.sigma)
    print('threshold  : %.2e' % args.threshold)
    print('output file: %s' % args.outputfile)
    print('------------------------------')

    # load the image
    try :
        #img_color = imageio.imread(args.inputfile)
        img_color = plt.imread(args.inputfile)
        print('%s loaded...' % args.inputfile)
    except :
        print('Cannot open \'%s\'.' % args.inputfile)
        sys.exit(1)
    # uncomment the following 2 lines to show the color image
    # plt.imshow(np.uint8(img_color))
    # plt.show()

    # perform RGB to gray conversion
    print('perform RGB to grayscale conversion...')
    img_gray = rgb2gray(img_color)
    # uncomment the following 2 lines to show the grayscale image
    # plt.imshow(np.float32(img_gray), cmap = 'gray')
    # plt.show()

    smooth1D(img_gray, 1)

    # print(img)
    # perform corner detection
    print('perform Harris corner detection...')
    corners = harris(img_gray, args.sigma, args.threshold)

    # plot the corners
    print('%d corners detected...' % len(corners))
    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]
    fig = plt.figure()
    plt.imshow(np.float32(img_gray), cmap = 'gray')
    plt.plot(x, y,'r+',markersize = 5)
    plt.show()

    # save corners to a file
    if args.outputfile :
        save(args.outputfile, corners)
        print('corners saved to \'%s\'...' % args.outputfile)

if __name__ == '__main__':
    main()
