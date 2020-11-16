"""
                Image utils

compress_nadia - ready
compress_zlib - ready
plot_numpy2D - ready
convert_numpy_to_binary - ready
convert_difftxt_to_numpy - ready
addNoiseCorrect - ready
agu_compress_color - ready
get_compressed_image - ready
convert2D2matlab - ready


Version: 2.0
Status: ready
2020/11/10
By Krivenko S.S.
"""
import numpy as np
import pandas as pd
import os
import cv2
import subprocess
from sklearn.metrics import mean_squared_error
import matlab
import psnrhvsm as p
from matplotlib import pyplot as plt
import zlib

# -------------------------------- [ compress_nadia ] -------------------------------- #
def compress_nadia(path_to_coder, fname, zip_name):
    """
    Compress binary using compr05.exe

    Args:        
        fname: source binary filename
        zip_name: compressed filename

    Returns:
        fname_size: size of source binary filename
        zip_name_size: size of compressed filename
        CR: compression ratio
    """
    full_path_fname = os.path.join(os.getcwd(), fname)
    full_path_zip_name = os.path.join(os.getcwd(), zip_name)
    full_path_coder = path_to_coder + 'compr05.exe'

    argsLine = [full_path_fname + ' ' + full_path_zip_name]
    cmd = [full_path_coder, argsLine]
    subprocess.check_call(cmd)    

    fname_size = os.path.getsize(fname)
    zip_name_size = os.path.getsize(zip_name)
    
    os.remove(zip_name)

    return fname_size, zip_name_size, fname_size / zip_name_size
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ compress_zlib ] -------------------------------- #
def compress_zlib(fname, zip_name):
    """
    Compress binary using ZLIB

    Args:        
        fname: source binary filename
        zip_name: compressed filename

    Returns:
        fname_size: size of source binary filename
        zip_name_size: size of compressed filename
        CR: compression ratio
    """
    str_object1 = open(fname, 'rb').read()
    str_object2 = zlib.compress(str_object1, 9)
    f = open(zip_name, 'wb')
    f.write(str_object2)
    f.close()
    fname_size = os.path.getsize(fname)
    zip_name_size = os.path.getsize(zip_name)
    
    os.remove(zip_name)
    return fname_size, zip_name_size, fname_size / zip_name_size
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ plot_numpy2D ] -------------------------------- #
def plot_numpy2D(inarray, pedestal=127):
    """
    Show binary image

    Args:        
        inarray: 2D array
        pedestal: pedestal for visual acceptance

    Returns:

    """        
    plt.imshow(inarray + pedestal, cmap='gray')
    plt.show()
    #cv2.imshow('image', inarray + pedestal)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()    
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ convert_numpy_to_binary ] -------------------------------- #
def convert_numpy_to_binary(inarray, datatype, bin_name):
    """
    Convert numpy 2D array to binary

    Args:        
        txt_name: full path to difference TXT

    Returns:
        out: output 2D numpy array
    """    
    inarray.astype(datatype).tofile(bin_name)    
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ convert_difftxt_to_numpy ] -------------------------------- #
def convert_difftxt_to_numpy(txt_name):
    """
    Convert difference TXT to numpy 2D array

    Args:        
        txt_name: full path to difference TXT

    Returns:
        out: output 2D numpy array
    """
    out = np.loadtxt(txt_name)
    return out
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ get_percent_SPIHT_ADCT ] -------------------------------- #
def get_percent_SPIHT_ADCT(folder, csv_name):
    """
        Analisys of benefit percents

    Args:        
        folder: folder with result

    Returns:
        outlist: output list of lists with data
    """    
    csvs = os.listdir(folder)
    for csvf in csvs:        
        if csv_name in csvf and 'ADCT' in csvf:
            result = np.loadtxt(os.path.normpath(os.path.join(folder, csvf)), delimiter=',')
            dfa = pd.DataFrame(result, columns=['mse', 'P0', 'MSEQS212', 'CR', 'QS', 'std', 'P1',
                'k', 'psnrhvsm', 'psnrhvs', 'mse_hvs_m', 'mse_hvs', 'psnr', 
                'MSEhvsmQS212', 'MSEhvsQS212', 'fsim'])            

        if csv_name in csvf and 'SPIHT' in csvf:
            result = np.loadtxt(os.path.normpath(os.path.join(folder, csvf)), delimiter=',')
            dfs = pd.DataFrame(result, columns=['mse', 'P0', 'MSEQS212', 'CR', 'QS', 'std', 'P1',
                'k', 'psnrhvsm', 'psnrhvs', 'mse_hvs_m', 'mse_hvs', 'psnr', 
                'MSEhvsmQS212', 'MSEhvsQS212', 'fsim'])

    psnrhvsm = 40
    df_sort = dfa.iloc[(dfa['psnrhvsm'] - psnrhvsm).abs().argsort()[:1]]
    dfa_cr = float(df_sort['CR'])
    df_sort = dfs.iloc[(dfs['psnrhvsm'] - psnrhvsm).abs().argsort()[:1]]
    dfs_cr = float(df_sort['CR'])

    percent = 100 * (dfa_cr - dfs_cr) / dfa_cr  
    return percent
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ addNoiseCorrect ] -------------------------------- #
def addNoiseCorrect(imageArray, iMean, iStd, datatype):
    """
    AGU-based compress for color image    
    """
    noiseArray = np.random.normal(iMean, iStd, imageArray.shape)
    outArray = imageArray + noiseArray
    minValue = np.iinfo(datatype).min
    maxValue = np.iinfo(datatype).max
    lessminIdx = np.nonzero(outArray < minValue)
    abovemaxIdx = np.nonzero(outArray > maxValue)
    outArray[lessminIdx] = minValue
    outArray[abovemaxIdx] = maxValue
    outArray = outArray.astype(datatype)
    return outArray
# -------------------------------- [  ] -------------------------------- #

# -------------------------------- [ agu_compress_color ] -------------------------------- #
def agu_compress_color(imageName, resFolder_path, QS, iMean, iStd, basis, outsize):
    """
    AGU-based compress for color image
    Example: agu_compress_color('baboon.png', 'd:\\temp\\', 12, 0, 0, 'YCrCb', (512, 512))

    Args:
        imageName: name of image
        resFolder_path: path to folder with images
        QS: quantization step
        iMean: mean values of noise (0 if noise free)
        iStd: std values of noise (0 if noise free)
        basis: 'RGB' or 'YCrCb'
        outsize: size of output image  -  (512, 512)

    Returns:
        result: dict with CR and output sizes
    """        

    rawExt = ".raw"
    coderExt = ".agu"
    outExt = ".out"
    space = " "
    temp0name = 'c0'
    temp1name = 'c1'
    temp2name = 'c2'

    inname = str(imageName + '_original.png')
    outname = str(imageName + '_compress.png')

    # read original color file
    FileName = os.path.normpath(os.path.join(resFolder_path, imageName))
    imageArray = cv2.imread((FileName), 1)  # blue green red
    if basis == 'RGB':
        channel0 = imageArray[:, :, 0]  # 1152x1500
        channel1 = imageArray[:, :, 1]  # 1152x1500
        channel2 = imageArray[:, :, 2]  # 1152x1500
        channel0 = cv2.resize(channel0, outsize, interpolation=cv2.INTER_CUBIC)
        channel1 = cv2.resize(channel1, outsize, interpolation=cv2.INTER_CUBIC)
        channel2 = cv2.resize(channel2, outsize, interpolation=cv2.INTER_CUBIC)
        inColorArray = cv2.resize(imageArray, outsize, interpolation=cv2.INTER_CUBIC)
    elif basis == 'YCrCb':
        imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2YCR_CB)  # Y Cr Cb
        channel0 = imageArray[:, :, 0]  # 1152x1500
        channel1 = imageArray[:, :, 1]  # 1152x1500
        channel2 = imageArray[:, :, 2]  # 1152x1500
        channel0 = cv2.resize(channel0, outsize, interpolation=cv2.INTER_CUBIC)
        channel1 = cv2.resize(channel1, outsize, interpolation=cv2.INTER_CUBIC)
        channel2 = cv2.resize(channel2, outsize, interpolation=cv2.INTER_CUBIC)
        inColorArray = cv2.resize(cv2.cvtColor(imageArray, cv2.COLOR_YCrCb2BGR), outsize, interpolation=cv2.INTER_CUBIC)

    #plt.imshow(channel0, cmap='gray')
    #plt.imshow(channel1, cmap='gray')
    #plt.imshow(channel2, cmap='gray')


    cv2.imwrite(inname, inColorArray, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    result = dict()
    #plt.imshow(imageArray, cmap='gray')

    # if noise needs
    if iMean != 0 or iStd != 0:
        channel0 = addNoiseCorrect(channel0, iMean, iStd, np.uint8)
        channel1 = addNoiseCorrect(channel1, iMean, iStd, np.uint8)
        channel2 = addNoiseCorrect(channel2, iMean, iStd, np.uint8)

    # convert to raw
    bmp2raw(channel0, str(temp0name) + rawExt)
    bmp2raw(channel1, str(temp1name) + rawExt)
    bmp2raw(channel2, str(temp2name) + rawExt)

    # compressing chan#0
    argsLine = ["e " + str(temp0name) + rawExt + space + str(temp0name) + coderExt + space + str(QS)]
    cmd = ["AGU", argsLine]
    subprocess.check_call(cmd)
    # decompressing
    argsLine = ["d " + str(temp0name) + coderExt + space + str(temp0name) + outExt]
    cmd = ["AGU", argsLine]
    subprocess.check_call(cmd)
    # raw -> image array
    cimageArray0 = raw2bmp(str(temp0name) + outExt)

    # compressing chan#1
    argsLine = ["e " + str(temp1name) + rawExt + space + str(temp1name) + coderExt + space + str(QS)]
    cmd = ["AGU", argsLine]
    subprocess.check_call(cmd)
    # decompressing
    argsLine = ["d " + str(temp1name) + coderExt + space + str(temp1name) + outExt]
    cmd = ["AGU", argsLine]
    subprocess.check_call(cmd)
    # raw -> image array
    cimageArray1 = raw2bmp(str(temp1name) + outExt)

    # compressing chan#2
    argsLine = ["e " + str(temp2name) + rawExt + space + str(temp2name) + coderExt + space + str(QS)]
    cmd = ["AGU", argsLine]
    subprocess.check_call(cmd)
    # decompressing
    argsLine = ["d " + str(temp2name) + coderExt + space + str(temp2name) + outExt]
    cmd = ["AGU", argsLine]
    subprocess.check_call(cmd)
    # raw -> image array
    cimageArray2 = raw2bmp(str(temp2name) + outExt)

    #plt.imshow(cimageArray, cmap='gray')
    #plt.imsave('cimageArray', cimageArray, cmap='gray')
    #plt.imsave('ccimageArray.png', imageArray - cimageArray + 128, cmap='gray')

    #p_hvs_m, p_hvs, mse_hvs_m, mse_hvs = p.psnrhvsm(channel0, cimageArray0)

    # check sizes
    sourceSize0 = os.path.getsize(str(temp0name) + rawExt)
    compressSize0 = os.path.getsize(str(temp0name) + coderExt)
    sourceSize1 = os.path.getsize(str(temp1name) + rawExt)
    compressSize1 = os.path.getsize(str(temp1name) + coderExt)
    sourceSize2 = os.path.getsize(str(temp2name) + rawExt)
    compressSize2 = os.path.getsize(str(temp2name) + coderExt)

    CR0 = sourceSize0 / compressSize0
    CR1 = sourceSize1 / compressSize1
    CR2 = sourceSize2 / compressSize2

    result['sourceSize'] = sourceSize0 + sourceSize1 + sourceSize2
    result['compressSize'] = compressSize0 + compressSize1 + compressSize2
    result['CR'] = result['sourceSize']  / result['compressSize']

    outColorArray = np.zeros((outsize[0], outsize[1], 3), dtype=np.uint8)
    outColorArray[:, :, 0] = cimageArray0
    outColorArray[:, :, 1] = cimageArray1
    outColorArray[:, :, 2] = cimageArray2

    if basis == 'YCrCb':
        outColorArray = cv2.cvtColor(outColorArray, cv2.COLOR_YCrCb2BGR)  # YCrCb to BGR

    cv2.imwrite(outname, outColorArray, [cv2.IMWRITE_PNG_COMPRESSION, 0])


    #cv2.imshow('image', cimageArray) cv2.COLOR_BGR2YCR_CB
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #plt.imshow(imageArray - cimageArray + 128, cmap='gray')
    #plt.figure(1)
    #plt.subplot(221)
    #plt.imshow(imageArray)
    #plt.subplot(222)
    #plt.imshow(cimageArray)
    #plt.show()

    return result
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ get_compressed_image ] -------------------------------- #
def get_compressed_image(path_to_image, QS=13, coder='ADCT', fsim=False):
    """
    Compress image and get its PNG version

    Args:
        path_to_image: path to image folder
        QS: quantization step
        coder: required coder (AGU, ADCT, AGUm, ADTCm)

    Returns:
        aqs: average QS
    """    
    tempImagename = "temp"
    rawExt = ".raw"
    coderExt = ".cod"
    outExt = ".out"
    space = " "

    csvs = os.listdir(path_to_image)    
    for csvf in csvs:
        FileName = os.path.normpath(os.path.join(path_to_image, csvf))
        imageArray = cv2.imread((FileName), 0)

        # convert to raw
        bmp2raw(imageArray, str(tempImagename) + rawExt)    

        # compressing
        argsLine = ["e " + str(tempImagename) + rawExt + space + str(tempImagename) + coderExt + space + str(QS)]
        cmd = [coder, argsLine]
        subprocess.check_call(cmd)

        # decompressing
        argsLine = ["d " + str(tempImagename) + coderExt + space + str(tempImagename) + outExt]
        cmd = [coder, argsLine]
        subprocess.check_call(cmd)    

        # raw -> image array
        cimageArray = raw2bmp(str(tempImagename) + outExt)

        # save to PNG with no compression
        png_name = csvf[:-4] + '_qs_' + str(QS) + '.png'
        cv2.imwrite(png_name, cimageArray, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # save diff to PNG with no compression
        png_name = csvf[:-4] + '_qs_' + str(QS) + '_diff.png'
        cv2.imwrite(png_name, (imageArray - cimageArray) + 128, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # check sizes
        sourceSize = os.path.getsize(str(tempImagename) + rawExt)
        compressSize = os.path.getsize(str(tempImagename) + coderExt)

        # calc metrics
        psnrhvsm, psnrhvs, mse_hvs_m, mse_hvs = p.psnrhvsm(imageArray, cimageArray)
        mse = mean_squared_error(np.float32(imageArray), np.float32(cimageArray))
        psnr = 10 * np.log10(255 * 255 / mse)        
        metrics = dict()
        metrics['mse'] = mse
        metrics['psnr'] = psnr
        metrics['mse_hvs'] = mse_hvs
        metrics['psnrhvs'] = psnrhvs
        metrics['mse_hvs_m'] = mse_hvs_m
        metrics['psnrhvsm'] = psnrhvsm        
        metrics['cr'] = np.divide(sourceSize, compressSize)

        # matlab engine
        if fsim:
            mengine = matlab.engine.start_matlab()
            mengine.cd('d:\\Krivenko\\Work\\MATLAB\\2020\\New20200428\\')
            a = convert2D2matlab(imageArray)
            b = convert2D2matlab(cimageArray)
            fsim = mengine.fsim(a, b)
            mengine.quit()
            metrics['fsim'] = fsim
        else:
            metrics['fsim'] = -1            

        # save metrics to txt
        txt_name = csvf[:-4] + '_qs_' + str(QS) + '.txt'
        with open(txt_name, 'w') as file:
            print(metrics, file=file)
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ convert2D2matlab ] -------------------------------- #
def convert2D2matlab(array2D):
    """
        Convert 2D array to Matlab compatiable data

    Args:        
        array2D: 2D ndarray

    Returns:
        outlist: output list of lists with data
    """    
    outlist = list()
    for i in range(array2D.shape[0]):
        outlist.append(list(array2D[i, :]))
    outlist = matlab.uint8(outlist)
    return outlist
# -------------------------------- [ ] -------------------------------- #

# -------------------------------- [ bmp2raw ] -------------------------------- #
"""

"""
def bmp2raw(imageArray, rawFilename):

    N, M = np.shape(imageArray)
    rawArray = np.reshape(imageArray, (N * M, 1), order = 'C')
    rawArray.tofile(rawFilename)
# -------------------------------- [ bmp2raw ] -------------------------------- #

# -------------------------------- [ raw2bmp ] -------------------------------- #
"""

"""
def raw2bmp(rawFilename):
    dt = np.dtype(np.uint8)
    imageArray = np.fromfile(rawFilename, dtype = dt)
    Nbig = np.shape(imageArray)[0]
    N = int(np.sqrt(Nbig))
    imageArray = np.reshape(imageArray, (N, N), order = 'C')
    return imageArray
# -------------------------------- [ raw2bmp ] -------------------------------- #