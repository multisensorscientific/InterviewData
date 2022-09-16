#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sept 9 2022

@author: ryan

Copyright (c) 2022 Kuva 
"""

import numpy as np
import cv2

def create_detection_overlay_img(framedata, maxval=0.1):
    """
    Assumes the framedata array is MxNx8, with the 8 channels corresponding to:
    Raw_A, Raw_B, Raw_D, Raw_E, OD_M, OD_Z0, OD_Z1, Mask

    maxval input provides the upper bound on OD_M values for jet colormap. Min is 0.

    Creates a normalized log-space view of the raw swir data, typically used to visualize
    the scene in an 8-bit format. Overlays positive OD_M values on top of the raw swir visualization.
    Include the mask layer to visualize the gas detections made by the baseline Kuva algorithm.
    """

    try:
        assert len(framedata.shape) == 3
        assert framedata.shape[2] == 8
    except:
        print("Error: requires framedata input with 8 channels (Raw_A, Raw_B, Raw_D, Raw_E, OD_M, OD_Z0, OD_Z1, Mask)")
        return

    baseImg = np.copy(framedata[...,:4])
    baseImg = np.clip(baseImg, 0.0, None)
    baseImg = -1 * np.ma.log(np.ma.masked_values(baseImg, 0.0))

    # After clipping negative intensities and converting to negative log space,
    # project raw data to the "zero-sum" plane by substracting the band means
    # This removes most of the intensity component, to normalize across changing
    # illumination levels. The magnitude of the remaining vector (spectral content of the scene)
    # is useful for visualization purposes.
    baseImg = baseImg - np.mean(baseImg, axis=2)[...,np.newaxis]
    baseImg = np.linalg.norm(baseImg, axis=2)

    #Create contrast enhanced grayscale from magnitude image
    baseImg = np.clip(baseImg, 0.0, None)
    if np.max(baseImg) > 0:
        baseImg = baseImg / np.max(baseImg) * 255  # rescale the image
        baseImg = baseImg.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2))
        baseImg = clahe.apply(baseImg)
    else:
        baseImg = np.zeros_like(baseImg, dtype=np.uint8)


    overlayData = np.copy(framedata[...,4])
    maskimg = np.copy(framedata[...,-1]).astype(np.bool) #Mask data packed as float but should be convertible to bool
    overlayData[np.where(~maskimg)] = 0.0

    
    #Rescale the grayscale base img for RGB conversion
    baseImg = np.clip(baseImg, 0.0, None) / np.max(baseImg) * 255  # rescale the image
    baseImg = baseImg.astype(np.uint8)
    baseImg = cv2.cvtColor(baseImg, cv2.COLOR_GRAY2BGR)

    #Prepare the overlay
    #Zero out values below the min value
    ret, threshmat = cv2.threshold(overlayData, 0., 0., cv2.THRESH_TOZERO)
    #Clip values above max threshold - Note, no clipping if data max is below threshold
    ret, threshmat = cv2.threshold(threshmat, maxval, maxval, cv2.THRESH_TRUNC)
    #Scale data to 0-255 range based on the max threshold (not based on range of clipped data)
    if maxval > 0:
        normed = threshmat / maxval
        normed = (normed * 255).astype(dtype=np.uint8)
    else:
        normed = np.zeros_like(threshmat, dtype=np.uint8)

    colorNormed = cv2.applyColorMap(normed, cv2.COLORMAP_JET)

    # Make normed binary for bitwise_and operation
    ret,mask = cv2.threshold(normed, 0, 1, cv2.THRESH_BINARY)

    # Convert to 0/255 for alpha channel
    ret,mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # add alpha channel to dest
    splitImg = cv2.split(colorNormed)
    dest = cv2.merge([splitImg[0],splitImg[1],splitImg[2],mask])

    #Apply the overlay to the reference image
    # set values of dest to MatC values
    dest[:,:,0] = np.where(dest[:,:,3]==0, baseImg[:,:,0], dest[:,:,0])
    dest[:,:,1] = np.where(dest[:,:,3]==0, baseImg[:,:,1], dest[:,:,1])
    dest[:,:,2] = np.where(dest[:,:,3]==0, baseImg[:,:,2], dest[:,:,2])

    # convert dest to BGR
    dest = cv2.cvtColor(dest, cv2.COLOR_BGRA2BGR)

    # perform overlay
    ALPHA = 0.7
    baseImg = cv2.addWeighted(baseImg, 1-ALPHA, dest, ALPHA, 0, dtype=cv2.CV_8U)

    # convert back to RGB
    baseImg = cv2.cvtColor(baseImg, cv2.COLOR_BGR2RGB)

    return baseImg

def create_colorized_od_img(odimg, clip_negative_od=False):
    """
    Assumes the odimg array is MxNx3, with the 3 channels OD_M, OD_Z0, OD_Z1

    Creates an 8-bit RGB image for visualization, with OD_M (gas signal) as green channel
    (Z0=red, Z1=blue)

    Handles negative OD values by either clipping or flipping (absolute value)
    Clipping allows for a higher contrast image but using the absolute value to
    bring the negative values into play will accentuate the difference
    between the spectral content of noise and signal
    """

    try:
        assert len(odimg.shape) == 3
        assert odimg.shape[2] == 3
    except:
        print("Error: requires input with 3 or more channels")
        return

    img = np.copy(odimg)

    if clip_negative_od:
        alpha = 0.01
        img = np.clip(img, 0.0, None)
    else:
        alpha = 0.05
        img = np.abs(img)

    img = img[:,:,[1,0,2]] #R=Z0, G=Gas(M), B=Z1

    #Get scaling range based on img, without including values of exactly 0.0,
    # which often represent OD values masked out due to low light levels
    all_vals = np.sort(img.ravel())
    all_vals = all_vals[all_vals != 0]
    n = all_vals.size
    if n > 0:
        min_idx = int(np.round(alpha*n))

        #subtract 1 from index to prevent idx=n
        max_idx = int(np.round((1-alpha)*n))-1

        #Make sure max idx does not drop below min idx because of the -1 index
        max_idx = max(max_idx,min_idx+1)

        min_val = all_vals[min_idx]
        max_val = all_vals[max_idx]

        #Scale based on that scaling range
        img = (img-min_val)/(max_val-min_val)*255
        img = np.clip(img,0,255).astype(np.uint8)
    else:
        #Image was all zero - clip as is and leave zeros
        img = np.clip(img,0,255).astype(np.uint8)

    return img