#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 2022

@author: ryan

Copyright (c) 2022 Kuva 
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import data_utils as utils


def load_img_data(filename):
    loaded = np.load(filename, allow_pickle=True)
    return loaded['img_stack'], loaded['img_channels'], loaded['frame_times']


if __name__ == '__main__':

    try:
        file = sys.argv[1]
    except:
        file = None

    # Default to test data
    if file is None:
        file = "data_samples/2c559191_img.npz"

    key_frame_idx = 6

    try:
        assert(file is not None)
        assert(len(file)>0)
        assert(os.path.exists(file))
    except:
        print("Error: need to specify a valid (.npz) image data file")
        sys.exit(1)

    stack, chans, times = load_img_data(file)

    try:
        assert(stack is not None)
        assert(chans is not None)
        assert(times is not None)
    except:
        print("Error: problem loading data")
        sys.exit(1)

    print("Data contains channels: {}".format(chans))

    #View one of the raw channels for the first frame
    plt.figure()
    plt.imshow(stack[key_frame_idx,:,:,0], vmin=0.)
    plt.suptitle(times[key_frame_idx])

    c_od = utils.create_colorized_od_img(stack[key_frame_idx,:,:,4:7])

    #View the colorized OD for the first frame
    plt.figure()
    plt.imshow(c_od)
    plt.suptitle(times[key_frame_idx])

    gas_overlay = utils.create_detection_overlay_img(stack[key_frame_idx,...])

    plt.figure()
    plt.imshow(gas_overlay)
    plt.suptitle(times[key_frame_idx])


    plt.show()