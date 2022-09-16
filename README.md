# InterviewData
Sandbox for algorithms and data science interviews

## Task

Demonstrate training and testing a binary classifier on the example data

## Details

* The load_data.py script demos how to load the sample data and visualize images

* Sample data includes two datasets

    * Each set comprizes a compressed numpy array with image stack and metadata, as well as a separate label csv

    * The image stack `img_stack` is a WxMxNx8 array (W sequential frames of MxN images, 8 channels)

    * The 8 channel labels are in `img_channels`:
        * The first 4 are "raw" data from the image sensor
        * The 3 "od" channels are the result of our processing algorithm
        * OD_M is correlated with the amount of gas absorption predicted, if we attribute the signal to gas
        * The mask channel is another output of the processing algorithm, showing what pixels were predicted to contain gas
        * The mask channel is mostly included for visualization of the existing detection algorithm, not as a feature to train on

    * The image stack is accompanied by a list of timestamps for all W frames

    * The label csv associates a label with each of the W timestamps. Ignore the "prediction" field. These are binary labels for each frame indicating the presence of gas anywhere in frame.

    * One of the datasets is mostly "True" labels, the other is all "False" labels. They will have to be merged to be used for training 

* The demo script shows 3 visualizations of a single image frame, for context
    * Basic plot of one of the raw channels
    * False color representation of the 3 OD channels. In this case, OD_M is used for the green channel, and allows us to identify gas visually with more accuracy than the existing detection algorithm. Human reviewers use this view to label data.
    * Gas overlay view which uses the mask layer to demonstrate the existing detection algorithm. Gives an idea of the final goal.




