A convolutional neural network built for UC Berkeley SETI

Input: .fits files of dimension 16x512, representing 512 radio frequency bands sampled 16 times in sequence. These files are the result of breaking up much larger .fil files representing radio signals with high resolution in frequency, low resolution in time. 
Output: Labels 0 or 1, representing noise and signal. If signal, it's passed on to other classifiers to determine if it is potentially arising from ET. 

Relies on data being labeled in a two column table called labels.csv
Consists of three convolutional layers, 3 pooling layers, 2 fully connected layers. Adding more layers does not appear to improve accuracy. 
Writes the images corresponding to misclassified signals to a folder, hopefully allowing us to understand what type of signal/noise our network has trouble classifying. 
Use tensorboard to visualize loss and accuracy with training epoch.
Test accuracy is currently around .996
