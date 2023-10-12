# Supplementary code for "Noise Learning of Instruments for High-contrast, High-resolution and Fast Hyperspectral Microscopy and Nanoscopy"

**Noise-learning** is a python package containing code for denoising micro- and nano- Raman spectra. 
*This repository is still under construction.*
1. System requirements
2. Installation guide
3. Demo
4. Instructions for use

## 1. System requirements
**Software:**

-  Python = 3.8.13
-  PyCharm Community = 2022.1.3 

**Python packages:**
-  PyTorch = 1.5
-  numpy
-  scipy
-  matplotlib.pyplot
-  math
-  random
-  os

## 2. Installation guide
We recommend the users to install the python environment and relevant python packages via [Anaconda3](https://www.anaconda.com/download). 

The typical install time should be around one hour.

After setup the running environment, download and unzip the code.

## 3. Demo
We provide  Raman spectra dataset and pretrained AUnet model for model inferencing.

Download the pretrained checkpoint model, and the Raman spectra dataset, then run the **Main.py** in PyCharm, you will obtain the denoised Raman spetra. 
The default output file type is *.mat that can be directly loaded in Matlab for visualization. You can also save the results as .txt files, and plot the curves in matplotlib.pyplot, or Origin to check the spectra before and after denoising.

## 4. Instructions for use

