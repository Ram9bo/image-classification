# User Manual
This document provides instructions and information for users of the Biofilm Classification Tool.
The purpose of the tool is to classify AFM images of bacterial biofilm.

The development of the classification scheme and the AI model is described in our paper. 
<br>
van Dun et al., _Machine learning assisted classification of staphylococcal biofilm maturity_, Biofilm,
2025, https://doi.org/10.1016/j.bioflm.2025.100283

## Quickstart
To use the tool, the following three steps are needed
1. Place the images to be classified in the 'images' folder
2. Run the executable 'classify.exe'
3. View the results in the 'results.csv' file

## Input Images
The provided model was trained on a set of images obtained using Atomic Force Microscopy. 
The procedure and exact settings for the AFM machine are described in our paper.
Of particular relevance is the surface area of the biofilm that was imaged. The set of images used each show a 5x5 μm surface.
Attempting to classify images of larger or smaller surfaces will lead to uncertain results, but are almost guaranteed to be classified with less
accuracy than images of a matching surface area.

## The Model
The provided model is one of the 50 that were trained in the final experiment of the study. 
The average accuracy of all models in this experiment is 66%. 
The provided model is one of the best from this experiment and scored around 80% accuracy.
Due to the limited size of the data set, however, the model may not be as good at classifying
arbitrary images that the user provides. As the paper states, the model should be considered a proof-of-concept 
and not be relied on without any human oversight.

## Runtime
Running the executable file will open a command window in which the application will log its progress.
Starting up the application can be expected to take between 20 and 90 seconds, depending on the specifications of the machine.
Classification of the images itself will then proceed automatically, and should only take a few seconds.
Processing a large batch of images at once is much faster than running the tool multiple times for individual images.

## Results
The results of the classification will be recorded in the 'results.csv' file. 
This file will be **overwritten** each time the tool is used, so the user should copy the results if required.
The CSV file will contain two columns, "File Name" and "Class Label".
Each row corresponds to one classified image.

## Test Images
The distribution of this tool comes with a handful of images from the original data set, located in the 'images' folder.
These can be used to test the application and its runtime performance.
