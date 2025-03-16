# Classification of AFM Images of Biofilm

## The Classification Tool
Users seeking to use our model to classify their biofilm images can use the
Biofilm Classification Tool. This can be found on the [Releases](https://github.com/Ram9bo/image-classification/releases) page. 

## The Code
The code in this repo is meant as a reference for the conducted study. Technically proficient users can use it
to train new versions of the model on their own data, but we would encourage researchers to build new models
that fit their datasets and the needs of their study.

Users that do want to use the code to train a model on their own data should modify the BASE_DATA_DIR
in 'dataloader.py' to point to a directory with a matching structure.

### Running from source
Download or clone the repository to your machine. Install the required dependencies. Use 'run.py' as entry point
to the system.

### Installing Dependencies
Python 3.10.11 \
pip install -r requirements.txt
