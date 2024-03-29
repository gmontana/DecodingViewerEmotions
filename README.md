# Decoding Viewer Emotions in Video Ads: Predictive Insights through Deep Learning

This repository offers access to the code and data necessary to replicate the findings of "Decoding Viewer Emotions in Video Ads: Predictive Insights through Deep Learning" by Alexey Antonov, Shravan Sampath Kumar, Jiefei Wei, William Headley, Orlando Wood, and Giovanni Montana.

## Paper Summary

Our study introduces a novel deep learning framework capable of predicting viewers' emotional reactions to video advertisements using short, 5-second excerpts. Leveraging a dataset derived from System1’s proprietary methodologies, encompassing over 30,000 video ads annotated by around 75 viewers each, our methodology integrates convolutional neural networks to process both video and audio data, achieving notable accuracy in identifying salient emotional excerpts.

## Repository Contents

- Python code for training the Temporal Shift Module (TSM) augmented neural network models detailed in the paper.
- A dataset consisting of 5-second video excerpts, totaling over 30,000 clips, utilized for training, validation, and testing, annotated for eight distinct emotional categories and their temporal onset.
- Pre-trained model weights facilitating the reproduction of reported experimental outcomes.

Given the dataset's substantial volume, both video excerpts and model weights are hosted externally. To access them for research purposes, contact Giovanni Montana at g.montana@warwick.ac.uk with your affiliation details.

## Dataset Access Disclaimer

The dataset leverages System1's proprietary "Test Your Ad®" tool for public, educational, and illustrative use. The advertisements and excerpts, while derived from System1's tool, remain the property of their original owners. Usage beyond this study's scope requires explicit permission from those owners. By accessing the dataset, you agree to these conditions.

## Using the Code

The included Python code, leveraging the PyTorch framework, is well-documented and user-friendly. It allows for the reproduction of the paper's experiments or adaptation for your datasets.

In addition to training the TSM-augmented architectures described in our paper, this code also supports inference tasks.

Should you encounter issues or have questions, please open a GitHub issue on this repository.
