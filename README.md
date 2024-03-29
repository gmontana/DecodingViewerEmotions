# DecodingViewerEmotions

This repository provides access to code and data to reproduce the results from the paper "Predicting Ad Effectiveness from Short Video Excerpts".

## Paper Summary

This paper presents a deep learning approach to predict the effectiveness of video advertisements from short 5-second excerpts. We trained convolutional neural networks on a dataset of over 30,000 short video clips, using effectiveness ratings gathered from real consumer panels as the prediction targets.

Our models are able to predict ad effectiveness from these short excerpts with strong accuracy. This demonstrates the potential to quickly and automatically assess ad creative, enabling faster and lower-cost ad testing. The paper provides details on the model architectures, training procedures, and experimental results.

## Repository Contents

To enable reproducible research, this repository provides:

- Python code to train the neural network models described in the paper
- The dataset of 5-second video excerpts used for model training, validation, and testing (over 30,000 clips in total)
- Trained model weights that reproduce the experimental results reported in the paper

Due to the large size of the datasets, the video clips and model weights are hosted separately by the University of Warwick. To request access for research purposes, please email [g.montana@warwick.ac.uk](mailto:g.montana@warwick.ac.uk) with your name and institutional affiliation.

## Dataset Access Disclaimer

Please note that the excerpts displayed here are sourced from System1's proprietary rating tool, "System1 Test Your Ad®". These excerpts are provided for public access for educational and illustrative purposes only. While these excerpts have been derived from our tool, the advertisements from which they originate are not owned by System1. As such, these advertisements or the excerpts cannot be distributed or used beyond the context of this study without the express consent of their respective owners. By accessing these excerpts, you acknowledge and agree to these terms.

## Using the Code

The Python code included here is documented and should be self-explanatory to run. It uses the PyTorch deep learning framework. See the code files for details on the required dependencies.

With the video clip data and pre-trained model weights, you can use the code to reproduce the experiments described in the paper, or adapt it to train models on your own data.

Feel free to submit a GitHub issue if you have any questions!
