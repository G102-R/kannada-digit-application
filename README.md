# kannada-digit-application

### This project is similar to the MNIST image classification project but with handwritten Kannada numerals. The dataset has been made possible by Vinay Prabhu and can be accessed on Kaggle. This image dataset consists of 70,000 images for the training set and 5000 images for the test set. The model chosen for this multiclass classification is a CNN (convolutional neural network) and the goal is to accurately classify the images into one of ten classes/labels.

## Project Description
### Handwritten numeral images have proven to be great baseline models for image classification and this project adds to the already rich repertoire of robust image classifiers. The performance of a classifier can be measured using several metrics depending on how the dataset is structured. This dataset is a balanced dataset, meaning, there are equal number of images for all ten labels and hence metrics such as accuracy, precision, recall can be used to measure model performance. The image below provides a high level view of how the model classifies an image into one of ten labels.

## GUI : Streamlit
### Model performance verification using a realtime, interactive interface to make live predictions. The Streamlit sketchpad takes in hand drawn numerals and diplays the prediction in a sperate window. A numeral is drawn and the model predicts the label with the accuracy displayed.

### This project was an inspiration from the hugely popular MNIST dataset for handwritten digit classification.
