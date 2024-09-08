# ASL Translator using CNN-LSTM

This project implements an American Sign Language (ASL) video-to-text translator using CNN-LSTM architectures for processing and interpreting sign language gestures. The model processes video data using `MediaPipe` for landmark extraction and uses a deep learning architecture to predict the corresponding ASL word.

## Project Overview

The pipeline involves:
1. **Data Processing**: Extracting landmarks from videos using MediaPipe.
2. **Data Preparation**: Converting video frames to matrices and preparing the dataset.
3. **Modeling**: Training CNN-LSTM models to classify and predict ASL signs.
4. **Evaluation and Prediction**: Evaluating model performance on test data and making predictions on unseen videos.

## Requirements

To run this notebook, you'll need the following Python packages:

- `mediapipe`
- `tensorflow`
- `opencv-python`
- `matplotlib`
- `scikit-learn`

You can install these dependencies using the following command:

```bash```
!pip install mediapipe tensorflow transformers opencv-python matplotlib scikit-learn


## Model Architecture

This project uses two CNN-LSTM architectures:

    CNN-LSTM Model A: Uses a sequential stack of CNN layers followed by LSTM layers to extract temporal information from video frames.
    CNN-LSTM Model B: A more complex architecture with additional layers for feature extraction and better generalization.

## How to Run the Project

    Clone or download this repository to your local machine or open the .ipynb file on Google Colab.
    Ensure the dataset is structured into train, val, and test directories, where each directory contains subdirectories named after ASL words, each containing corresponding video files.
    Run the notebook step by step to:
        Process the videos into feature matrices.
        Train the CNN-LSTM models.
        Evaluate the model on the test set.
        Use the prediction pipeline to predict ASL signs from new videos.

## Dataset

The dataset used in this project consists of video recordings of various ASL signs. Each video is processed into a matrix of landmarks using MediaPipe. The dataset is split into train, val, and test directories for model training, validation, and evaluation.
Example Usage

To predict the ASL sign for a new video:

python

```
video_path = "video_data/val/computer/12335.mp4"
prediction = pipeline(video_path)
print(f"The model predicts: {prediction}")
```

Saving and Loading the Model

Once the model is trained, it can be saved and later loaded for inference:

python

### Save the trained model
```
model_b.save('sign_model.keras')
```

#### Load the model
```
model_b = tf.keras.models.load_model('sign_model.keras')
```

Results

The performance of the models is evaluated on the test set using accuracy and loss curves. The trained model can predict ASL signs with high accuracy, making it a useful tool for translating ASL videos into text.
## References

    MediaPipe
    TensorFlow


This `README.md` file explains the project setup, key functionalities, and how to run the code, providing a clear overview for anyone using it on GitHub or Colab.

