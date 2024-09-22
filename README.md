# SARCASM-DETECTION
This project aims to develop a sarcasm detection system that can classify whether a given news headline is sarcastic or not. Using deep learning techniques, specifically Long Short-Term Memory (LSTM) networks, the model processes textual data and predicts sarcasm with high accuracy.



## Key Features

- **Dataset**: The model is trained on the Sarcasm Headlines Dataset, which contains labeled headlines. The dataset is in JSON format and is zipped for storage efficiency.
  
- **Text Preprocessing**: Headlines are cleaned by removing punctuation, converting text to lowercase, and filtering out common stopwords.

- **Deep Learning Model**: The model utilizes a Sequential neural network consisting of:
  - Embedding Layer: Transforms words into dense vectors.
  - Bidirectional LSTM Layers: Captures context from both directions of text (forward and backward).
  - Dense Layers: Applies nonlinear transformations to make the final classification.

- **Training and Evaluation**: The model is trained using an 80/20 train-test split, and early stopping is used to avoid overfitting. The evaluation is based on accuracy and confusion matrix visualizations.

## Libraries Used

- **TensorFlow** and **Keras** for building and training the neural network.
- **NLTK** for text preprocessing.
- **Pandas** for data manipulation.
- **Seaborn** and **Matplotlib** for visualizing model performance.

