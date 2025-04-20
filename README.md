# NLP
# News Headline Classification Project

## Overview
This project implements a deep learning-based news headline classification system that automatically categorizes news headlines into predefined categories based on their content. The system uses Natural Language Processing (NLP) techniques and a neural network to achieve high accuracy in classification tasks.

## Features
- Processes and categorizes news headlines into 9 consolidated categories
- Handles class imbalance through oversampling techniques
- Uses TF-IDF vectorization for feature extraction
- Implements a deep neural network for classification
- Provides model persistence for deployment
- Includes an interactive interface for real-time headline classification

## Technologies Used
- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework for model creation and training
- **scikit-learn**: For TF-IDF vectorization, data splitting, and evaluation metrics
- **imbalanced-learn**: For handling class imbalance through oversampling
- **pandas**: For data manipulation and preprocessing
- **NumPy**: For numerical operations
- **Matplotlib & Seaborn**: For data visualization and model performance analysis
- **pickle**: For serializing and deserializing Python objects

## NLP Techniques Implemented
- **Text Preprocessing**: Converting text to lowercase and removing non-alphanumeric characters
- **TF-IDF Vectorization**: Converting text data to numerical features while capturing word importance
- **N-gram Analysis**: Capturing phrases (bigrams) in addition to individual words
- **Class Imbalance Handling**: Using RandomOverSampler to balance underrepresented categories

## Neural Network Architecture
- Input layer based on TF-IDF feature size (10,000 features)
- Dense hidden layers with ReLU activation
- Dropout layers for regularization
- L2 regularization to prevent overfitting
- Softmax output layer for multi-class classification

## Dataset
The project uses the "News Category Dataset v3" which contains headlines from various news sources. The original categories are consolidated into 9 main categories:
- Politics/News
- Health/Lifestyle
- Entertainment
- Sports
- Tech
- Business
- Lifestyle
- Family
- Social Issues
- Crime/Other

## Usage

### Training the Model
```python
# Run the training script
# This will train the model and save it as 'news_category_model.h5'
python train_model.py
```

### Using the Trained Model
```python
# Load the saved model
loaded_model = tf.keras.models.load_model('news_category_model.h5')

# Load the vectorizer and category names
with open('tfidf_vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

with open('category_names.pkl', 'rb') as f:
    category_names = pickle.load(f)

# Classify a headline
def predict_headline(headline):
    headline_cleaned = headline.lower().replace(r'[^\w\s]', '')
    headline_vec = loaded_vectorizer.transform([headline_cleaned]).toarray()
    prediction = loaded_model.predict(headline_vec, verbose=0)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    predicted_category = category_names[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx]
    return predicted_category, confidence
```

### Interactive Mode
```python
# Run the interactive classification tool
python classify_headlines.py
```

## Performance
The model achieves approximately 85% accuracy on the test set after addressing class imbalance issues. The confusion matrix and accuracy plots are generated during training to visualize model performance.

## Files Included
- `train_model.py`: Script for preprocessing data and training the model
- `classify_headlines.py`: Interactive tool for headline classification
- `news_category_model.h5`: Saved model file
- `tfidf_vectorizer.pkl`: Serialized TF-IDF vectorizer
- `category_names.pkl`: Serialized category names
- `README.md`: This file

## Future Improvements
- Implement more advanced NLP techniques like word embeddings (Word2Vec, GloVe)
- Experiment with transformer models like BERT for improved accuracy
- Create a web API for remote classification
- Add support for multiple languages
- Implement confidence thresholds for more reliable predictions

## Requirements
- Python 3.7+
- TensorFlow 2.x
- scikit-learn 1.0+
- imbalanced-learn
- pandas
- numpy
- matplotlib
- seaborn

## Installation
```bash
pip install tensorflow scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
