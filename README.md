# Movie Review Sentiment Analysis with RNN Models

This project demonstrates sentiment analysis on movie reviews using Recurrent Neural Network (RNN) architectures, including Simple RNN, LSTM, and Bidirectional LSTM. It involves preprocessing movie reviews, tokenizing and lemmatizing them, and using word embeddings to train the models. The project compares the performance of different RNN architectures on the task of binary sentiment classification.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Models Implemented](#models-implemented)

## Project Overview
The goal of this project is to classify movie reviews as positive or negative using various RNN-based models. The project utilizes the Stanford Large Movie Review Dataset (IMDB), processes the text data, and builds machine learning models to predict sentiment.

## Dataset
The dataset used is the [IMDB Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz), which contains 50,000 movie reviews labeled as positive or negative.

## Project Structure
- **Data Preprocessing:** Loading the dataset, cleaning the text, and performing tokenization and lemmatization.
- **Feature Extraction:** Using word embeddings with Word2Vec to convert text data into numerical form.
- **Model Building:** Implementing Simple RNN, LSTM, and Bidirectional LSTM models for sentiment classification.
- **Model Evaluation:** Comparing the performance of different models using accuracy and loss metrics.

## Requirements
To run this project, you need the following Python packages:
- Python 3.x
- pandas
- numpy
- matplotlib
- tensorflow
- nltk
- gensim
- wordcloud
- scikit-learn

## Models Implemented
### 1. **Simple RNN**
   - **Architecture:** 
     - Embedding Layer
     - Simple RNN Layer
     - Dense Layers with Dropout
   - **Description:** A basic Recurrent Neural Network architecture that captures sequential dependencies in the text data. It uses an Embedding layer to learn word representations followed by a Simple RNN layer to process the sequences.

### 2. **LSTM (Long Short-Term Memory)**
   - **Architecture:** 
     - Embedding Layer
     - LSTM Layer
     - Dense Layers with Dropout
   - **Description:** The LSTM network is designed to capture long-term dependencies in the text. It is more effective than a simple RNN in handling vanishing gradient problems, making it suitable for sequences where context from earlier in the sequence is crucial for sentiment prediction.

### 3. **Bidirectional LSTM**
   - **Architecture:** 
     - Embedding Layer
     - Bidirectional LSTM Layer
     - Global Max Pooling
     - Dense Layers with Dropout
   - **Description:** The Bidirectional LSTM processes the input sequence in both forward and backward directions, providing a more comprehensive context understanding. This bidirectional approach allows the model to capture dependencies in both directions, enhancing its ability to predict sentiment accurately.
