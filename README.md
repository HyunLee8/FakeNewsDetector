#  Fake News Detection using LSTM

A deep learning project that detects whether a news headline is **fake or real** using **Natural Language Processing (NLP)** and **LSTM (Long Short-Term Memory)** neural networks.

---

## Overview

This project trains an LSTM-based model on a dataset of true and fake news headlines to classify text as **real** or **fake**.  
It uses text preprocessing (tokenization, stopword removal, stemming) and word embeddings before feeding data into the neural network.

---

## Features

- Preprocessing of text data with:
  - Stopword removal
  - Stemming using `SnowballStemmer`
  - Custom regex-based cleaning
- Word embeddings via Keras `Embedding` layer
- LSTM-based model for sequential learning
- Evaluation using accuracy and adjustable thresholds
- Early stopping and best model checkpoint saving

---

## Project Structure

