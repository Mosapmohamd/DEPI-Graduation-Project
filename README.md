# Tourism Sentiment Analysis and Recommendation System

## Project Overview
This project leverages AI and data science to enhance Egypt's tourism sector by analyzing visitor sentiments from online reviews. Using Natural Language Processing (NLP) and Machine Learning (ML), the system classifies sentiments and generates actionable recommendations to improve tourist experiences.

https://bilstmpy-m6s8ts8adzgn4zfzyi9zji.streamlit.app/

![Screenshot 2025-05-13 130829](https://github.com/user-attachments/assets/581a1ee4-3b96-4691-9774-505a0ca08ba9)


## Table of Contents

- [Features](#features)
- [Project Scope](#project-scope)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [System Components](#system-components)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Future Work](#future-work)
- [Team](#team)

## Features
- **Sentiment Classification**: Advanced ML models to categorize reviews as positive or negative
- **Aspect-Based Sentiment Analysis**: Extracting specific aspects tourists are happy or unhappy about
- **Recommendation Generation**: AI-powered suggestions for tourism improvement based on negative feedback
- **Interactive Web Interface**: Real-time sentiment prediction and visualization through Streamlit
- **Data-Driven Insights**: Actionable intelligence for tourism stakeholders

## Project Scope
The project analyzes English-language reviews from Tripadvisor and Quora covering Egypt's historical sites, museums, and cultural venues. It includes:
- Data collection from online platforms
- Text preprocessing and cleaning
- Sentiment classification using various ML models
- Aspect-based sentiment extraction
- LLM-based recommendation generation
- Web application deployment for real-time predictions

## Methodology

### Data Collection and Preprocessing
1. **Data Gathering**: Scraped over 30,000 tourism reviews from Tripadvisor and Quora
2. **Text Preprocessing**:
   - Lowercasing
   - URL removal
   - Emoji conversion
   - Slang replacement
   - Tokenization
   - Stopword removal
   - Lemmatization
3. **Sentiment Annotation**: Employed ensemble of pre-trained models (Roberta, VADER, DistilBERT) with majority voting

### Handling Class Imbalance
The dataset contained significantly more positive reviews (34,000) than negative ones (4,000). Two approaches were implemented:

1. **For BiLSTM Model**:
   - Targeted undersampling
   - Class weights
   - Focal loss function
   - F1-based early stopping

2. **For Classical ML and Other Deep Learning Models**:
   - Train/test splitting first
   - Upsampling applied only to training data
   - Original imbalanced test set preserved for realistic evaluation

## Model Architecture

### Models Evaluated
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | 87.27% | 88% |
| LGBMClassifier | 86.8% | 87% |
| XGBoost Classifier | 87% | 89% |
| Support Vector Machine (SVM) | 88.2% | 87% |
| Stochastic Gradient Descent (SGD) | 87.26% | 89% |
| Random Forest | 91.12% | 88% |
| BiLSTM | 90.09% | 89.9% |
| GRU | 91% | 88% |

### BiLSTM Architecture (Selected Model)
- **Input**: Tokenized and padded sequences (max length = 384)
- **Embedding Layer**: vocab_size=10,000, output_dim=100
- **SpatialDropout1D**: 0.4 rate
- **Bidirectional LSTM**: 64 units with forward and backward processing
- **Dropout Layer**: 0.4 rate
- **Dense Layer**: 64 units with ReLU activation
- **Dropout Layer**: 0.5 rate
- **Output**: Dense layer with sigmoid activation
- **Loss Function**: BinaryFocalCrossentropy
- **Optimizer**: Adam (learning rate = 0.0001)
- **Early Stopping**: Custom F1-based

## Results
The BiLSTM model was selected as the final model due to:
1. **Minority Class Performance**: Highest F1-score and recall for underrepresented negative reviews
2. **Strong Generalization**: Less overfitting compared to other models
3. **Optimal Metrics Balance**: Best overall performance across accuracy, precision, recall, and F1-score

## System Components

### BiLSTM model for Classifying Reviews

### Recommendation System
The project integrates Large Language Models (LLMs) to generate actionable recommendations:
1. Local testing with Ollama
2. Final implementation with Google Gemini Flash via Langchain and Google Generative AI
3. Triggered by negative sentiment detection to provide improvement suggestions

### Web Application
- Built with Streamlit
- Provides user interface for input and output display
- Integrates the BiLSTM model for sentiment prediction
- Displays LLM-generated recommendations based on negative feedback

## Usage
1. Enter or paste a tourism review text into the input field
2. Click "Analyze Sentiment" to process the review
3. View the sentiment classification result
4. For negative reviews, examine the generated recommendations for improvement

## Future Work
- **Multi-Language Support**: Extend analysis to Arabic, French, German, and other languages
- **Improved Aspect Extraction**: Fine-tune models for more specific topic sentiment extraction
- **Real-Time Monitoring**: Integrate live review streams for dynamic tracking
- **Geospatial Analysis**: Create sentiment heatmaps highlighting positive and negative areas

## Team
- Mosap Mohamed
- Maryam Sayed
- Aya Ehab
- Mohamed Mahmoud
- Naden Mohamed
- Ahmed Walied

