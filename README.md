# Cyberbullying Detection Using Machine Learning
Cyberbullying has become a major concern on social media platforms, where millions of tweets are posted every day. This project aims to build a machine-learning model that automatically detects harmful or abusive content in tweets. By combining multiple datasets and applying NLP techniques, the system identifies whether a tweet contains cyberbullying content.

## Features
* Predicts whether a tweet is *bullying* or *non-bullying*
* Uses TF-IDF vectorization for text feature extraction
* Implements machine learning classifiers for detection
* Includes data cleaning, preprocessing, and exploratory analysis
* Visualizations for dataset distribution
* Ready-to-train and easy to extend

# Project Structure
├── data/
│   ├── scraped_labeled_tweets.csv
│   ├── public_data_labeled.csv
├── Cyberbullying_Detection.ipynb
├── README.md
└── requirements.txt

# Workflow Overview 
1. Data Loading
   Loads both scraped and public labeled datasets.

2. Data Cleaning
   * Removes duplicates
   * Cleans tweets by removing URLs, mentions, special symbols
   * Lowercasing and text normalization

3. Exploratory Data Analysis
   * Visualizes class distribution
   * Displays sample tweets and insights

4. Text Preprocessing
   * Tokenization
   * Stopwords removal
   * Lemmatization
   * TF-IDF vectorization

5. Model Training
   Tested models include:
   * Logistic Regression
   * SVM
   * Naive Bayes

6. Model Evaluation
   Metrics used:
   * Accuracy
   * Precision
   * Recall
   * F1 Score

# Technologies Used 
* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib/Seaborn
* NLTK / Text Preprocessing
* Jupyter Notebook

# Results
* Achieved strong accuracy and balanced precision-recall scores.
* Model effectively identifies abusive language based on linguistic patterns.

# How to Run the Project
1. Clone the repository:

```bash
git clone https://github.com/nithin-nayak08/Detection-of-Cyberbulling.git
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook CDetection-of-Cyberbulling.ipynb
```

4. Run all cells to train and evaluate the model.

---


# Contributing
Contributions are welcome!
Feel free to fork this repository, create a branch, and submit a pull request.


# Contact
**Author:** Nithin
**Project:** Detection-of-Cyberbulling Using Machine Learning.
