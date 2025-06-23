import mlflow.sklearn
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
import dagshub
import os

# Initialize DAGsHub + MLflow tracking
dagshub.init(repo_owner='vinayak910', repo_name='mlops-mini-project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/vinayak910/mlops-mini-project.mlflow")

# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
df.drop(columns=['tweet_id'], inplace=True)

# ===== TEXT CLEANING FUNCTIONS =====
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def lemmatization(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(df):
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        return df
    except Exception as e:
        print(f"Normalization error: {e}")
        raise

# Normalize
df = normalize_text(df)

# Filter for binary classification
df = df[df['sentiment'].isin(['happiness', 'sadness'])]
df['sentiment'] = df['sentiment'].replace({'sadness': 0, 'happiness': 1})

# ===== MLflow EXPERIMENT CONFIG =====
mlflow.set_experiment("BoW vs TF-IDF with Classifiers")

vectorizers = {
    'BoW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

algorithms = {
    'LogisticRegression': LogisticRegression(C=1.0),
    'MultinomialNB': MultinomialNB(alpha=1.0),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=None),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
}

# ===== EXPERIMENT LOOP =====
with mlflow.start_run(run_name="All Experiments") as parent_run:
    for algo_name, algorithm in algorithms.items():
        for vec_name, vectorizer in vectorizers.items():
            try:
                with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True) as child_run:
                    X = vectorizer.fit_transform(df['content'])
                    y = df['sentiment']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

                    mlflow.log_param("vectorizer", vec_name)
                    mlflow.log_param("algorithm", algo_name)
                    mlflow.log_param("test_size", 0.2)

                    # Train model
                    model = algorithm
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Log model parameters
                    if algo_name == 'LogisticRegression':
                        mlflow.log_param("C", str(round(model.C, 2)))
                    

                    elif algo_name == 'MultinomialNB':
                        mlflow.log_param("alpha", str(round(model.alpha, 2)))
                        mlflow.log_param("C", "NA")

                    elif algo_name == 'XGBoost':
                        mlflow.log_param("n_estimators", model.n_estimators)
                        mlflow.log_param("learning_rate", model.learning_rate)
                        mlflow.log_param("C", "NA")

                    elif algo_name == 'RandomForest':
                        mlflow.log_param("n_estimators", model.n_estimators)
                        mlflow.log_param("max_depth", model.max_depth)
                        mlflow.log_param("C", "NA")

                    elif algo_name == 'GradientBoosting':
                        mlflow.log_param("n_estimators", model.n_estimators)
                        mlflow.log_param("learning_rate", model.learning_rate)
                        mlflow.log_param("max_depth", model.max_depth)
                        mlflow.log_param("C", "NA")


                    # Log metrics
                    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
                    mlflow.log_metric("precision", precision_score(y_test, y_pred))
                    mlflow.log_metric("recall", recall_score(y_test, y_pred))
                    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))



                    # Optionally log the script (skip __file__ if not running as script)
                    if os.path.exists("your_script.py"):
                        mlflow.log_artifact("your_script.py")

                    print(f"{algo_name} + {vec_name} done.")

            except Exception as e:
                print(f"⚠️ Error with {algo_name} + {vec_name}: {e}")
                mlflow.log_param("error", str(e))
