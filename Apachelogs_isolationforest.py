import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from wordcloud import WordCloud
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pandas as pd
import re
import joblib

# Display the first few rows of the already loaded DataFrame
# Read Apache log file into a DataFrame using regex parsing

# Define the log pattern (adjust as needed for your log format)
log_pattern = re.compile(r'^\[(?P<datetime>.*?)\] \[(?P<level>\w+)\](?: \[client (?P<client>[^\]]+)\])? (?P<message>.*)$')

# Parse each line and extract fields
records = []
with open("Apache.log", "r") as f:
    for line in f:
        match = log_pattern.match(line.strip())
        if match:
            records.append(match.groupdict())
df = pd.DataFrame(records)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(by='datetime')

# Analyzing 'body' using NLP
# Basic text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # remove text in brackets
    text = re.sub(r'\W', ' ', text)  # remove non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text

#Applying the function
df['clean_body'] = df['message'].apply(lambda x: preprocess_text(str(x)))

# Importing libraries
from sklearn.ensemble import IsolationForest
import numpy as np
# Convert 'body' column to numerical features using CountVectorizer
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['clean_body']).toarray()

iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['anomaly_iforest'] = iso_forest.fit_predict(X)

# -1 indicates an anomaly, 1 indicates normal
anomalies_iforest = df[df['anomaly_iforest'] == -1]
print(f"Number of anomalies detected by Isolation Forest: {len(anomalies_iforest)}")

# Save the model to a file
joblib.dump(iso_forest, 'isolation_forest_model.pkl')

import joblib

# Load the model from a file
loaded_model = joblib.load('isolation_forest_model.pkl')
def predict_anomaly(new_texts):
    # Preprocess new texts
    clean_texts = [preprocess_text(str(t)) for t in new_texts]
    # Transform using the existing vectorizer
    X_new = vectorizer.transform(clean_texts).toarray()
    # Predict anomalies (-1: anomaly, 1: normal)
    return loaded_model.predict(X_new)

# Example usage:
# Example usage:
new_logs = ["attempt to invoke directory as script: /var/www/cgi-bin/","Authentication failure for user 'admin'"]
predictions = predict_anomaly(new_logs)
print(predictions)