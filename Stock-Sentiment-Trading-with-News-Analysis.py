!pip install newsapi-python
!pip install gensim
!pip install yfinance

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from gensim.downloader import load
import json
from newsapi import NewsApiClient

# Download required NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')

"""TSLA News Articles and Stock """


# Fetch Tesla Stock Data
def fetch_tesla_stock_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    stock_data = yf.download("TSLA", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    stock_data.reset_index(inplace=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
    return stock_data

# Fetch Tesla News Data
def fetch_tesla_news_by_weeks():
    api_key = "16aa8e5e0e6c4f34a6c1746193575766"  # Replace with your NewsAPI key
    newsapi = NewsApiClient(api_key=api_key)
    today = datetime.now()
    start_date = today - timedelta(days=30)
    all_articles = []
    for i in range(0, 30, 7):
        week_start = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
        week_end = (start_date + timedelta(days=i + 6)).strftime('%Y-%m-%d')
        articles = newsapi.get_everything(q="Tesla AND TSLA", from_param=week_start, to=week_end, language="en", page_size=100)
        all_articles.extend(articles.get("articles", []))
    news_df = pd.DataFrame(all_articles)
    if 'publishedAt' in news_df.columns:
        news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt']).dt.date
    return news_df

# Load Data
tesla_stock_df = fetch_tesla_stock_data()
news_df = fetch_tesla_news_by_weeks()

# Flatten the MultiIndex columns in tesla_stock_df if present
if isinstance(tesla_stock_df.columns, pd.MultiIndex):
    tesla_stock_df.columns = [' '.join(col).strip() for col in tesla_stock_df.columns.values]

# Verify the new column names
print("Flattened columns in tesla_stock_df:", tesla_stock_df.columns)

# Ensure 'Date' is properly formatted in tesla_stock_df
if 'Date' in tesla_stock_df.columns:
    tesla_stock_df['Date'] = pd.to_datetime(tesla_stock_df['Date'], errors='coerce').dt.date
else:
    raise KeyError("'Date' column is missing in tesla_stock_df after flattening. Check the structure.")

# Ensure 'publishedAt' (renamed 'published') is properly formatted in news_df
if 'publishedAt' in news_df.columns:
    news_df['published'] = pd.to_datetime(news_df['publishedAt'], errors='coerce').dt.date
else:
    raise KeyError("'publishedAt' column is missing in news_df.")

# Flatten the 'source' column in news_df
if 'source' in news_df.columns:
    news_df['source'] = news_df['source'].apply(lambda x: x['name'] if isinstance(x, dict) else None)

# Drop rows with invalid or missing dates
tesla_stock_df = tesla_stock_df.dropna(subset=['Date'])
news_df = news_df.dropna(subset=['published'])

# Merge the dataframes on the date columns
merged_df = pd.merge(
    tesla_stock_df,
    news_df,
    left_on='Date',
    right_on='published',
    how='left'
)

# Debugging: Check the first few rows of the merged dataframe
print("First few rows of merged_df:")
print(merged_df.head())

# Remove duplicate rows
duplicate_count = merged_df.drop(columns=['source'], errors='ignore').duplicated().sum()
print(f"Duplicate rows: {duplicate_count}")
merged_df = merged_df.drop_duplicates()

# Drop unnecessary columns if they exist
columns_to_drop = ['source_str']
merged_df = merged_df.drop(columns=[col for col in columns_to_drop if col in merged_df.columns], errors='ignore')

# Check and summarize remaining missing values
missing_values = merged_df.isnull().sum()
print("Missing values per column:")
print(missing_values)

# Save the merged DataFrame to a file for future use
merged_df.to_csv('merged_tesla_data_last_3_months.csv', index=False)
merged_df.to_json('merged_tesla_data_last_3_months.json', orient='records', date_format='iso')
print("Merged data saved to 'merged_tesla_data_last_3_months.csv' and 'merged_tesla_data_last_3_months.json'.")

"""Integrate Stock and News Data"""

if news_df.empty:
    print("No news data available.")
    raise KeyError("The news DataFrame is empty. Please check the API response.")

# Drop rows with critical missing values
critical_columns = ['title', 'content', 'publishedAt']
merged_df = merged_df.dropna(subset=critical_columns)

# Fill missing values for non-critical columns
merged_df['author'] = merged_df['author'].fillna('Unknown Author')
merged_df['description'] = merged_df['description'].fillna('No description available.')
merged_df['urlToImage'] = merged_df['urlToImage'].fillna('No image available.')

# Check for missing values after cleanup
missing_values_after_cleanup = merged_df.isnull().sum()
print("Missing values after cleanup:")
print(missing_values_after_cleanup)

# Clean column names to remove extra spaces
tesla_stock_df.columns = [''.join(col.split()) for col in tesla_stock_df.columns]

# Verify the cleaned column names
print("Cleaned columns in tesla_stock_df:", tesla_stock_df.columns)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re

def clean_text(text, perform_stemming=False):
    """
    Cleans and preprocesses text by removing HTML tags, digits, special characters,
    converting to lowercase, removing stopwords, and applying lemmatization or stemming.

    Args:
        text (str): Input text to clean.
        perform_stemming (bool): Whether to apply stemming instead of lemmatization. Defaults to False.

    Returns:
        str: Cleaned and preprocessed text.
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', str(text))

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Convert text to lowercase
    text = text.lower()

    # Tokenize text
    tokens = text.split()

    # Load stopwords
    stop_words = set(stopwords.words('english'))

    # Initialize lemmatizer and stemmer
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # Remove stopwords and apply lemmatization or stemming
    if perform_stemming:
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    else:
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Rejoin tokens into a single string
    return ' '.join(tokens)

from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize VADER SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """
    Analyze sentiment using VADER and return the compound score.

    Args:
        text (str): Input text to analyze.

    Returns:
        float: Compound sentiment score.
    """
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

# Apply text cleaning
merged_df['clean_content'] = merged_df['content'].fillna('').apply(clean_text)

# Apply VADER sentiment analysis
merged_df['sentiment_score'] = merged_df['clean_content'].apply(analyze_sentiment)

# Display a sample of sentiment scores for debugging
print("Sample sentiment scores:")
print(merged_df[['clean_content', 'sentiment_score']].head())

"""Transaction Logging"""

# Initialize a global time variable
current_time = datetime.now()

def log_trade(transaction_type, shares, amount, sentiment_score, balance):
    global current_time  # Use the global variable to track time
    trade = {
        'date': current_time.strftime('%Y-%m-%d %H:%M:%S'),  # Unique timestamp
        'transaction_type': transaction_type,
        'shares': shares,
        'amount': amount,
        'sentiment_score': sentiment_score,
        'remaining_balance': balance
    }

    # Increment the timestamp slightly for the next trade
    current_time += timedelta(seconds=1)

    # Append the trade to the log
    trade_log.append(trade)

    # Save the trade log to a JSON file
    with open('trade_log.json', 'w') as file:
        json.dump(trade_log, file, indent=4)

"""Generate Features (TF-IDF and Word2Vec)"""

from gensim.models import KeyedVectors

# Specify the path to the Word2Vec Google News model
model_path = "/content/drive/MyDrive/GoogleNews-vectors-negative300.bin"

# Load the Word2Vec Google News model from the file
print(f"Loading Word2Vec model from {model_path}. This may take some time...")
word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Confirm model is loaded successfully
print("Model loaded successfully!")
print(f"Number of words in the vocabulary: {len(word2vec.index_to_key)}")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors

# Load Word2Vec Google News model from the local file path
model_path = "/content/drive/MyDrive/GoogleNews-vectors-negative300.bin"
print(f"Loading Word2Vec model from {model_path}. This may take some time...")
word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)
print("Word2Vec model loaded successfully!")

# Generate TF-IDF Features
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X_tfidf = vectorizer.fit_transform(merged_df['clean_content']).toarray()

# Generate Word2Vec Features
def get_embedding(text):
    tokens = text.split()
    embeddings = [word2vec[word] for word in tokens if word in word2vec]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(300)

# Apply the embedding function to each text in the DataFrame
print("Generating Word2Vec embeddings for the content...")
merged_df['embedding'] = merged_df['clean_content'].apply(get_embedding)

# Stack the embeddings into a 2D numpy array
X_word2vec = np.vstack(merged_df['embedding'].values)
print("Word2Vec embeddings generated successfully!")

merged_df['sentiment_score']

# Binary Sentiment Labels
merged_df['sentiment_score'] = X_tfidf.sum(axis=1)  # Placeholder sentiment logic
y = (merged_df['sentiment_score'] > 0).astype(int)

# Split Data
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# Neural Network for Classification
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=(128, 64), dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_sizes[0])
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)  # Added dropout
        self.fc2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.dropout2 = nn.Dropout(dropout_rate)  # Added dropout
        self.fc3 = nn.Linear(hidden_layer_sizes[1], 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after activation
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout after activation
        x = self.fc3(x)
        return x

class NewsDataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features (array-like): Input features (e.g., TF-IDF or Word2Vec embeddings).
            labels (array-like): Corresponding labels for the features.
        """
        # Check for missing values in a general way
        if isinstance(features, pd.DataFrame) or isinstance(features, pd.Series):
            if features.isnull().values.any():
                raise ValueError("Features contain missing values. Clean your data before training.")
        elif isinstance(features, np.ndarray):
            if np.isnan(features).any():
                raise ValueError("Features contain missing values. Clean your data before training.")

        if isinstance(features, pd.DataFrame) or isinstance(features, pd.Series):
            features = features.to_numpy()
        if isinstance(labels, pd.Series):
            labels = labels.to_numpy()

        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_and_evaluate(X_train, X_test, y_train, y_test, input_size, epochs=10, patience=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_size=input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)

    # Create datasets and dataloaders
    train_dataset = NewsDataset(X_train, y_train)
    test_dataset = NewsDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    best_loss = float('inf')
    patience_counter = 0

    # Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluate on the validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Print training and validation loss
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Check for early stopping
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0  # Reset counter if validation loss improves
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    # Evaluation Loop
    predictions = []
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Compute additional metrics
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

    return model, predictions

import json
from datetime import datetime, timedelta

# Initialize account parameters
initial_balance = 100000
current_balance = initial_balance
current_shares = 0
shares_per_trade = 10  # Adjust trade size
trade_log = []

# Global timestamp for unique trade dates
current_time = datetime.now()

def log_trade(transaction_type, shares, amount, sentiment_score, balance):
    """Log each trade to a JSON file."""
    global current_time
    trade = {
        'date': current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'transaction_type': transaction_type,
        'shares': shares,
        'amount': amount,
        'sentiment_score': sentiment_score,
        'remaining_balance': balance
    }
    current_time += timedelta(seconds=1)
    trade_log.append(trade)
    with open('trade_log.json', 'w') as file:
        json.dump(trade_log, file, indent=4)

# Validate data
required_columns = ['sentiment_score', 'Close TSLA']
if not all(col in merged_df.columns for col in required_columns):
    raise ValueError(f"Missing required columns in merged_df: {required_columns}")

merged_df = merged_df.dropna(subset=required_columns)

# Backtesting logic
for index, row in merged_df.iterrows():
    sentiment_score = row['sentiment_score']
    closing_price = row['Close TSLA']

    if sentiment_score > 0 and current_balance >= closing_price * shares_per_trade:
        # Buy shares
        cost = closing_price * shares_per_trade
        current_balance -= cost
        current_shares += shares_per_trade
        log_trade('Buy', shares_per_trade, cost, sentiment_score, current_balance)

    elif sentiment_score < 0 and current_shares >= shares_per_trade:
        # Sell shares
        revenue = closing_price * shares_per_trade
        current_balance += revenue
        current_shares -= shares_per_trade
        log_trade('Sell', shares_per_trade, revenue, sentiment_score, current_balance)

# Calculate final performance metrics
final_value = current_balance + current_shares * merged_df['Close TSLA'].iloc[-1]
total_gain_loss = final_value - initial_balance
percent_return = (total_gain_loss / initial_balance) * 100

# Display trading summary
print(f"--- Trading Summary ---")
print(f"Total $gain or $loss: ${total_gain_loss:.2f}")
print(f"Percentage return compared to the initial balance: {percent_return:.2f}%")

# Save final summary
with open('final_trade_summary.json', 'w') as file:
    json.dump({
        'total_gain_loss': total_gain_loss,
        'percent_return': percent_return,
        'initial_balance': initial_balance,
        'final_balance': current_balance,
        'final_shares': current_shares,
    }, file, indent=4)

import matplotlib.dates as mdates
from datetime import datetime

# Convert date strings to datetime objects
dates = [datetime.strptime(trade['date'], '%Y-%m-%d %H:%M:%S') for trade in trade_log]
balances = [trade['remaining_balance'] for trade in trade_log]

# Plot the account balance over time
plt.figure(figsize=(10, 6))
plt.plot(dates, balances, marker='o')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xlabel('Date')
plt.ylabel('Balance')
plt.title('Account Balance Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.metrics import classification_report

# Helper function to format and print reports
def print_formatted_report(title, report):
    print("\n" + "=" * 50)
    print(f"{title:^50}")
    print("=" * 50)
    print(report)

# Train and evaluate the TF-IDF-based model
print("\n" + "=" * 50)
print("Training and Evaluating the TF-IDF-Based Model".center(50))
print("=" * 50)
model_tfidf, predictions_tfidf = train_and_evaluate(
    X_train_tfidf, X_test_tfidf, y_train, y_test, input_size=X_train_tfidf.shape[1], epochs=10
)

# Train and evaluate the Word2Vec-based model
print("\n" + "=" * 50)
print("Training and Evaluating the Word2Vec-Based Model".center(50))
print("=" * 50)
X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(
    X_word2vec, y, test_size=0.2, random_state=42
)
model_word2vec, predictions_word2vec = train_and_evaluate(
    X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec, input_size=X_train_word2vec.shape[1], epochs=10
)

# Generate classification reports for both models with output_dict=True
report_tfidf_dict = classification_report(y_test, predictions_tfidf, output_dict=True)
report_tfidf = classification_report(y_test, predictions_tfidf)
print_formatted_report("TF-IDF Classification Report", report_tfidf)

report_word2vec_dict = classification_report(y_test_word2vec, predictions_word2vec, output_dict=True)
report_word2vec = classification_report(y_test_word2vec, predictions_word2vec)
print_formatted_report("Word2Vec Classification Report", report_word2vec)

# Create a comparison DataFrame
comparison_df = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1-Score", "Accuracy"],
    "TF-IDF": [
        report_tfidf_dict["weighted avg"]["precision"],
        report_tfidf_dict["weighted avg"]["recall"],
        report_tfidf_dict["weighted avg"]["f1-score"],
        report_tfidf_dict["accuracy"]
    ],
    "Word2Vec": [
        report_word2vec_dict["weighted avg"]["precision"],
        report_word2vec_dict["weighted avg"]["recall"],
        report_word2vec_dict["weighted avg"]["f1-score"],
        report_word2vec_dict["accuracy"]
    ]
})

# Display the comparison
print("\n" + "=" * 50)
print("Comparison of TF-IDF and Word2Vec Models".center(50))
print("=" * 50)
print(comparison_df.to_string(index=False))

# Save the comparison to a CSV file
comparison_file = 'performance_comparison.csv'
comparison_df.to_csv(comparison_file, index=False)
print("\nComparison data saved to:", comparison_file)
