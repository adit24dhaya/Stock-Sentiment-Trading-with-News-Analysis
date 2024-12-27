# Stock Sentiment Trading with News Analysis

This project explores the relationship between news sentiment and stock price movements, specifically focusing on Tesla (TSLA). It uses Natural Language Processing (NLP) techniques to analyze news articles and a neural network to predict sentiment, which is then used in a simulated trading strategy.

## Description

This project combines financial data analysis with NLP to create a system that simulates trading Tesla stock based on the sentiment expressed in news articles. The workflow involves:

1.  **Data Acquisition:** Gathering historical Tesla stock prices using `yfinance` and news articles related to Tesla using the NewsAPI.
2.  **Data Preprocessing:** Cleaning and preprocessing the text data using techniques like removing HTML tags, stop words, and applying lemmatization.
3.  **Sentiment Analysis:** Calculating sentiment scores using VADER (Valence Aware Dictionary and sEntiment Reasoner) and training a neural network on TF-IDF and Word2Vec features.
4.  **Trading Simulation:** Implementing a basic trading strategy that buys or sells Tesla stock based on the predicted sentiment and a set of predefined rules.
5.  **Performance Evaluation:** Evaluating the trading strategy's performance by calculating metrics such as total gain/loss and percentage return and visualizing the account balance over time.

## Key Features

*   **News Data Collection:** Automated fetching of Tesla-related news articles from the NewsAPI.
*   **Text Preprocessing:** Robust text cleaning and preprocessing pipeline.
*   **Sentiment Analysis:** Sentiment scoring using VADER and a trained neural network.
*   **Feature Engineering:** TF-IDF and Word2Vec feature extraction for text representation.
*   **Neural Network Classification:** Multi-Layer Perceptron (MLP) for sentiment classification.
*   **Backtesting Framework:** Simulation of a trading strategy based on news sentiment.
*   **Performance Metrics:** Calculation of key trading metrics and visualization of account balance.

## Technologies Used

*   Python
*   yfinance
*   NewsAPI
*   nltk
*   gensim
*   scikit-learn
*   PyTorch
*   pandas
*   matplotlib

## Installation

1.  Clone the repository:

    ```bash
    git clone [invalid URL removed]
    ```

2.  Navigate to the project directory:

    ```bash
    cd StockSentimentTrading
    ```

3.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

    Create a `requirements.txt` file with the following contents:

    ```
    newsapi-python
    gensim
    yfinance
    scikit-learn
    torch
    pandas
    matplotlib
    nltk
    transformers
    accelerate
    bitsandbytes
    ```

4. You will need to download the Google News Word2Vec model and store it in your Google Drive. Then, mount your drive in the Colab notebook.

5.  Obtain a NewsAPI key and replace `"16aa8e5e0e6c4f34a6c1746193575766"` with your actual key in the script.

## Usage

Run the main script:

```bash
python your_script_name.py # Replace your_script_name.py with the actual name of your script.
```
## The script will:

* Fetch stock and news data.
* Preprocess the text data and perform sentiment analysis.
* Train the neural network models.
* Simulate the trading strategy.
* Print performance metrics and generate a balance plot.

##    Example Output
The script will output performance metrics like total gain/loss, percentage return, and a plot of the account balance over time. It will also generate classification reports for the sentiment analysis models.

##    Disclaimer
This project is for educational purposes only. Trading stocks based on sentiment analysis is highly risky, and past performance is not indicative of future results. Do not use this project for actual financial trading without consulting a qualified financial advisor.

##Future Improvements
Explore more sophisticated trading strategies.
Implement different sentiment analysis models or fine-tune existing ones.
Incorporate other financial indicators or market data.
Develop a more robust backtesting framework with transaction costs and slippage.
Deploy the system as a web application or API.
