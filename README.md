# Analyzing and predicting the future performance of NVIDIA (NVDA) as an asset in an investment portfolio. 
---

## Table of Contents

1. [Overview](#overview)
2. [Methods of Assessment](#methods-of-assessment)
3. [Objectives](#objectives)
4. [Libraries and Packages Used](#libraries-and-packages-used)
5. [Data Collection and Preparation](#data-collection-and-preparation)
6. [Step 1: Clustering with K-Means and Affinity Propagation](#step-1-clustering-with-k-means-and-affinity-propagation)
7. [Step 2: Sentiment Analysis of Asset "NVDA"](#step-2-sentiment-analysis-of-asset-nvda)
8. [Step 3: Predicting Future Value (Next 30 Days)](#step-3-predicting-future-value-next-30-days)
9. [Results](#results)
10. [Conclusion](#conclusion)
11. [Installation Instructions](#installation-instructions)
12. [License](#license)
13. [Acknowledgements](#acknowledgements)

---

## Overview

This project aims to recommend whether to add the asset "NVDA" (NVIDIA Corporation) to an existing portfolio using multiple techniques. These include:

1. **Clustering Algorithms**: To assess if the asset is different from the existing assets in the portfolio.
2. **Sentiment Analysis**: To analyze the sentiment around the asset using news articles and other textual content.
3. **Time Series Prediction**: To predict the future value of the asset (next 30 days).

---

## Methods of Assessment

1. **Clustering Algorithms**:
    - **K-Means Clustering**
    - **Affinity Propagation**
   
2. **Sentiment Analysis** using NLP on news and articles.

3. **Future Value Prediction** (Next 30 days):
    - **SARIMAX (Time Series Analysis)**
    - **LSTM**

---

## Objectives

The goal is to evaluate:

1. **Volatility of the Asset**: Using clustering to check how the asset behaves compared to existing assets in the portfolio.
2. **Sentiment of the Asset**: Analyzing the public sentiment around the asset.
3. **Future Value**: Forecasting the asset's future value using time series analysis and LSTM.

---

## Libraries and Packages Used

- `yfinance` for financial data collection
- `sklearn` for clustering algorithms
- `transformers` for NLP models and sentiment analysis
- `tensorflow` for LSTM model implementation
- `matplotlib`, `seaborn` for data visualization
- `requests`, `beautifulsoup4` for scraping news articles
- `pandas`, `numpy` for data handling

---

## Data Collection and Preparation

### Stock Tickers

The list of stock tickers (excluding 'PEAK' as it is delisted) includes:

```python
tickers = ['BDX', 'BK', 'BLK', 'BRO', 'C', 'CB', 'CDW', 'CINF', 'CMCSA', 'CME', 'CMG', 'CPT',
'MCD', 'MCHP', 'MGM', 'MMC', 'MO', 'MSI', 'MTD', 'NEM', 'NI', 'NVR', 'ORCL',
'ORLY', 'PEP', 'PFE', 'PGR', 'PNW', 'RF', 'RHI', 'RL', 'SJM', 'SPG', 'STLD',
'STX', 'TECH', 'TEL', 'TER', 'TMUS', 'TRGP', 'UDR', 'UNP', 'VTR', 'WEC', 'WHR',
'WTW', 'XRAY', 'XYL', 'GOOG', 'HBAN', 'HES', 'HII', 'HPQ','NVDA']
```

# Stock Portfolio Analysis and Prediction: NVDA

## Overview

This project aims to analyze the stock **NVIDIA (NVDA)** through multiple techniques, including **clustering**, **sentiment analysis**, and **future value prediction**. The analysis spans a period from **2021-10-15 to 2024-10-15**

## Steps

### Step 1: Clustering with K-Means and Affinity Propagation

#### K-Means Clustering
The clustering process follows these steps:
1. Standardize the returns and volatility of the stocks.
2. Apply **K-Means clustering** with `n_clusters=7`.
3. Visualize the clusters using mean return and volatility.

#### Affinity Propagation
The **Affinity Propagation** algorithm is applied using:
- **Damping**: 0.5
- **Affinity**: 'euclidean'
- The estimated number of clusters is 8.

### Step 2: Sentiment Analysis of Asset "NVDA"

#### News Scraping
- Articles related to **NVDA** are scraped from **Yahoo Finance** using **BeautifulSoup**.
- Irrelevant URLs are filtered out.

#### Summarization and Sentiment Analysis
1. **Summarization**: A pre-trained model (**human-centered-summarization/financial-summarization-pegasus**) is used to summarize the articles.
2. **Sentiment Analysis**: The sentiment of the summarized content is analyzed using the **transformers sentiment-analysis pipeline**.

### Step 3: Predicting Future Value (Next 30 Days)

#### SARIMAX (Time Series Analysis)
The **SARIMAX model** is used to predict future values of **NVDA** based on historical data.

#### LSTM (Long Short-Term Memory)
A neural network-based approach is used to predict the next 30 days' values of **NVDA** using past data.

## Results

- **Clustering Results**: Both **K-Means** and **Affinity Propagation** suggest that **NVDA** is quite distinct from the existing portfolio.
- **Sentiment**: The sentiment analysis indicates a **positive outlook** based on recent articles.
- **Future Value**: Predictive models suggest a **positive trend** for **NVDA** in the coming 30 days.

## Conclusion

Based on the clustering results, sentiment analysis, and future value prediction, **NVDA** appears to be a strong addition to the existing portfolio. Its distinct characteristics, positive sentiment, and projected growth make it an attractive asset.

## Installation Instructions

To run this project, ensure you have the necessary libraries installed by running the following commands:

```bash
pip install yfinance
pip install sentencepiece
pip install transformers
pip install sklearn
pip install tensorflow
pip install beautifulsoup4
pip install requests
```
## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Yahoo Finance** for financial data
- **Transformers** by Hugging Face for NLP and sentiment analysis tools
- **Keras** and **TensorFlow** for the LSTM model
- **BeautifulSoup** for web scraping
