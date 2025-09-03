# AI Stock Analysis Dashboard

StockYfy is a comprehensive web application built with Flask for stock market analysis. It provides users with tools for technical analysis, historical performance tracking, and an AI-powered research assistant to query financial news and documents.

## Features

- **Interactive Stock Dashboard:**
  - View real-time price and growth for popular stocks like AAPL, MSFT, GOOGL, and TSLA.
  - Dynamic sparkline charts for a quick visual overview of recent performance.
- **In-Depth Technical Analysis:**
  - Generate detailed, interactive charts for any stock ticker within a specified date range.
  - Calculate and visualize key technical indicators including:
    - Candlestick Charts
    - Simple Moving Averages (SMA 20, 50, 200)
    - Exponential Moving Averages (EMA 20, 50)
    - Relative Strength Index (RSI)
    - Bollinger Bands (BB)
  - Calculate potential profit or loss based on purchase/sell dates and number of shares.
- **Historical Performance Summary with Statistical Data:**
  - Fetch historical stock data for a given ticker and date range with some insightful Stats.
  - Generate a downloadable Excel (`.xlsx`) file containing the raw data along with calculated indicators like Daily/Cumulative Returns, MAs, EMAs, and RSI.
- **AI-Powered News Research Tool:**
  - Ingest financial data from multiple sources: URLs, raw text, or CSV files.
  - Uses Cohere for high-quality text embeddings and FAISS for efficient vector indexing and similarity search.
  - Query the ingested documents using natural language to get factual, context-aware answers from an LLM (Mistral-7B).

## Tech Stack

- **Backend:** Python, Flask, Gunicorn
- **Frontend:** HTML, CSS, JavaScript
- **Data & AI:**
  - **`yfinance`**: For fetching stock market data.
  - **`pandas` & `numpy`**: For data manipulation and analysis.
  - **`matplotlib` & `plotly`**: For generating static and interactive charts.
  - **`cohere`**: For creating text embeddings.
  - **`faiss-cpu`**: For fast similarity search on vector embeddings.
  - **`langchain`**: For text splitting and processing.
  - **`openpyxl`**: For generating Excel files.
  - **OpenRouter API**: To access the Mistral-7B language model for the Q&A feature.
- **Deployment:** `Procfile` included for Heroku-like deployments.

## Local Setup and Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/rohitrebel/ai-stock-analysis-dashboard-application.git
    cd ai-stock-analysis-dashboard-application
    ```

2.  **Create and Activate a Virtual Environment:**

    ```bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    Create a file named `.env` in the root directory and add your API keys. The app uses Cohere for embeddings and OpenRouter for the LLM.

    ```
    OPENAI_API_KEY="your-openrouter-api-key"
    COHERE_API_KEY="your-cohere-api-key"
    ```

    _Note: The `OPENAI_API_KEY` variable is used to authorize requests to the OpenRouter API endpoint specified in `app.py`._

5.  **Run the Application:**
    ```bash
    flask run
    ```
    The application will be running at `http://127.0.0.1:5000`.

## Usage Guide

- **Analyze Stocks:**

  1.  Navigate to the **Analyze Stocks** page.
  2.  In the "Returns analyzer" section, enter a stock ticker (e.g., `AAPL`), number of shares, purchase date, and sell date.
  3.  Click "Submit" to view your investment summary and detailed technical analysis charts.

- **Get Performance Statistics:**

  1.  Navigate to the **Performance Statistics** page.
  2.  Enter a stock ticker and a date range.
  3.  Click "Submit". A download button will appear once processing is complete.
  4.  Click "Download Data" to save the historical data and technical indicators as an Excel file.

- **Use the News Research Tool:**
  1.  Go to the **News Research Tool** page.
  2.  Select your data input type (CSV, Text, or URL).
  3.  Provide the data (upload a file, paste text, or enter up to three URLs) and click "Submit".
  4.  After the data is processed, a "Query Box" will appear.
  5.  Enter your question about the provided data and click "Ask" to receive an AI-generated answer.

## Project Structure

```
.
├── LICENSE
├── Procfile
├── app.py
├── requirements.txt
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       ├── navbar.js
│       ├── prompttext.js
│       ├── research.js
│       ├── summary.js
│       └── update_input.js
└── templates/
    ├── index.html
    ├── perf_summary.html
    ├── research.html
    └── stock_analysis.html
```

## Output

![alt text](https://res.cloudinary.com/ddrbrwcvz/image/upload/v1756924544/Screenshot_5576_jikqvn.png)
![alt text](https://res.cloudinary.com/ddrbrwcvz/image/upload/v1756924544/Screenshot_5577_qkknfk.png)
![alt text](https://res.cloudinary.com/ddrbrwcvz/image/upload/v1756924544/Screenshot_5579_ms2dmz.png)
![alt text](https://res.cloudinary.com/ddrbrwcvz/image/upload/v1756924544/Screenshot_5578_uldgws.png)
![alt text](https://res.cloudinary.com/ddrbrwcvz/image/upload/v1756924544/Screenshot_5580_xcnma5.png)
![alt text](https://res.cloudinary.com/ddrbrwcvz/image/upload/v1756924545/Screenshot_5581_rrvtte.png)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
