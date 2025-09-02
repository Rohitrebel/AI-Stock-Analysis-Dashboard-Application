import yfinance as yf
from flask import Flask, render_template, request, send_file, jsonify, Response
from io import BytesIO
import pandas as pd
import matplotlib            
matplotlib.use("Agg")      
import matplotlib.pyplot as plt  # NEW
import plotly.graph_objects as go
import numpy as np
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from datetime import date, timedelta
from io import StringIO
from llama_cpp import Llama
import os
from dotenv import load_dotenv
import requests

app = Flask(__name__)

today = date.today()
yesterday = today - timedelta(days=5)

# Globals to store index + raw data
faiss_index = None
retrieved_info = None
csv_file = None
ticker= None
encoder = SentenceTransformer("all-mpnet-base-v2")

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
  raise ValueError("OpenAI API key not found in .env file")


ticker_logo = {
  'AAPL': ["fa-brands", "fa-apple", "#ffffff", "background:#000;"],   
  'MSFT': ["fa-brands", "fa-windows", "#ffffff", "background:#000;"],  
  'GOOGL': ["fa-brands", "fa-google", "#B197FC", "background:#fff;"],  
  'TSLA': ["fa-brands", "fa-tesla", "#ffffff", "background:#fff;"]     
}

def get_stock_data(ticker):
    global today, yesterday

    if not ticker:
        ticker = "AAPL"
    stock_ticker = yf.Ticker(ticker)
    history = stock_ticker.history(start=yesterday, end=today, interval="1d", prepost=True)
    if history is None or history.empty:
        close_value = None
        growth = None
    else:
  
        close_value = history['Close'].iloc[-1]
        try:
            growth = ((history['Close'].iloc[-1] - history['Open'].iloc[-1]) / history['Open'].iloc[-1]) * 100
        except Exception:
            growth = None
    ticker_content = {
        "ticker": ticker,
        "ticker_color_logo": ticker_logo.get(ticker, ["fa-solid", "fa-chart-line", "#fff", "background:#222;"])
    }
    card = {
        "ticker": ticker,       
        "price": close_value,   
        "growth": growth
    }

    return ticker_content, card


@app.route('/', methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route('/stock-analysis', methods=["GET", "POST"])   
def stock_analysis():                                   
    if request.method == "POST":
        ticker = request.form.get("ticker")
    else:
        ticker = request.form.get('stock-name')


    # List of tickers to show in cards
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]

    stock_cards = []

    for t in tickers:
        ticker_content, card = get_stock_data(t)
        stock_cards.append({
            "ticker_content": ticker_content,
            "card": card
        })
    return render_template("stock_analysis.html", stock_cards=stock_cards,  submitted=True)

@app.route('/stream')
def stream():
    stock_ticker = request.args.get('ticker')
    color = request.args.get('color', 'lightyellow')
    figsize = (3, 1)

    # Always isolate matplotlib state per request
    fig, ax = plt.subplots(figsize=figsize)
    today = date.today()
    try:
        if not stock_ticker:
            ax.axis('off')
        else:
            stock = yf.Ticker(stock_ticker)
            history = stock.history(start=yesterday, end=today, interval="30m", prepost=True)

            if history.empty or "Close" not in history:
                ax.axis('off')
            else:
                df = history.dropna(subset=["Close"])
                if not df.empty:
                    x = df.index
                    y = df["Close"]

                    ax.plot(x, y, color=color, linewidth=1.5)
                    ax.fill_between(x, y, y.min(), color=color, alpha=0.2)
                    ax.axis("off")
                    ax.margins(x=0, y=0)
                else:
                    ax.axis('off')
    except Exception as e:
        print(f"Error generating graph for {stock_ticker}: {e}")
        ax.axis('off')

    # Export as PNG safely
    img = BytesIO()
    fig.savefig(img, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # IMPORTANT: close figure to avoid overlap
    img.seek(0)

    return send_file(img, mimetype='image/png')

@app.route('/stock-analysis/analyze', methods=['GET', 'POST'])
def analyze():
  # prepare the stock cards for the page (same as on /stock-analysis)
  tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
  stock_cards = []
  for t in tickers:
    ticker_content, card = get_stock_data(t)
    stock_cards.append({"ticker_content": ticker_content, "card": card})

  # show page cards when no POST
  if request.method != "POST":
    return render_template('stock_analysis.html', stock_cards=stock_cards,  submitted=True)

  # POST processing
  ticker = request.form.get('stock-name')
  amount = request.form.get('amount')
  shares = request.form.get('shares') or 0
  purchaseDate = request.form.get('date-purchase')
  sellDate = request.form.get('date-sell')

  try:
  # basic validation
    if not(ticker and purchaseDate and sellDate):
      return render_template('stock_analysis.html',
              stock_cards=stock_cards,
              graph="<p>Please provide ticker, purchase date and sell date.</p>",  submitted=True)

    if purchaseDate == sellDate:
      return render_template('stock_analysis.html',
                          stock_cards=stock_cards,
                          graph="<p>Purchase Date and Sell Date cannot be the same</p>")

    stock = yf.Ticker(ticker)
    history_data = stock.history(start=purchaseDate, end=sellDate, interval="1d", prepost=True)

    if history_data is None or history_data.empty or 'Close' not in history_data:
      return render_template('stock_analysis.html',
                        stock_cards=stock_cards,
                        graph="<p>Sorry! Data Not Available for the given range.</p>",  submitted=True)

        # dataframe preparation
    df = history_data.reset_index().dropna(subset=['Close']).copy()
    # ensure Date column is datetime and sorted
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- technical indicators ---
    # SMA
    df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(50, min_periods=1).mean()
    df['MA200'] = df['Close'].rolling(200, min_periods=1).mean()

    # EMA
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # Daily return
    df['Daily Return'] = df['Close'].pct_change()
    df['Cumulative Return'] = (1 + df['Daily Return']).cumprod()

    # RSI (14)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    roll_up = gain.rolling(14, min_periods=1).mean()
    roll_down = loss.rolling(14, min_periods=1).mean()
    RS = roll_up / (roll_down.replace(0, np.nan))
    df['RSI'] = 100 - (100 / (1 + RS))
    df['RSI']=df['RSI'].fillna(50)  # sensible default for early rows

    # Bollinger Bands
    std20 = df['Close'].rolling(20, min_periods=1).std()
    df['BB_Middle'] = df['MA20']
    df['BB_Upper'] = df['MA20'] + 2 * std20
    df['BB_Lower'] = df['MA20'] - 2 * std20

    # --- chart creation (Plotly) ---
    # Candlestick
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df["Date"], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candle'
    )])
    fig_candle.update_layout(title=f"{ticker.upper()} Candlestick Chart",
              xaxis_rangeslider_visible=False,
              template='plotly_white', autosize=True)

    # Moving averages (SMA) chart
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name="Close"))
    fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], mode='lines', name="MA20"))
    fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], mode='lines', name="MA50"))
    fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['MA200'], mode='lines', name="MA200"))
    fig_ma.update_layout(title=f"{ticker.upper()} SMA 20/50/200", template='plotly_white', autosize=True)

    # EMA chart
    fig_ema = go.Figure()
    fig_ema.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
    fig_ema.add_trace(go.Scatter(x=df['Date'], y=df['EMA20'], mode='lines', name='EMA20'))
    fig_ema.add_trace(go.Scatter(x=df['Date'], y=df['EMA50'], mode='lines', name='EMA50'))
    fig_ema.update_layout(title=f"{ticker.upper()} EMA 20 & 50", template='plotly_white', autosize=True)

    # RSI chart
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI'))
    fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
    fig_rsi.update_layout(title=f"{ticker.upper()} RSI (14)", template='plotly_white', autosize=True)

    # Bollinger bands
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
    fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], mode='lines', name='BB Upper'))
    fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], mode='lines', name='BB Lower'))
    fig_bb.update_layout(title=f"{ticker.upper()} Bollinger Bands", template='plotly_white', autosize=True)


    # --- profit / loss calculation: use nearest available rows (safer approach) ---
    # convert user dates to timestamps
    pd_purchase = pd.to_datetime(purchaseDate)
    pd_sell = pd.to_datetime(sellDate)

    # find nearest prior-or-equal rows (if exact not found)
    df_indexed = df.set_index('Date').sort_index()
    # purchase row: last index <= purchaseDate
    try:
      purchase_row = df_indexed.loc[:pd_purchase].iloc[-1]
      purchase_price = purchase_row['Close']
    except Exception:
      # fallback to first row
      purchase_price = df['Close'].iloc[0]

    try:
      sell_row = df_indexed.loc[:pd_sell].iloc[-1]
      sell_price = sell_row['Close']
    except Exception:
      # fallback to last row
      sell_price = df['Close'].iloc[-1]

    try:
      shares_int = int(shares)
    except Exception:
      shares_int = 0

    profit_loss = (sell_price - purchase_price) * shares_int
    profit_pct = ((sell_price - purchase_price) / purchase_price) * 100 if purchase_price != 0 else None

    # convert figures to HTML
    config_common = {'displayModeBar': True, 'modeBarButtonsToRemove': ['lasso2d', 'select', 'pan']}
    graph_candle = fig_candle.to_html(full_html=False, config=config_common)
    graph_ma = fig_ma.to_html(full_html=False, config=config_common)
    graph_ema = fig_ema.to_html(full_html=False, config=config_common)
    graph_rsi = fig_rsi.to_html(full_html=False, config=config_common)
    graph_bb = fig_bb.to_html(full_html=False, config=config_common)
    

    # render with all graphs and stock_cards (so cards don't disappear)
    return render_template('stock_analysis.html',  submitted=True,
      stock_cards=stock_cards,
      graph_candle=graph_candle,
      graph_ma=graph_ma,
      graph_ema=graph_ema,
      graph_rsi=graph_rsi,
      graph_bb=graph_bb,
      profit_loss=profit_loss,
      profit_pct=profit_pct)

  except Exception as err:
    # log helpful info to console and show user-friendly message
    print("Error Occurred in analyze():", err)
    return render_template('stock_analysis.html',  submitted=True,
      stock_cards=stock_cards,
      graph="<p>Something went wrong while processing your request.</p>")




@app.route("/news-research-tool", methods=["GET", "POST"])
def research():
  global faiss_index, retrieved_info

  if request.method == "POST":
    if request.form.get("url-analyze"):
      urls = [request.form.get("url1"), request.form.get("url2"), request.form.get("url3")]
      urls = [url for url in urls if url]
      loader = UnstructuredURLLoader(urls)
      data = loader.load()
      text = " ".join([d.page_content for d in data])

    elif request.form.get("csv-analyze"):
      csv_file = request.files["fileUpload"]
      if not csv_file:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

      df = pd.read_csv(csv_file)
      

      text = "\n".join(df.astype(str).agg(" ".join, axis=1).tolist())
      text = f"{ticker}_stock_data\n" + text 

    elif request.form.get("text-analyze"):
      text = request.form.get("text")

    else:
      return jsonify({"status": "error", "message": "No valid input type"})

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " "], 
            chunk_size=400, 
            chunk_overlap=100
        )
    chunks = splitter.split_text(text)

    if not chunks:
      return jsonify({"status": "error", "message": "No valid text to process"}), 400


    # Encode chunks
    vectors = encoder.encode(chunks, convert_to_numpy=True)

    # Build FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Store globally
    faiss_index = index
    retrieved_info = chunks

    return jsonify({"status": "done"})

  return render_template("research.html")


def get_response(user_input):
  url = "https://openrouter.ai/api/v1/chat/completions"
  headers = {
    "Authorization": "Bearer {}".format(OPENAI_API_KEY),
    "Content-Type": "application/json"
  }

  body = {
    "model": "mistralai/mistral-7b-instruct",
    "messages": [
      {"role": "system", "content": "You are a financial research assistant.Your job is to explain financial news clearly and factually. Base your answer ONLY on the context given. If the context does not contain enough information to answer, say: I don’t know based on the given data. When explaining: - Answer the query of the user clearly. - Summarize Clearly.- Mention important numbers and company names. - Avoid copying raw sentences from context. Rewrite in natural English."},
      {"role": "user", "content": user_input}
    ],
  }

  response = requests.post(url, headers=headers, json=body)
  return response.json()["choices"][0]["message"]["content"]


@app.route("/continueprocess", methods=["POST"])
def continueurl():
  global faiss_index, retrieved_info, encoder, llm

  if faiss_index is None or retrieved_info is None:
    return jsonify({"error": "No data indexed yet. Please upload/submit data first."}), 400


  data = request.get_json()
  query = data.get("userPrompt")

  # Encode query
  query_vec = encoder.encode([query], convert_to_numpy=True).reshape(1, -1)

  # Search in FAISS
  distances, indices = faiss_index.search(query_vec, k=3)
  results = [retrieved_info[i] for i in indices[0]]

  # Build context for LLM
  context = "\n".join(results)
  prompt = f"""

Context:
{context}

Question: {query}

Answer:
"""
  answer = get_response(prompt)

  return jsonify({"answer": answer})

@app.route('/performance-summary', methods=['GET','POST'], endpoint='summary')
def summary():
  if request.method == "POST":
    if request.form.get('submit-request'):
      global csv_file, ticker
      ticker = request.form.get('stock-ticker')
      start_date = request.form.get('start-date')
      end_date = request.form.get('end-date')

      stock_ticker = yf.Ticker(ticker)
      history_data = stock_ticker.history(start=start_date, end=end_date, interval="1d")

      df = pd.DataFrame(history_data)
      df = df.reset_index()
      csv_buffer = StringIO()
      df.to_csv(csv_buffer, index=False)
      csv_buffer.seek(0)

      csv_file = csv_buffer.getvalue()

      return jsonify({"status": "done"})

    elif request.form.get('download-request'):
      return Response(csv_file, mimetype="text/csv", 
      headers={"Content-Disposition": f"attachment;filename={ticker}_data.csv"})

  return render_template('perf_summary.html')    





if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host='0.0.0.0', port=port)





