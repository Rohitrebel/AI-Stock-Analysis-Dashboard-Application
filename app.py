import os
from io import BytesIO, StringIO
from datetime import date, timedelta
from functools import lru_cache
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_file, jsonify, Response

# Keep these two light imports at top
import numpy as np
import faiss

# Keep tokenizers quiet & a bit leaner
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

app = Flask(__name__)

# --------- Globals kept minimal ---------
faiss_index = None
retrieved_info = None
csv_file = None
ticker = None

# Config
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # smaller than mpnet
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Simple UI assets
ticker_logo = {
  "AAPL": ["fa-brands", "fa-apple", "#ffffff", "background:#000;"],
  "MSFT": ["fa-brands", "fa-windows", "#ffffff", "background:#000;"],
  "GOOGL": ["fa-brands", "fa-google", "#B197FC", "background:#fff;"],
  "TSLA": ["fa-brands", "fa-tesla", "#ffffff", "background:#fff;"],
}

def _today_yesterday(days_back=5):
  today = date.today()
  yesterday = today - timedelta(days=days_back)
  return today, yesterday

@lru_cache(maxsize=1)
def get_encoder():
  # Lazy-load the sentence transformer (saves ~200–400MB at boot)
  from sentence_transformers import SentenceTransformer
  return SentenceTransformer(MODEL_NAME)  # all-MiniLM-L6-v2 by default

def get_stock_data(ticker_in):
  # Lazy imports for heavy libs
  import yfinance as yf
  import pandas as pd  # noqa: F401

  today, yesterday = _today_yesterday()

  if not ticker_in:
    ticker_in = "AAPL"

  stock_ticker = yf.Ticker(ticker_in)
  history = stock_ticker.history(start=yesterday, end=today, interval="1d", prepost=True)

  if history is None or history.empty:
    close_value = None
    growth = None
  else:
    close_value = history["Close"].iloc[-1]
    try:
      growth = ((history["Close"].iloc[-1] - history["Open"].iloc[-1]) / history["Open"].iloc[-1]) * 100
    except Exception:
      growth = None

  ticker_content = {
    "ticker": ticker_in,
    "ticker_color_logo": ticker_logo.get(ticker_in, ["fa-solid", "fa-chart-line", "#fff", "background:#222;"]),
  }
  card = {"ticker": ticker_in, "price": close_value, "growth": growth}
  return ticker_content, card

# -------------------- Routes --------------------

@app.route("/", methods=["GET", "POST"])
def home():
  return render_template("index.html")

@app.route("/stock-analysis", methods=["GET", "POST"])
def stock_analysis():
  ticker_in = request.form.get("ticker") if request.method == "POST" else request.form.get("stock-name")

  tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
  stock_cards = []
  for t in tickers:
    ticker_content, card = get_stock_data(t)
    stock_cards.append({"ticker_content": ticker_content, "card": card})

  return render_template("stock_analysis.html", stock_cards=stock_cards, submitted=True)

@app.route("/stream")
def stream():
  # Lazy import heavy matplotlib only when needed
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import yfinance as yf

  stock_ticker = request.args.get("ticker")
  color = request.args.get("color", "lightyellow")
  figsize = (3, 1)

  fig, ax = plt.subplots(figsize=figsize)
  today, yesterday = _today_yesterday()
  try:
    if not stock_ticker:
      ax.axis("off")
    else:
      stock = yf.Ticker(stock_ticker)
      history = stock.history(start=yesterday, end=today, interval="30m", prepost=True)
      if history.empty or "Close" not in history:
        ax.axis("off")
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
          ax.axis("off")
  except Exception as e:
    print(f"Error generating graph for {stock_ticker}: {e}")
    ax.axis("off")

  img = BytesIO()
  fig.savefig(img, format="png", transparent=True, bbox_inches="tight", pad_inches=0)
  plt.close(fig)
  img.seek(0)
  return send_file(img, mimetype="image/png")

@app.route("/stock-analysis/analyze", methods=["GET", "POST"])
def analyze():
  import yfinance as yf
  import pandas as pd
  import plotly.graph_objects as go

  tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
  stock_cards = []
  for t in tickers:
    ticker_content, card = get_stock_data(t)
    stock_cards.append({"ticker_content": ticker_content, "card": card})

  if request.method != "POST":
    return render_template("stock_analysis.html", stock_cards=stock_cards, submitted=True)

  ticker_in = request.form.get("stock-name")
  shares = request.form.get("shares") or 0
  purchaseDate = request.form.get("date-purchase")
  sellDate = request.form.get("date-sell")

  try:
    if not (ticker_in and purchaseDate and sellDate):
      return render_template(
        "stock_analysis.html",
        stock_cards=stock_cards,
        graph="<p>Please provide ticker, purchase date and sell date.</p>",
        submitted=True,
      )

    if purchaseDate == sellDate:
      return render_template(
        "stock_analysis.html",
        stock_cards=stock_cards,
        graph="<p>Purchase Date and Sell Date cannot be the same</p>",
      )

    stock = yf.Ticker(ticker_in)
    history_data = stock.history(start=purchaseDate, end=sellDate, interval="1d", prepost=True)

    if history_data is None or history_data.empty or "Close" not in history_data:
      return render_template(
        "stock_analysis.html",
        stock_cards=stock_cards,
        graph="<p>Sorry! Data Not Available for the given range.</p>",
        submitted=True,
      )

    df = history_data.reset_index().dropna(subset=["Close"]).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Indicators
    df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
    df["MA200"] = df["Close"].rolling(200, min_periods=1).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["Daily Return"] = df["Close"].pct_change()
    df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    roll_up = gain.rolling(14, min_periods=1).mean()
    roll_down = loss.rolling(14, min_periods=1).mean()
    RS = roll_up / (roll_down.replace(0, np.nan))
    df["RSI"] = 100 - (100 / (1 + RS))
    df["RSI"] = df["RSI"].fillna(50)

    std20 = df["Close"].rolling(20, min_periods=1).std()
    df["BB_Middle"] = df["MA20"]
    df["BB_Upper"] = df["MA20"] + 2 * std20
    df["BB_Lower"] = df["MA20"] - 2 * std20

    # Charts
    fig_candle = go.Figure(
      data=[
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candle",
      )])

    fig_candle.update_layout(
      title=f"{ticker_in.upper()} Candlestick Chart", xaxis_rangeslider_visible=False, template="plotly_white", width=900, autosize=False
    )

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
    fig_ma.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20"))
    fig_ma.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], mode="lines", name="MA50"))
    fig_ma.add_trace(go.Scatter(x=df["Date"], y=df["MA200"], mode="lines", name="MA200"))
    fig_ma.update_layout(title=f"{ticker_in.upper()} SMA 20/50/200", template="plotly_white", width=900, autosize=False)

    fig_ema = go.Figure()
    fig_ema.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
    fig_ema.add_trace(go.Scatter(x=df["Date"], y=df["EMA20"], mode="lines", name="EMA20"))
    fig_ema.add_trace(go.Scatter(x=df["Date"], y=df["EMA50"], mode="lines", name="EMA50"))
    fig_ema.update_layout(title=f"{ticker_in.upper()} EMA 20 & 50", template="plotly_white", width=900, autosize=False)

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], mode="lines", name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dot")
    fig_rsi.add_hline(y=30, line_dash="dot")
    fig_rsi.update_layout(title=f"{ticker_in.upper()} RSI (14)", template="plotly_white", width=900, autosize=False)

    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
    fig_bb.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], mode="lines", name="BB Upper"))
    fig_bb.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], mode="lines", name="BB Lower"))
    fig_bb.update_layout(title=f"{ticker_in.upper()} Bollinger Bands", template="plotly_white", width=900, autosize=False)

    # Profit / loss
    pd_purchase = pd.to_datetime(purchaseDate)
    pd_sell = pd.to_datetime(sellDate)
    df_indexed = df.set_index("Date").sort_index()

    try:
      purchase_price = df_indexed.loc[:pd_purchase].iloc[-1]["Close"]
    except Exception:
      purchase_price = df["Close"].iloc[0]

    try:
        sell_price = df_indexed.loc[:pd_sell].iloc[-1]["Close"]
    except Exception:
        sell_price = df["Close"].iloc[-1]

    try:
        shares_int = int(shares)
    except Exception:
        shares_int = 0

    profit_loss = (sell_price - purchase_price) * shares_int
    profit_pct = ((sell_price - purchase_price) / purchase_price) * 100 if purchase_price != 0 else None

    config_common = {"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select", "pan"]}
    graph_candle = fig_candle.to_html(full_html=False, config=config_common)
    graph_ma = fig_ma.to_html(full_html=False, config=config_common)
    graph_ema = fig_ema.to_html(full_html=False, config=config_common)
    graph_rsi = fig_rsi.to_html(full_html=False, config=config_common)
    graph_bb = fig_bb.to_html(full_html=False, config=config_common)

    return render_template(
        "stock_analysis.html",
        submitted=True,
        stock_cards=stock_cards,
        graph_candle=graph_candle,
        graph_ma=graph_ma,
        graph_ema=graph_ema,
        graph_rsi=graph_rsi,
        graph_bb=graph_bb,
        profit_loss=profit_loss,
        profit_pct=profit_pct,
    )

  except Exception as err:
    print("Error Occurred in analyze():", err)
    return render_template(
        "stock_analysis.html",
        submitted=True,
        stock_cards=stock_cards,
        graph="<p>Something went wrong while processing your request.</p>",
    )

@app.route("/news-research-tool", methods=["GET", "POST"])
def research():
  global faiss_index, retrieved_info, ticker

  if request.method == "POST":
    # Lazy import heavy langchain only if needed
    from langchain_community.document_loaders import UnstructuredURLLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import pandas as pd

    if request.form.get("url-analyze"):
      urls = [request.form.get("url1"), request.form.get("url2"), request.form.get("url3")]
      urls = [u for u in urls if u]
      if not urls:
        return jsonify({"status": "error", "message": "No URLs provided"}), 400
      loader = UnstructuredURLLoader(urls)
      data = loader.load()
      text = " ".join([d.page_content for d in data])

    elif request.form.get("csv-analyze"):
      csv_up = request.files.get("fileUpload")
      if not csv_up:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
      df = pd.read_csv(csv_up)
      text = "\n".join(df.astype(str).agg(" ".join, axis=1).tolist())
      text = f"{ticker or 'TICKER'}_stock_data\n" + text

    elif request.form.get("text-analyze"):
      text = request.form.get("text") or ""
      if not text.strip():
        return jsonify({"status": "error", "message": "No text provided"}), 400
    else:
      return jsonify({"status": "error", "message": "No valid input type"}), 400

    # Split & embed
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_text(text)
    if not chunks:
      return jsonify({"status": "error", "message": "No valid text to process"}), 400

    encoder = get_encoder()
    vectors = encoder.encode(chunks, convert_to_numpy=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss_index = index
    retrieved_info = chunks
    return jsonify({"status": "done"})

  return render_template("research.html")

def get_response(user_input):
  load_dotenv()
  OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
  
  if not OPENAI_API_KEY:
    return "OpenAI API key not configured on server."

  import requests
  url = "https://openrouter.ai/api/v1/chat/completions"
  headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
  body = {
    "model": "mistralai/mistral-7b-instruct",
    "messages": [
      {"role": "system", "content": "You are a financial research assistant. Your job is to explain financial news clearly and factually. Base your answer ONLY on the context given. If the context does not contain enough information to answer, say: I don’t know based on the given data. When explaining: - Answer the query clearly. - Summarize clearly. - Mention important numbers and company names. - Avoid copying raw sentences from context. Rewrite in natural English."},
      {"role": "user", "content": user_input},
    ],
  }
  try:
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
  except Exception as e:
    return f"LLM call failed: {e}"

@app.route("/continueprocess", methods=["POST"])
def continueurl():
  global faiss_index, retrieved_info

  if faiss_index is None or retrieved_info is None:
    return jsonify({"error": "No data indexed yet. Please upload/submit data first."}), 400

  if not retrieved_info:
    return jsonify({"error": "No data stored for retrieval"}), 400

  data = request.get_json(silent=True) or {}
  query = data.get("userPrompt", "")

  try:
    encoder = get_encoder()
    query_vec = encoder.encode([query], convert_to_numpy=True).reshape(1, -1)
    distances, indices = faiss_index.search(query_vec, k=3)
    results = [retrieved_info[i] for i in indices[0]]
    context = "\n".join(results)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    answer = get_response(prompt)
    return jsonify({"answer": answer})
  except Exception as e:
    return jsonify({"error": f"Failed to process: {e}"}), 500

@app.route("/performance-summary", methods=["GET", "POST"], endpoint="summary")
def summary():
  import yfinance as yf
  import pandas as pd

  if request.method == "POST":
    if request.form.get("submit-request"):
      global csv_file, ticker
      ticker = request.form.get("stock-ticker")
      start_date = request.form.get("start-date")
      end_date = request.form.get("end-date")

      stock_ticker = yf.Ticker(ticker)
      history_data = stock_ticker.history(start=start_date, end=end_date, interval="1d")

      df = pd.DataFrame(history_data).reset_index()
      csv_buffer = StringIO()
      df.to_csv(csv_buffer, index=False)
      csv_buffer.seek(0)
      csv_file = csv_buffer.getvalue()
      return jsonify({"status": "done"})

    elif request.form.get("download-request"):
      return Response(
          csv_file,
          mimetype="text/csv",
          headers={"Content-Disposition": f"attachment;filename={ticker or 'ticker'}_data.csv"},
      )

  return render_template("perf_summary.html")

if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host="0.0.0.0", port=port)
