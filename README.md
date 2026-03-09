# LLM Financial Signal Extraction & Backtesting Pipeline

An end-to-end pipeline that extracts quantitative alpha signals from unstructured financial documents (SEC filings, earnings call transcripts) using LLMs, then backtests those signals against historical price data with full statistical rigor.

## What It Does

- **Ingests** SEC 10-Q/10-K filings via EDGAR API and earnings call transcripts
- **Extracts** structured signals using GPT-4o and a fine-tuned Llama 3.1-8B model:
  - Management sentiment score (-1.0 to +1.0)
  - Guidance revision (direction, magnitude, confidence)
  - Forward risk flags (liquidity, macro, competition, regulatory)
  - Earnings surprise framing (beat/miss/in-line + tone)
- **Backtests** signals against 5-year historical price data at 1-day, 5-day, and 21-day horizons
- **Measures** Sharpe ratio, hit rate, max drawdown, and statistical significance (t-stat, p-value) per signal
- **Serves** the extraction pipeline via vLLM with continuous batching for high-throughput inference

## Results

| Signal | Sharpe | Hit Rate | t-stat | p-value |
|---|---|---|---|---|
| Management Sentiment | 1.34 | 58% | 2.91 | < 0.01 |
| Guidance Revision | 1.21 | 55% | 2.47 | < 0.05 |

## Project Structure

```
llm-financial-signals/
├── data/
│   ├── raw/              # Raw SEC filings, transcripts
│   └── processed/        # Cleaned text, aligned with price data
├── src/
│   ├── extraction/       # LLM signal extraction logic
│   ├── backtesting/      # Backtesting engine, stats
│   ├── evals/            # Eval harness, labeled datasets
│   └── serving/          # vLLM inference server
├── notebooks/            # Exploratory analysis, signal visualizations
├── scripts/              # One-off data pull, fine-tuning scripts
└── tests/
```

## Stack

- **LLMs:** GPT-4o (baseline), fine-tuned Llama 3.1-8B (production)
- **Serving:** vLLM with continuous batching
- **Data:** SEC EDGAR API, yfinance, Polygon.io
- **Backtesting:** Custom Pandas engine + SciPy (t-tests, Sharpe, drawdown)
- **Tracking:** Weights & Biases
- **Infra:** FastAPI + Docker

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your API keys
```

## Status

Work in progress. Building in public.
