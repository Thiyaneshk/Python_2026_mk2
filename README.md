# Python_2026_mk2
cleaned up version

# 📊 Portfolio Data Engineering Project – Plan

## Stage 0 – Baseline & Scope

- Objective: Build an end to end personal portfolio platform for **IN / CA / US** stocks, ETFs, and SGBs.
- Data source: Yahoo Finance via `yfinance` (personal, research use only).
- Special rule: **Track SGB bond performance versus gold price** (e.g., gold benchmarks or GOLDBEES), but treat BEES ETFs in general as separate; logic for them will be handled later.
- Tech stack (initial):
  - Python, Streamlit
  - pandas, yfinance
  - SQLite (later Postgres)
  - dbt, Airflow (later stages)

---

## Stage 1 – Portfolio Upload & Viewing

**Goal:** Start from a blank Python project and get to “upload CSV → see portfolio” reliably.

### 1.1 Project skeleton

- Create repo structure:
  - `src/` or `app/` for code
  - `data/` for local files (CSV/JSON/DB)
  - `notebooks/` (optional) for ad hoc analysis
  - `requirements.txt` or `pyproject.toml`
- Set up virtualenv, basic lint/format (black/ruff) if desired.

### 1.2 Holdings model & CSV contract

- Define canonical holdings schema, based on `holdings-2.csv`:

  ```text
  Instrument, Qty, Avg_cost, LTP, Invested, Cur_val, PnL, Net_chg, Day_chg
•	Map to internal normalized model:
text
ticker, exchange, country, shares, buy_price, source, as_of_date
•	Add explicit handling for SGB instruments:
•	Recognize SGB tickers: SGB* (e.g., SGBMAY29I, SGBSEP31II-GB).
•	Tag them as asset_type = "SGB" and link to a gold benchmark symbol (e.g., GOLDINR, GOLDBEES) for later relative performance.
1.3 Streamlit app – minimal version
•	Page: Portfolio Upload & View
•	st.file_uploader to upload holdings CSV.
•	Parse into pandas, validate:
•	Non empty Instrument, numeric Qty and Avg cost.
•	Basic country/exchange inference (initially assume NSE / India for your current file).
•	Persist to data/portfolio.json (or data/holdings_current.json).
•	Display:
•	Table of holdings (from JSON).
•	Summary metrics: total invested, current value, P&L.
1.4 Live prices integration (yfinance)
•	For each holding, derive the correct Yahoo symbol:
•	IN: append .NS or .BO as needed.
•	CA: append .TO / .V.
•	US: raw ticker.
•	Use yfinance to fetch:
•	Latest price for each ticker.
•	Optional: short history for charts.
•	Replace LTP from CSV with live quote (or show both: broker vs Yahoo).
Stage 1 Done When:
•	You can upload holdings-2.csv, see:
•	Parsed normalized holdings.
•	Live prices per instrument.
•	Basic portfolio stats in a Streamlit UI.
________________________________________
Stage 2 – Watchlist & Research Hooks
Goal: CRUD watchlist + prep for LLM research later.
2.1 Watchlist data model
•	Define watchlist schema:
text
ticker, exchange, country, reason, status, added_date
•	Persist to data/watchlist.json.
2.2 Streamlit Watchlist UI
•	Add a Watchlist tab:
•	Add new ticker (manual form).
•	Promote from holdings to watchlist and vice versa.
•	Mark status: active / purchased / removed.
•	Show current price for watchlist tickers.
2.3 LLM research scaffolding
•	Add a panel/button:
•	Select ticker(s) from watchlist or holdings.
•	Enter a research question.
•	Store request metadata in data/research_requests.json (ticker, question, timestamp).
•	No actual LLM calls yet – just structure hooks so later you can plug in OpenAI/other.
Stage 2 Done When:
•	You can maintain a watchlist in the app.
•	Watchlist persists across sessions.
•	Research requests for tickers are logged for future LLM workflows.
________________________________________
Stage 3 – Database + Indicators + dbt
Goal: Move from flat files to a proper DB, and set up a modeling layer for indicators (RSI, ST, etc.).
3.1 Choose DB & schema
•	Start with SQLite (data/portfolio.db) for local dev.
•	Tables (raw layer):
•	raw_holdings
•	raw_watchlist
•	raw_price_history (daily OHLCV for all tickers)
•	Tables/views (modeled via dbt):
•	dim_ticker – ticker metadata (ticker, exchange, country, asset_type; SGB vs equity vs ETF).
•	fct_positions – position snapshots (shares, cost, market value).
•	fct_price_history – cleaned OHLCV.
•	fct_indicators – RSI, short/long MA, SuperTrend etc.
3.2 ETL loaders
•	Write Python scripts to:
•	Load current holdings/watchlist JSON → raw_holdings / raw_watchlist.
•	For each active ticker, call yfinance and append OHLCV to raw_price_history.
3.3 dbt project
•	Create dbt project pointing at SQLite (or Postgres if you switch early).
•	Implement models:
•	stg_raw_holdings, stg_price_history.
•	fct_price_history with cleaned/standardized columns.
•	fct_indicators computing:
•	RSI (14 period).
•	MA (20/50/200).
•	SuperTrend (if desired, or leave for Python first).
•	Add basic dbt tests (unique keys, not null, accepted values).
3.4 SGB vs Gold tracking
•	In dim_ticker, tag SGB tickers and attach a gold benchmark symbol.
•	In fct_indicators or a dedicated fct_sgb_vs_gold model:
•	Compute SGB total return vs gold benchmark over multiple windows (1Y, 3Y, etc.).
•	This is where SGB ↔ gold performance logic lives (BEES ETFs can be separate later).
Stage 3 Done When:
•	Holdings, watchlist, and price history are in SQLite.
•	dbt can build indicator views.
•	You have at least one model that compares SGB performance vs gold.
________________________________________
Stage 4 – Scheduling & Orchestration (Airflow-ready)
Goal: Automate daily data refresh and transformations.
4.1 ETL orchestration
•	Introduce Airflow locally (Docker/Astro):
•	DAG: portfolio_daily_etl
•	Task 1: ingest latest holdings (optional; if you export regularly).
•	Task 2: fetch daily OHLCV for all tickers (yfinance).
•	Task 3: run dbt run + test.
•	Schedule: once per day (e.g., after IN/US markets close).
4.2 Config & secrets
•	Move DB connection strings, API keys, etc. to:
•	.env for local.
•	Airflow connections/Variables for DAG.
4.3 Cloud ready design
•	Keep DB config abstract so you can change SQLite → Postgres/Supabase later.
•	Optionally define a Dockerfile and docker-compose.yml for:
•	Streamlit
•	Airflow
•	DB (Postgres)
•	dbt
Stage 4 Done When:
•	A daily job can run unattended, updating price history and dbt models.
•	You can trigger the DAG manually and see updated indicators in the DB.
________________________________________
Stage 5 – Analysis, Candlesticks & LLM Insights
Goal: Use the curated DB data for rich analytics and LLM based commentary.
5.1 Streamlit analytics pages
•	Pages/tabs:
•	Indicators & Signals:
•	Pull from fct_indicators.
•	Show RSI, MA, etc. with thresholds.
•	Screen tickers by signal (e.g., RSI < 30).
•	Candlestick & Volume Profile:
•	Use fct_price_history to drive Plotly candlestick/volume charts.
•	Reuse/refactor your existing market profile code.
•	SGB vs Gold:
•	Visual comparison of SGB returns vs gold over time.
5.2 LLM-based research & explanation
•	For tickers flagged by indicators (e.g., oversold SGB, high momentum equity):
•	Generate:
•	Plain English explanation of RSI/indicator state.
•	Optional news/summary (if you later integrate external APIs).
•	Persist outputs in a research_notes table:
•	ticker, date, signal, prompt, summary.
Stage 5 Done When:
•	You’re using DB/dbt data in Streamlit for:
•	Technical and portfolio analytics.
•	Visual candlestick/volume charts.
•	Basic LLM generated commentary.
________________________________________
Stage 6 – Hardening & Extras (Optional)
•	Testing:
•	Unit tests for ETL and indicator logic.
•	dbt tests for data quality.
•	Data quality tooling (optional):
•	Great Expectations/Soda for more rigorous checks.
•	Extended domains:
•	Options positions, FX, or multi currency P&L.
•	Packaging:
•	Turn ETL/analytics into Python packages for reuse.
________________________________________
Notes & Constraints
•	SGB tracking: always ensure SGB performance is evaluated against a gold benchmark, not just absolute price.
•	BEES ETFs: treat them as normal holdings for now; add custom rules later.
•	Personal use: yfinance/Yahoo Finance data is for personal research; do not treat as production/commercial data.

