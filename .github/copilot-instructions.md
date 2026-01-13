## Repo quick-start for AI coding agents

This file contains focused, actionable conventions and pointers to help an AI agent be productive in this Streamlit portfolio app.

- **Big picture:** The app is a Streamlit multi-page dashboard (entry: `streamlit_app.py`) with feature files under `app/` (examples: `0001_Portfolio_Stage01_Upload_and_View.py`, `0001_Portfolio_Stage02_Watchlist_and_Research.py`). Core helpers live in `utils/`.

- **Primary flow:** CSV upload → parse (`parse_holdings_csv`) → save/load JSON (`data/portfolio_stage01.json`) → fetch live prices (via `get_live_price`, uses `yfinance`) → render tables & Plotly charts.

- **Key files to read first:**
  - `streamlit_app.py` — app entry/route registration
  - `app/0001_Portfolio_Stage02_Watchlist_and_Research.py` — advanced UI and comparisons
  - `app/0001_Portfolio_Stage01_Upload_and_View.py` — CSV parsing and basic view
  - `utils/yf_symbols.py` — `to_yf_symbol(ticker)` appends `.NS` for Indian tickers
  - `utils/utils.py` — NSE fetch utilities (`get_nifty50_list`, `get_stock_list`), caching patterns

- **Project-specific patterns and conventions (do not change lightly):**
  - `to_yf_symbol()` enforces `.NS` for non-index Indian tickers — use it before calling yfinance.
  - SGB (sovereign gold bonds) handling is custom: `get_live_price()` maps SGB tickers to gold proxies (e.g. uses `GC=F` and `INR=X`). See `get_live_price` in the app files.
  - Persistent portfolio stored at `data/portfolio_stage01.json` — updates must preserve schema: list of objects with `ticker`, `shares`, `buy_price`, `asset_type`.
  - Use Streamlit caching decorators already present (`@st.cache_data`, `@st.cache_resource`) for repeated network calls.

- **External integrations:**
  - Yahoo Finance (`yfinance`) for historical and live pricing
  - Plotly for charts (`plotly.express` + `plotly.graph_objects`) and Streamlit for rendering
  - NSE CSV scrapers in `utils/_fetch_nse_csv()` with retry/backoff logic

- **Current known issues & recent fixes:**
  - `use_container_width` was removed by Streamlit — use `width='stretch'` (True) or `width='content'` (False). Example: `st.plotly_chart(fig, width='stretch')`.
  - `NIFTY50_YF` constant expected in `app/0001_Portfolio_Stage02_Watchlist_and_Research.py`; define it as `'^NSEI'` when adding the benchmark symbol to yfinance queries.

- **Developer workflows / commands:**
  - Run locally (activate venv first):
    - Windows PowerShell:
      - `python -m venv .venv` (if not created)
      - `.\.venv\Scripts\Activate.ps1`
      - `pip install -r requirements.txt` or use `pyproject.toml` tooling
      - `streamlit run streamlit_app.py`

- **When editing files:**
  - Preserve the `st.cache_data` usage for network-bound functions.
  - Use `to_yf_symbol()` when converting user tickers for yfinance requests.
  - Maintain `PORTFOLIO_FILE` JSON schema used by `load_portfolio()` / `save_portfolio()`.

- **Testing edits locally:**
  - Make small, iterative changes and run `streamlit run streamlit_app.py` to observe runtime errors (they are printed in the browser/console).
  - If adding network calls, use `@st.cache_data(ttl=300)` for light caching.

- **If you need to add new constants or mappings:**
  - Add them near the existing `CONFIGURATION` block at top of the app file (where `GOLD_PROXY` and `DATA_DIR` are defined).

If anything is unclear or you'd like this trimmed to a shorter checklist or expanded with more example diffs, tell me which areas to focus on.
