def to_yf_symbol(ticker: str) -> str:
    if ticker is None:
        return ""
    t = str(ticker).strip()
    if not t:
        return ""
    if t.startswith("^") or "." in t:
        return t
    return f"{t}.NS"
