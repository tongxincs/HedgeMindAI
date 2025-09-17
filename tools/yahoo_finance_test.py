import yfinance as yf
import pandas as pd
from yahoo_finance import (
    get_quarterly_earnings_json,
    get_insider_transactions_json,
)


def test_get_info():
    # Choose a few tickers to check different categories
    tickers = ["GOOG", "XLE"]

    for symbol in tickers:
        print("=" * 60)
        print(f"Fetching info for {symbol} ...")
        try:
            stock = yf.Ticker(symbol)
            info = stock.info

            print(f"Symbol: {info.get('symbol')}")
            print(f"Date: {info.get('date')}")
            print(f"Sector: {info.get('sector')}")
            print(f"Industry: {info.get('industry')}")
            print(f"Market Cap: {info.get('marketCap')}")
            print(f"Forward PE: {info.get('forwardPE')}")
            print(f"Revenue Growth: {info.get('revenueGrowth')}")
            print(f"Profit Margins: {info.get('profitMargins')}")
            print(f"Debt to Equity: {info.get('debtToEquity')}")

        except Exception as e:
            print(f"⚠️ Failed to fetch {symbol}: {e}")

        print("\n")


def test_quarterly_and_insiders_json():
    """
    Verifies the LLM-friendly JSON your agents will consume.
    Prints all quarterly earnings (up to N) and insider summary + last tx.
    """
    tickers = ["PLTR"]

    for symbol in tickers:
        print("=" * 60)
        print(f"[JSON] Verifying normalized outputs for {symbol} ...")

        try:
            # ---------- Quarterly earnings ----------
            q = get_quarterly_earnings_json(symbol, max_quarters=4)
            qs = q.get("quarters", [])
            if not qs:
                print("  Quarterly JSON: (no data)")
            else:
                print("  Quarterly JSON (latest quarters):")
                for quote in qs:
                    growth = quote.get("growth", {})
                    rev_g = growth.get("revenue", {})
                    print(
                        f"    period={quote['period']} "
                        f"rev={quote.get('revenue')} net={quote.get('net_income')} eps={quote.get('eps_diluted')} "
                        f"rev_qoq={rev_g.get('qoq')} rev_yoy={rev_g.get('yoy')}"
                    )

            # ---------- Insider transactions ----------
            i = get_insider_transactions_json(symbol, last_n=15)
            summ = i.get("summary", {})
            print("\n  Insider JSON summary:")
            print(
                f"    total={summ.get('total_transactions')} "
                f"net_shares={summ.get('net_shares')} "
                f"total_value_usd={summ.get('total_value_usd')}"
            )
            print(f"    by_type={summ.get('by_type_counts')}")

            tx = i.get("transactions", [])
            if tx:
                print("  Recent insider transactions (last up to 5):")
                for t in tx[-5:]:  # show last 5 for context
                    print(
                        f"    date={t.get('date')} text={t.get('text')} "
                        f"filer={t.get('filer')} shares={t.get('shares')} "
                        f"price={t.get('price')} value={t.get('value')}"
                    )
            else:
                print("  No insider transactions in window.")

        except Exception as e:
            print(f"⚠️ JSON verification failed for {symbol}: {e}")

        print("\n")


def debug_print_raw_insider_transactions(tickers=["PLTR"], max_rows=8):
    for symbol in tickers:
        print("=" * 70)
        print(f"[DEBUG] Insider transactions → {symbol}")
        try:
            t = yf.Ticker(symbol)
            info = t.info or {}
            print(f"  quoteType: {info.get('quoteType')}")
            print(f"  longName:  {info.get('longName') or info.get('shortName') or ''}")

            df = getattr(t, "insider_transactions", None)
            if not isinstance(df, pd.DataFrame) or df.empty:
                print("  (No insider_transactions table available or it's empty.)")
                continue

            print(f"  shape: {df.shape}")
            print(f"  columns: {list(df.columns)}")
            print(f"  dtypes: {{k:str(v) for k,v in df.dtypes.items()}}")

            # choose columns that actually exist for this schema
            possible = [
                ["Start Date","Insider","Transaction","Shares","Value","Ownership","Price","Text","URL","Position"],
                ["startDate","filer","transaction","shares","value","ownership","price","text","url","position"],
            ]
            view_cols = next(( [c for c in cols if c in df.columns] for cols in possible ), [])
            if not view_cols:
                print("  (No matching view columns found.)")
                print(df.head(max_rows))
            else:
                sdf = df.sort_values(view_cols[0]) if view_cols[0] in df.columns else df
                print("\n  --- head ---")
                print(sdf[view_cols].head(max_rows).to_string(index=False))
                print("\n  --- tail ---")
                print(sdf[view_cols].tail(max_rows).to_string(index=False))

        except Exception as e:
            print(f"  ⚠️ error while fetching/parsing {symbol}: {e}")


if __name__ == "__main__":
    test_get_info()
    test_quarterly_and_insiders_json()
    debug_print_raw_insider_transactions()