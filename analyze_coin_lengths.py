import sys
from pathlib import Path
import pandas as pd


def main() -> int:
    # Configuration (allow optional CLI args: input_csv [output_csv])
    if len(sys.argv) >= 2:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = Path("cleaned_preprocessed.csv")

    if len(sys.argv) >= 3:
        output_csv = Path(sys.argv[2])
    else:
        # Derive output name from input name
        stem = csv_path.stem
        output_csv = Path(f"coin_lengths_summary_{stem}.csv")

    print("\n" + "=" * 50)
    print(f"COIN OHLC LENGTH ANALYSIS ({csv_path.name})")
    print("=" * 50)

    if not csv_path.exists():
        print(f"❌ File not found: {csv_path.resolve()}")
        return 1

    # Load data with robust date parsing
    print("Loading data...")
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"], low_memory=False)
    except Exception:
        # Fallback if date column missing or cannot be parsed
        df = pd.read_csv(csv_path, low_memory=False)

    print(f"Data shape: {df.shape}")

    # Validate required columns
    required_cols = ["coin", "open", "high", "low", "close"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        print("⚠ Missing expected columns:", missing_required)
        print("Proceeding with available columns. OHLC completeness will only consider present columns.")

    # Determine which OHLC columns are present
    ohlc_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]

    if "coin" not in df.columns:
        print("❌ Required column 'coin' not found in CSV.")
        return 2

    # Ensure date exists for unique date counts and ranges
    has_date = "date" in df.columns

    # Group and compute metrics
    print("Computing per-coin OHLC lengths...")

    records = []
    for coin, g in df.groupby("coin"):
        total_rows = len(g)

        # Rows with all available OHLC columns non-null
        if ohlc_cols:
            complete_ohlc_rows = g[ohlc_cols].notna().all(axis=1).sum()
        else:
            complete_ohlc_rows = pd.NA

        # Unique date count and ranges
        if has_date:
            unique_dates = g["date"].nunique(dropna=True)
            try:
                min_date = g["date"].min()
                max_date = g["date"].max()
            except Exception:
                min_date = pd.NA
                max_date = pd.NA
        else:
            unique_dates = pd.NA
            min_date = pd.NA
            max_date = pd.NA

        records.append(
            {
                "coin": coin,
                "total_rows": int(total_rows),
                "complete_ohlc_rows": (int(complete_ohlc_rows) if pd.notna(complete_ohlc_rows) else pd.NA),
                "unique_dates": (int(unique_dates) if pd.notna(unique_dates) else pd.NA),
                "min_date": min_date,
                "max_date": max_date,
            }
        )

    summary = pd.DataFrame.from_records(records)

    # Sort for readability: by coin name then by total_rows desc
    summary_sorted_by_coin = summary.sort_values(["coin"]).reset_index(drop=True)
    summary_sorted_by_len = summary.sort_values(["total_rows"], ascending=False).reset_index(drop=True)

    # Save to CSV
    summary.to_csv(output_csv, index=False)

    # Print outputs
    print("\nPer-coin lengths (sorted by coin):")
    print(summary_sorted_by_coin.to_string(index=False))

    print("\nTop coins by total_rows:")
    print(summary_sorted_by_len.head(30).to_string(index=False))

    print(f"\n✓ Wrote detailed summary to: {output_csv.resolve()}")

    # Provide a minimal machine-readable line for quick copy/paste
    print("\ncoin,total_rows,complete_ohlc_rows,unique_dates,min_date,max_date (first 10 by length):")
    print(summary_sorted_by_len.head(10).to_csv(index=False).strip())

    return 0


if __name__ == "__main__":
    sys.exit(main())


