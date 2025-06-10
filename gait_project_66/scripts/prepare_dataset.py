import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/processed_gait_dataset.csv")

def load_and_label(dir_path):
    dfs = []
    for file in dir_path.glob("*.csv"):
        df = pd.read_csv(file, header=None)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def main():
    print("🔄 Loading CSVs...")
    normal_df = load_and_label(RAW_DIR / "normal")
    cerebellar_df = load_and_label(RAW_DIR / "cerebellar")
    full_df = pd.concat([normal_df, cerebellar_df], ignore_index=True)
    full_df.to_csv(OUTPUT_FILE, index=False, header=False)
    print(f"✅ Saved processed dataset: {OUTPUT_FILE}")
    print(f"📊 Shape: {full_df.shape}")

if __name__ == "__main__":
    main()
