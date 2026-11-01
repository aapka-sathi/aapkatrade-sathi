"""
backend/train_model.py — NiftyBot LSTM Brain Trainer
=====================================================
Purpose : Trains a BiLSTM model on Indian market data (Nifty 50, Sensex, Bank Nifty).
          Outputs trading_brain.tflite → place it in:
          frontend/app/src/main/assets/trading_brain.tflite

Usage   : cd D:\\NewEnterpriseProjectTrading\\backend
          pip install -r requirements.txt
          python train_model.py

Output  : trading_brain.tflite (copy manually to assets/)

Feature Layout (must stay in sync with FeatureExtractor.java):
  F0: Normalized Close      (MinMax)
  F1: Normalized Volume     (MinMax)
  F2: RSI-14                (÷100)
  F3: Normalized MACD       (MinMax)
  F4: Normalized EMA-10     (MinMax)
  F5: Normalized EMA-30     (MinMax)
  F6: ATR-14 / Close        (ratio)
  F7: Day-of-Week           (÷4)
  F8: News Sentiment        (placeholder 0.5 during training)
  F9: Padding               (zero)

Model Output: [PUT_prob, NEUTRAL_prob, CALL_prob] (3-class softmax)
"""

import numpy as np
import tensorflow as tf
import os
import warnings
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

TICKERS     = ["^NSEI", "^BSESN", "^NSEBANK"]   # Indian markets only
TIME_STEPS  = 60    # 60 trading days lookback
FEATURES    = 10    # feature count (sync with FeatureExtractor.java)
EPOCHS      = 30
BATCH_SIZE  = 64
PERIOD      = "5y"  # 5 years of data

# Signal label thresholds
CALL_THRESHOLD = 0.003   # next-day return > +0.3% → CALL
PUT_THRESHOLD  = -0.003  # next-day return < -0.3% → PUT


# ─── Technical Indicators ─────────────────────────────────────────────────────

def compute_rsi(series, period=14):
    """RSI-14 normalized to [0, 100]. Returns pandas Series."""
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs   = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def compute_macd(series, fast=12, slow=26):
    """MACD line = EMA(fast) - EMA(slow)."""
    return series.ewm(span=fast, adjust=False).mean() \
         - series.ewm(span=slow, adjust=False).mean()

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_atr(high, low, close, period=14):
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()

def minmax(series_or_array):
    """MinMax normalize to [0, 1]."""
    arr = series_or_array.values if hasattr(series_or_array, 'values') else series_or_array
    return MinMaxScaler().fit_transform(arr.reshape(-1, 1)).flatten()


# ─── Label Generation ─────────────────────────────────────────────────────────

def make_labels(close_series):
    """
    Generate 3-class labels based on next-day return:
      0 = PUT    (return < -0.3%)
      1 = NEUTRAL
      2 = CALL   (return > +0.3%)
    """
    returns = close_series.pct_change().shift(-1)  # next-day return
    labels  = np.where(returns > CALL_THRESHOLD, 2,
               np.where(returns < PUT_THRESHOLD, 0, 1))
    return labels


# ─── Data Loading ─────────────────────────────────────────────────────────────

def get_training_data():
    """
    Downloads 5 years of daily OHLCV for Indian indices.
    Returns X: float32 [N, 60, 10], y: int32 [N] (0/1/2)
    """
    X_all, y_all = [], []

    print("=" * 60)
    print("NiftyBot — LSTM Brain Trainer for Indian Markets")
    print(f"Training on: {', '.join(TICKERS)}")
    print("=" * 60)

    for ticker in TICKERS:
        print(f"\n[{ticker}] Downloading {PERIOD} of data...")
        try:
            df = yf.download(ticker, period=PERIOD, interval="1d",
                             progress=False, auto_adjust=True)
        except Exception as e:
            print(f"  [{ticker}] FAILED: {e}")
            continue

        # Flatten MultiIndex columns (yfinance >= 0.2.x)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.dropna(subset=["Close", "Volume", "High", "Low"])
        if len(df) < TIME_STEPS + 40:
            print(f"  [{ticker}] Only {len(df)} rows — skipping")
            continue

        close  = df["Close"]
        volume = df["Volume"]
        high   = df["High"]
        low    = df["Low"]

        # ── Compute features ──────────────────────────────────────────────────
        rsi14   = compute_rsi(close, 14)
        macd    = compute_macd(close, 12, 26)
        ema10   = compute_ema(close, 10)
        ema30   = compute_ema(close, 30)
        atr14   = compute_atr(high, low, close, 14)

        norm_close  = minmax(close)
        norm_vol    = minmax(volume)
        norm_macd   = minmax(macd)
        norm_ema10  = minmax(ema10)
        norm_ema30  = minmax(ema30)
        atr_ratio   = (atr14 / close).values
        rsi_norm    = (rsi14 / 100.0).values
        dow_norm    = (df.index.dayofweek / 4.0).values  # Mon=0, Fri=4 → ÷4

        # Labels (3-class: PUT/NEUTRAL/CALL)
        labels = make_labels(close)

        # Build combined feature matrix
        combined = np.column_stack([
            norm_close,   # F0
            norm_vol,     # F1
            rsi_norm,     # F2
            norm_macd,    # F3
            norm_ema10,   # F4
            norm_ema30,   # F5
            atr_ratio,    # F6
            dow_norm,     # F7
            np.full(len(close), 0.5),   # F8: sentiment (0.5 = neutral during training)
            np.zeros(len(close)),        # F9: padding
        ])

        # Drop NaN rows (first ~30 from EMA/RSI warmup)
        valid = ~np.isnan(combined).any(axis=1)
        combined = combined[valid]
        labels_arr = labels[valid]

        if len(combined) < TIME_STEPS + 1:
            print(f"  [{ticker}] After NaN drop: {len(combined)} rows — skipping.")
            continue

        # Build sliding windows
        for i in range(len(combined) - TIME_STEPS):
            X_all.append(combined[i : i + TIME_STEPS])
            y_all.append(labels_arr[i + TIME_STEPS])

        call_count    = np.sum(labels_arr == 2)
        put_count     = np.sum(labels_arr == 0)
        neutral_count = np.sum(labels_arr == 1)
        print(f"  [{ticker}] ✓ {len(combined) - TIME_STEPS} samples | "
              f"CALL: {call_count} PUT: {put_count} NEUTRAL: {neutral_count}")

    if not X_all:
        raise RuntimeError("No training data — check internet connection.")

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)
    print(f"\nTotal samples: {len(X)}")
    print(f"Class distribution — CALL: {np.sum(y==2)} | NEUTRAL: {np.sum(y==1)} | PUT: {np.sum(y==0)}")
    return X, y


# ─── Model Architecture ────────────────────────────────────────────────────────

def build_model():
    """
    Bidirectional LSTM — 3-class classifier for CALL / NEUTRAL / PUT.
    Architecture tuned for Indian market volatility and 60-day lookback.
    """
    inp = tf.keras.Input(shape=(TIME_STEPS, FEATURES), name="market_input")

    # BiLSTM Layer 1: captures broad 60-day patterns
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True), name="bilstm_64"
    )(inp)
    x = tf.keras.layers.Dropout(0.25)(x)

    # BiLSTM Layer 2: refines signal
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=False), name="bilstm_32"
    )(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Dense head
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)

    # Output: 3 classes [PUT, NEUTRAL, CALL]
    output = tf.keras.layers.Dense(3, activation="softmax", name="signal_out")(x)

    model = tf.keras.Model(inp, output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def build_and_export():
    print("\nBuilding BiLSTM model for Indian markets...")
    model = build_model()
    model.summary()

    X, y = get_training_data()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-5, verbose=1
        ),
    ]

    print(f"\nTraining on {len(X)} samples (up to {EPOCHS} epochs)...")
    model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    # ── Convert to TFLite ─────────────────────────────────────────────────────
    print("\nConverting to TFLite for Android...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # ── Determine output paths ────────────────────────────────────────────────
    #
    # GitHub Actions: script is run from repo ROOT (where .github/ lives)
    #   → writes to repo root so Actions can git-add and push it.
    #
    # Local dev:      script lives in backend/ folder
    #   → writes to BOTH repo root AND assets/ so Android Studio picks it up.
    #
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..")) if os.path.basename(script_dir).lower() == "backend" else script_dir

    # On GitHub Actions, CWD is the repo root and there's no frontend/ folder
    is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"

    # Always write to repo root (GitHub Actions will git-push this)
    root_path = os.path.join(repo_root, "trading_brain.tflite")
    with open(root_path, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(root_path) / 1024
    print(f"\n✅ trading_brain.tflite → repo root ({size_kb:.1f} KB)")

    if not is_github_actions:
        # Also write to assets/ for local development (removes manual copy step)
        assets_dir = os.path.join(repo_root, "frontend", "app", "src", "main", "assets")
        os.makedirs(assets_dir, exist_ok=True)
        assets_path = os.path.join(assets_dir, "trading_brain.tflite")
        with open(assets_path, "wb") as f:
            f.write(tflite_model)
        print(f"✅ trading_brain.tflite → assets/ ({size_kb:.1f} KB)")
        print(f"\n{'='*60}")
        print(f"🎉 NO MANUAL COPY NEEDED — assets/ updated automatically!")
        print(f"   In Android Studio → Build → Clean Project → Run ▶")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"🤖 GitHub Actions: trading_brain.tflite will be pushed to repo.")
        print(f"   Android app will OTA-download it on next startup.")
        print(f"{'='*60}")


if __name__ == "__main__":
    build_and_export()

