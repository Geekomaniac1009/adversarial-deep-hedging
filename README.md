# Adversarial Deep Hedging

> A GAN-augmented deep reinforcement learning framework for options hedging on NIFTY derivatives, implementing the CVaR-minimizing deep hedger of [Šiška & Szpruch (2023)](https://arxiv.org/pdf/2307.13217) with a GAN-based adversarial scenario generator.

---

## Overview

Classical options hedging relies on the Black-Scholes (BSM) model, which assumes constant volatility, continuous rebalancing, and no transaction costs — assumptions that systematically break down in real markets. This project implements an **adversarial deep hedging** pipeline that:

1. **Learns market dynamics** from real NIFTY minute-level futures and options data.
2. **Generates adversarial price paths** using a Generative Adversarial Network (GAN) trained on real price trajectories, producing harder and more realistic training scenarios than Geometric Brownian Motion (GBM) alone.
3. **Trains a deep hedger** — a feedforward neural network that sequentially predicts optimal delta positions — using **CVaR (Conditional Value at Risk)** as the loss, directly optimizing tail-risk rather than mean squared hedging error.
4. **Compares four scenarios** in a 2×2 evaluation matrix: Deep Hedger vs. BSM Delta Hedger, each evaluated on GBM-generated and GAN-generated test paths.

The project is inspired by and implements ideas from:

> **Šiška, J. & Szpruch, L. (2023).** *Robust Deep Hedging*. [arXiv:2307.13217](https://arxiv.org/pdf/2307.13217)

---

## Architecture

```
Real NIFTY Data (minute OHLCV)
        │
        ▼
┌───────────────────┐      ┌─────────────────────────────┐
│  Black-Scholes    │      │       GAN Training           │
│  Greeks & IV      │      │  Generator  ←→  Discriminator│
│  (bisection IV)   │      │  (price path synthesis)      │
└────────┬──────────┘      └────────────┬────────────────┘
         │                              │
         ▼                              ▼
┌──────────────────────────────────────────────────────┐
│               Deep Hedger (RNN-style)                │
│  Input: [S_t, BS_delta_t, delta_{t-1}]              │
│  Output: delta_t  (hedge position)                   │
│  Loss: CVaR_α (α = 5%)                              │
└──────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────┐
│              2×2 Evaluation                          │
│   ┌───────────────┬───────────────┐                  │
│   │  RNN | GBM   │  BSM | GBM   │                   │
│   ├───────────────┼───────────────┤                  │
│   │  RNN | GAN   │  BSM | GAN   │                   │
│   └───────────────┴───────────────┘                  │
│   Metrics: Mean P&L, Std Dev, VaR₅%, CVaR₅%         │
└──────────────────────────────────────────────────────┘
```

---

## Mathematical Framework

### Hedger P&L (from the paper)

The total P&L for a hedger who sold an option with payoff $Z(S)$ is:

$$\text{PL}_T(Z, S, \delta) := -Z(S_T) + (\delta \cdot S)_T - C_T(\delta)$$

where the **discrete trading gain** is:

$$(\delta \cdot S)_T := \sum_{i=0}^{n-1} \delta_{t_i} \left( S_{t_{i+1}} - S_{t_i} \right)$$

and the **transaction cost** (proportional to rebalancing size) is:

$$C_T(\delta) := \sum_{i=0}^{n} c \cdot S_{t_i} \left| \delta_{t_i} - \delta_{t_{i-1}} \right|, \quad \delta_{-1} = \delta_{t_n} = 0$$


### CVaR Loss

Rather than minimising mean squared hedging error, the deep hedger directly minimises **Conditional Value at Risk** at level $\alpha$:

$$\text{CVaR}_\alpha = -\mathbb{E}\left[\text{PL}_T \mid \text{PL}_T \leq \text{VaR}_\alpha\right]$$

This focuses learning on the worst-case tail of the P&L distribution — precisely where risk management matters most.

---

## Key Components

### 1. Real Data Pipeline
- Loads minute-level NIFTY futures and options data from CSV
- Parses option symbols to extract strike price and type (CE/PE)
- Computes time-to-expiration and filters to a single trading day

### 2. Implied Volatility Solver
- Bisection-method IV solver (100 iterations, tolerance $10^{-5}$)
- Applied to each option at each timestamp
- Feeds the realised IV surface into the delta hedging baseline

### 3. Black-Scholes Greeks Engine
- Closed-form Delta, Gamma, Theta for European calls and puts
- Used both as a comparison benchmark and as an input feature to the deep hedger (`BS_delta_t`)

### 4. GBM Simulator
- Generates 1,000 synthetic price paths under the risk-neutral measure ($\mu = r$)
- Parameters: $\sigma = 0.6$, $r = 0.05$, initialised at the real opening futures price
- 80/20 train/test split

### 5. GAN — Adversarial Scenario Generator
- **Generator**: 3-layer MLP (noise → 64 → 128 → `price_path_length`), linear output
- **Discriminator**: 3-layer MLP (path → 128 → 64 → 1), sigmoid output
- Trained with binary cross-entropy for 2,000 epochs on the real price path (normalised to $[-1, 1]$)
- Generates 1,000 paths post-training for adversarial training of the deep hedger

### 6. Deep Hedger
- **Architecture**: Single-step MLP with inputs `[S_t, BS_delta_t, δ_{t-1}]` → Dense(32, tanh) → Dense(1, tanh)
- **State**: Previous delta $\delta_{t-1}$ is fed back at each step, enabling sequential decision-making
- **Optimiser**: Adam, lr = 0.001
- **Loss**: CVaR at $\alpha = 5\%$ over a batch of simulated P&L values
- Trained separately on GBM paths (100 epochs) and GAN paths (30 epochs)

---

## Getting Started

### Requirements

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn tensorflow
```

Tested with Python 3.10, TensorFlow 2.13.

### Data Format

Upload a CSV file with minute-level NIFTY data containing at minimum:

| Column | Description |
|---|---|
| `date` | Date as integer `YYYYMMDD` |
| `minute_end` | Time as integer `HHMMSS` |
| `symbol` | Option/futures symbol string (e.g. `NIFTY2621025700CE`, `NIFTYFUT`) |
| `last_trade_price` | Last traded price |

### Running the Notebook

The notebook is designed for **Google Colab**. Open it, upload your CSV when prompted, and run cells sequentially. Key parameters to configure at the top:

```python
expiration_date = pd.to_datetime('2026-02-05 15:30:00')  # Expiry datetime
r = 0.05            # Risk-free rate
sigma = 0.6         # GBM volatility
num_paths = 1000    # Synthetic paths for GBM
num_gan_paths = 1000
gan_epochs = 2000
epochs = 100        # Deep hedger training epochs (GBM)
```

---

## Results & Evaluation

After training, the notebook produces:

- **Volatility smile** (IV vs. strike at 11 AM snapshot)
- **Delta, Gamma, Theta** curves from the real options chain
- **Simulated GBM paths** and **GAN-generated paths** visualisation
- **Implied volatility and delta** time series from the real-data delta hedging simulation
- **2×2 P&L distribution histograms** with KDE for all four hedging scenarios
- **Risk metrics table** reporting Mean P&L, Std Dev, VaR₅%, and CVaR₅% per scenario

The 2×2 comparison reveals whether the RNN hedger trained on adversarial GAN paths is more robust to tail events than the classic BSM benchmark — the central empirical question of this project.

---

## Project Structure

```
.
├── hedging.ipynb     # Main notebook (Colab-ready)
└── README.md
```

---

## License

MIT License. See `LICENSE` for details.
