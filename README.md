# Urban Traffic Flow Forecasting Engine

> A production-grade **Spatio-Temporal Graph Neural Network (STGNN)** for real-time urban traffic forecasting. Built to power city logistics platforms, adaptive routing systems, and smart-infrastructure dashboards.

---

## Overview

Modern cities generate vast streams of sensor telemetry from road networks — yet most traffic management systems still rely on heuristics and lagging indicators. This engine flips that model.

Using a graph-structured representation of the road network and a hybrid deep learning architecture, it forecasts traffic flow at every sensor node **12 steps (60 minutes) ahead** with sub-minute latency — enabling logistics operators and city planners to act on predictions, not reactions.

**Key capabilities:**
- Multi-horizon forecasting (configurable look-back and prediction windows)
- Spatial dependency modeling across arbitrary road-network topologies
- Temporal pattern capture at both short-term and long-range intervals
- Plug-and-play compatibility with the METR-LA sensor dataset and `.npz`-formatted equivalents

---

## Architecture

The model combines two complementary mechanisms to capture the full spatio-temporal dynamics of road networks:

### Spatial: Graph Convolution
The road network is modeled as a graph where each sensor is a node and adjacency encodes physical connectivity. A learnable spatial embedding captures long-range dependencies that a fixed adjacency matrix would miss — adapting over training to reflect how congestion propagates across non-adjacent corridors.

### Temporal: Gated Recurrent Units (GRU)
Sequence-to-sequence GRU layers process each node's historical flow readings, learning cyclical patterns (rush hours, weekday/weekend effects) and short-burst anomalies. Attention heads weight the most informative time steps in each prediction window.

### Fusion
Each spatio-temporal block interleaves graph convolution passes with GRU hidden-state updates, ensuring spatial and temporal signals co-evolve rather than being processed in separate silos.

```
Input Sequence (T × N × F)
        │
        ▼
┌───────────────────────┐
│  Spatial Embedding    │  ← Learned node positional encoding
│  (Graph Attention)    │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  GRU Temporal Encoder │  ← Multi-head attention over time steps
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Output Projection    │  → Forecast (H × N × F)
└───────────────────────┘
```

---

## Tech Stack

| Layer | Library / Tool |
|---|---|
| Deep Learning Framework | PyTorch ≥ 1.12 |
| Graph Neural Network Ops | PyTorch Geometric |
| Data Manipulation | NumPy, Pandas |
| Progress Monitoring | tqdm |
| Hardware Acceleration | CUDA (auto-detected, falls back to CPU) |
| Data Format | `.npz` (traffic flow), `.npy` (spatial embeddings) |

---

## Dataset

This project is validated on the **METR-LA** dataset — 207 traffic sensors on the Los Angeles highway network, sampled at 5-minute intervals over 4 months.

| Property | Value |
|---|---|
| Sensors | 207 |
| Time steps | 34,272 |
| Sample interval | 5 minutes |
| Feature | Traffic flow (vehicles/hour) |

Download the pre-processed data files and place them in a `data/` directory:

```
data/
├── metr-la.npz       # Traffic flow tensor
└── metr-la-se.npy    # Spatial node embeddings
```

---

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional but recommended)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/SiddGud/Traffic_Flow_Forecasting_Engine.git
cd Traffic_Flow_Forecasting_Engine

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install numpy pandas tqdm
```

---

## Training

```bash
python train.py \
  --traffic_file data/metr-la.npz \
  --SE_file data/metr-la-se.npy \
  --P 12 \
  --Q 12 \
  --L 1 \
  --K 4 \
  --d 16 \
  --batch_size 16 \
  --max_epoch 50 \
  --learning_rate 0.001 \
  --model_file checkpoints/stgnn_metrla \
  --log_file logs/training.log
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--P` | `12` | Number of historical time steps used as input (look-back window) |
| `--Q` | `12` | Number of future time steps to forecast |
| `--L` | `1` | Number of Spatio-Temporal attention blocks |
| `--K` | `4` | Number of attention heads per block |
| `--d` | `16` | Dimensionality of each attention head's output |
| `--batch_size` | `16` | Training batch size |
| `--max_epoch` | `50` | Maximum number of training epochs |
| `--learning_rate` | `0.001` | Initial learning rate (Adam optimizer) |
| `--train_ratio` | `0.6` | Fraction of data used for training |
| `--val_ratio` | `0.2` | Fraction of data used for validation |
| `--test_ratio` | `0.2` | Fraction of data held out for evaluation |

---

## Evaluation Metrics

Model performance is reported on three standard traffic forecasting metrics, computed on the held-out test set with zero-flow readings masked:

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — average magnitude of prediction error |
| **RMSE** | Root Mean Squared Error — penalises large deviations more heavily |
| **MAPE** | Mean Absolute Percentage Error — scale-independent relative accuracy |

---

## Project Structure

```
.
├── model.py          # STGNN model definition (Transform attention + GRU blocks)
├── train.py          # Training loop, validation, and checkpointing
├── utils.py          # Data loading, normalisation, and evaluation metrics
├── data/             # Dataset files (not versioned)
│   ├── metr-la.npz
│   └── metr-la-se.npy
├── checkpoints/      # Saved model weights (generated at runtime)
└── logs/             # Training logs (generated at runtime)
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
