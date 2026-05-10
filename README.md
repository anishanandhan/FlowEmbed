# FlowEmbed — Context-Aware Flow Embeddings for Adaptive AI-Based Network Traffic Classification

<p align="center">
  <strong>Samsung Ennovatex AX Hackathon — PS 02</strong><br>
  Cybersecurity AI · Metric Learning · Encrypted Traffic Classification · Agentic AI
</p>

---

## 🎯 Problem Statement

Traditional network traffic classification relies on **Deep Packet Inspection (DPI)**, which requires decrypting packet contents. With modern encrypted protocols (HTTPS, QUIC, TLS 1.3), DPI is rendered blind.

**FlowEmbed** classifies network traffic purely from **behavioral flow patterns** — packet timing, size distributions, inter-arrival jitter, burst rhythms — without ever decrypting content. This makes it fully **GDPR-compliant** and legally deployable in jurisdictions where Deep Packet Inspection is prohibited.

### Key Innovation: Metric Learning Flow Embeddings

We apply **contrastive metric learning** to network flow features, creating an embedding space where:
- Similar traffic types (YouTube ↔ Netflix) cluster together (cosine similarity > 0.7)
- Dissimilar traffic types (streaming ↔ gaming) actively repel each other (cosine similarity < 0.3)
- Novel traffic types can be classified with just **5 examples** (few-shot generalization)

---

## 🏗️ System Architecture

```text
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Traffic Capture │────→│  Flow Encoder    │────→│  k-NN Classifier│
│  (NFStream Live) │     │  (PyTorch MLP)   │     │  (FAISS IVFFlat)│
│  60 Features     │     │  3-Loss Strategy │     │  Voronoi Quant. │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
       ┌───────────────────────────────────────────────────┘
       │
┌──────▼──────┐     ┌──────────────────┐     ┌─────────────────┐
│ ADWIN Drift │────→│  Agentic LLM     │────→│  Flask Dashboard│
│ Detector    │     │  Alerter (Ollama)│     │  (Real-time)    │
│ (River)     │     │  XAI / SOC Agent │     │  UMAP + Alerts  │
└─────────────┘     └──────────────────┘     └─────────────────┘
```

---

## 📊 KPI Targets (Verified)

| KPI | Target | Final Status |
|-----|--------|--------------|
| Intra-class Cosine Similarity | > 0.7 | ✅ **0.925** |
| Inter-class Cosine Similarity | < 0.3 | ✅ **0.173** |
| Classification Accuracy | ≥ 90% | ✅ **91.60%** |
| XR/AR Zero-Shot Generalization | ≥ 85% | ✅ **92.4%** |
| Real-Time Latency | < 100ms/flow | ✅ **0.19ms** |

---

## 🧠 AI / GenAI / Agentic Capabilities

### Core AI Models
- **PyTorch Metric Learning Encoder** — Custom 3-layer MLP (BatchNorm + GELU + Skip connections) trained with a triple-objective loss: CenterLoss + CentroidRepulsionLoss + CrossEntropy. Maps raw 60-feature flow vectors into a 48-dimensional latent space.
- **FAISS IndexIVFFlat** — Meta's Voronoi cell quantization algorithm for production-scale, sub-millisecond k-Nearest Neighbor traffic classification.
- **River ADWIN** — Adaptive Windowing statistical algorithm for real-time concept drift detection in live 5G/6G traffic streams.

### GenAI Integration
- **Local LLM via Ollama** — Integrated a local Large Language Model (Mistral / Llama) securely via the Ollama API, ensuring complete data privacy for telecom infrastructure. No data ever leaves the edge node.
- **Explainable AI (XAI)** — The GenAI model ingests raw statistical flow features (e.g., `IAT_CV ~0.02`) and SHAP values, autonomously translating them into plain-English, human-readable SOC (Security Operations Center) alerts instead of opaque error codes.

### Agentic Pipeline
- **Autonomous SOC Analyst Agent** (`src/explainability/llm_alerter.py`) — An agentic script that operates without any human intervention:
  1. The **ADWIN algorithm** constantly monitors the traffic stream for behavioral drift.
  2. When drift is detected, the agent **autonomously isolates the anomalous flow** and packages telemetry data.
  3. The agent **independently queries the LLM** to generate a threat analysis.
  4. The agent **surfaces a dynamic "Threat Isolation" banner** on the dashboard UI — zero human intervention required.

---

## 📁 Project Structure

```text
FlowEmbed/
├── dashboard/                      # Flask web dashboard
│   ├── app.py                      # Backend API (KPIs, UMAP, classify, agentic alert)
│   ├── static/style.css            # Dark glassmorphic UI styling
│   └── templates/index.html        # Single-page dashboard with live visualization
│
├── src/                            # Core source modules
│   ├── models/
│   │   ├── encoder.py              # PyTorch MLP Metric Learning Encoder
│   │   ├── losses.py               # CenterLoss + CentroidRepulsionLoss + NT-Xent
│   │   └── projection_head.py      # Projection head for contrastive learning
│   ├── classifier/
│   │   ├── knn_classifier.py       # FAISS IndexIVFFlat k-NN classifier
│   │   ├── svm_classifier.py       # SVM baseline classifier
│   │   └── evaluator.py            # Cross-validation evaluation pipeline
│   ├── data/
│   │   ├── dataset.py              # Data loading and split management
│   │   ├── preprocessing.py        # Feature scaling and normalization
│   │   ├── nfstream_plugin.py      # Custom NFStream plugin (full 60-feature extraction)
│   │   ├── pcap_processor.py       # Raw PCAP → flow-level feature conversion
│   │   └── augmentations.py        # Flow-level data augmentations
│   ├── drift/
│   │   ├── drift_detector.py       # ADWIN drift detection (River)
│   │   └── online_updater.py       # Online model update on drift
│   ├── explainability/
│   │   ├── llm_alerter.py          # Autonomous LLM-based SOC alert agent
│   │   ├── shap_explainer.py       # SHAP feature importance analysis
│   │   └── umap_visualizer.py      # UMAP 2D embedding projection
│   ├── api/
│   │   ├── app.py                  # Production REST API
│   │   ├── routes.py               # API route definitions
│   │   └── websocket.py            # WebSocket for live streaming
│   └── config.py                   # Global paths and hyperparameters
│
├── scripts/                        # Executable pipelines
│   ├── process_live.py             # Samsung S23 PCAP → 60-feature → prediction
│   ├── process_pcaps.py            # Raw CSV/PCAP → flow-level features
│   ├── merge_datasets.py           # Unified 4-dataset cross-validation merger
│   ├── train_final.py              # Final production model training
│   ├── evaluate.py                 # Full evaluation with confusion matrix
│   ├── eval_xr_ar.py               # XR/AR zero-shot generalization evaluation
│   ├── few_shot_xrar.py            # 5-shot few-shot learning evaluation
│   └── live_inference.py           # Real-time interface sniffing (en0/lo0)
│
├── notebooks/                      # Dataset processing notebooks
│   ├── kaggle_process_5g.py        # Kaggle 5G dataset processor
│   ├── kaggle_process_cesnet.py    # CESNET QUIC dataset processor
│   ├── kaggle_process_andmal.py    # Android Malware dataset processor
│   ├── kaggle_process_iscx.py      # ISCX VPN dataset processor
│   └── kaggle_process_mirage.py    # Mirage mobile traffic processor
│
├── data/                           # Data directory (not tracked in git)
│   ├── raw/                        # Raw PCAPs and CSVs
│   ├── processed/                  # Flow-level CSVs
│   ├── splits/                     # train/val/test splits, scaler, label encoder
│   ├── embeddings/                 # FAISS index, train/test embeddings
│   └── checkpoints/                # Model weights (.pt files)
│
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 📦 Datasets Used

| Dataset | Traffic Types | Source |
|---------|--------------|--------|
| **Kaggle 5G** | Streaming, Gaming, VoIP | Kaggle public dataset |
| **CESNET QUIC** | QUIC-encrypted web traffic | Czech Education Network |
| **ISCX VPN** | VPN tunneled traffic | University of New Brunswick |
| **AndMal** | Android malware C2 traffic | Android Malware Dataset |
| **Mirage** | Mobile app traffic (WhatsApp, YouTube, etc.) | Mirage dataset |
| **Samsung S23 Ultra** | Live 5G capture (Gaming, Streaming, VoIP) | Custom capture |

**Training methodology:** Unified 80/20 stratified global cross-validation across all datasets.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- macOS / Linux
- (Optional) Ollama for LLM-powered alerts

### Installation

```bash
# Clone
git clone https://github.com/anishanandhan/FlowEmbed.git
cd FlowEmbed

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard

```bash
# Start the Flask UI and classification API
python dashboard/app.py
# Open http://127.0.0.1:5055
```

### Processing a Custom PCAP (Samsung S23)

```bash
# Process a PCAP file captured from a mobile device
python scripts/process_live.py path/to/your/capture.pcapng

# Or use the default Samsung S23 capture
python scripts/process_live.py
```

### Running Live Real-Time Inference

```bash
# Sniff on your Mac's WiFi interface
sudo python scripts/live_inference.py --interface en0

# Test on local loopback
python scripts/live_inference.py --interface lo0
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Pipeline | NFStream (C-based), pandas, NumPy |
| Feature Extraction | Custom NFStream Plugin (60 features: bursts, entropy, first-N packets) |
| Model Training | PyTorch (CenterLoss + CentroidRepulsionLoss + CrossEntropy) |
| Classification | FAISS IndexIVFFlat (Voronoi cell quantization), scikit-learn |
| Drift Detection | River (ADWIN algorithm) |
| GenAI / XAI | Local LLM via Ollama API (Mistral/Llama) + SHAP |
| Agentic AI | Autonomous SOC Analyst Agent (drift → isolate → query LLM → alert) |
| Visualization | UMAP (2D embedding projection), Plotly.js |
| Backend & UI | Flask REST API, HTML/CSS/JS (dark glassmorphic theme) |

---

## 📱 Samsung S23 Ultra Integration

We validated our system live on a **Samsung Galaxy S23 Ultra** — generating real 5G traffic across gaming, streaming, and VoIP apps. This demonstrates deployment-readiness at Samsung's own network edge, from handset to base station.

**Wireshark capture filters used for validation:**
| Traffic Type | Filter |
|-------------|--------|
| Streaming | `quic \|\| (tcp.port == 443 && frame.len > 1000)` |
| Gaming | `udp && frame.len < 300` |
| VoIP | `udp && (frame.len > 50 && frame.len < 200)` |

**Live prediction results (samsung122.pcapng):**
| Class | Flows | Percentage |
|-------|-------|------------|
| Streaming | 145 | 48.7% |
| Gaming | 90 | 30.2% |
| VoIP | 62 | 20.8% |
| Malware | 1 | 0.3% |

---

## 📊 Dataset
The dataset used in this project is officially published on IEEE DataPort:
[5G Network Traffic Dataset from Samsung Galaxy Mobile](https://dx.doi.org/10.21227/xxg9-ry60)


## 📄 License

Apache License 2.0 — See [LICENSE](LICENSE)

---

*Built for Samsung Ennovatex AX Hackathon by Anish Anandhan*
