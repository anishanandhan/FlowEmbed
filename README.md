# FlowEmbed — Context-Aware Flow Embeddings for Adaptive AI-Based Network Traffic Classification

<p align="center">
  <strong>Samsung Ennovatex AX Hackathon — PS 02</strong><br>
  Cybersecurity AI · Contrastive Learning · Encrypted Traffic Classification
</p>

---

## 🎯 Problem Statement

Traditional network traffic classification (DPI — Deep Packet Inspection) requires decrypting packet contents. With modern encrypted protocols (HTTPS, QUIC, TLS 1.3), DPI is blind. **FlowEmbed** classifies network traffic purely from behavioral flow patterns — packet timing, size distributions, RTT, jitter — without ever decrypting content.

### Key Innovation: Contrastive Flow Embeddings

We apply **contrastive learning (SimCLR / NT-Xent)** to network flow features, creating an embedding space where:
- Similar traffic types (YouTube ↔ Netflix) cluster together (cosine similarity > 0.7)
- Dissimilar traffic types (streaming ↔ gaming) actively repel each other (cosine similarity < 0.3)
- Novel traffic types can be classified with just 5 examples (few-shot generalization)

Crucially, this behavioral analysis requires zero payload decryption — making FlowEmbed fully GDPR-compliant and legally deployable in jurisdictions where Deep Packet Inspection is prohibited.

## 🏗️ System Architecture

```text
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Traffic Capture │────→│  Flow Encoder    │────→│  k-NN Classifier│
│  (NFStream Live) │     │  (PyTorch MLP)   │     │  (FAISS Index)  │
│                  │     │  3-Loss Strategy │     │                 │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
       ┌───────────────────────────────────────────────────┘
       │
┌──────▼──────┐     ┌──────────────────┐     ┌─────────────────┐
│ ADWIN Drift │────→│  LLM Alerter     │────→│  Flask Dashboard│
│ Detector    │     │  (Ollama/Mistral)│     │  (Real-time)    │
│ (River)     │     │                  │     │  UMAP + Alerts  │
└─────────────┘     └──────────────────┘     └─────────────────┘
```

## 📊 KPI Targets (Verified)

| KPI | Target | Final Status |
|-----|--------|--------|
| Intra-class Cosine Similarity | > 0.7 | ✅ **0.925** |
| Inter-class Cosine Similarity | < 0.3 | ✅ **0.173** |
| Classification Accuracy | ≥ 90% | ✅ **91.60%** |
| Generalization (new types) | ≥ 85% | ✅ **85.31%** |
| Real-Time Latency | < 100ms/flow | ✅ **0.19ms** |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- macOS / Linux
- (Optional) Ollama for LLM alerts

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

### Running Live Real-Time Inference

```bash
# To sniff on your Mac's WiFi interface
sudo python scripts/live_inference.py --interface en0

# To test it safely on your local loopback
python scripts/live_inference.py --interface lo0
```

### Running the Dashboard

```bash
# Start the Flask UI and classification API
python dashboard/app.py
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Pipeline | NFStream (C-based), pandas, NumPy, tcpreplay |
| Model Training | PyTorch, CenterLoss, CentroidRepulsionLoss, CrossEntropy |
| Classification | FAISS (IndexIVFFlat), scikit-learn |
| Drift Detection | River (ADWIN algorithm) |
| Explainability | Local LLM via Ollama API |
| Backend & UI | Flask, HTML/JS, UMAP |

## 📱 Samsung S23 Ultra Integration

We validated our system live on a Samsung Galaxy S23 Ultra — generating real 5G traffic across gaming, streaming, and VoIP. This demonstrates deployment-readiness at Samsung's own network edge, from handset to base station.

Custom mobile traffic dataset captured via S23 Ultra hotspot:
- YouTube 4K, Instagram Reels, Mobile Gaming (BGMI)
- WhatsApp Video Call, Google Maps, Background Idle

## 📄 License

Apache License 2.0 — See [LICENSE](LICENSE)

---

*Built for Samsung Ennovatex AX Hackathon by Anish Anandhan*
