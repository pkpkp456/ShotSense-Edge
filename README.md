# Gunshot Detection on Edge Devices

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![FPGA Ready](https://img.shields.io/badge/Hardware-FPGA%20Ready-red?style=for-the-badge&logo=xilinx)
![Status](https://img.shields.io/badge/Status-Research-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

## Overview

This repository presents a deep learning–based gunshot detection system designed for real-time edge deployment and extensible toward FPGA acceleration. The system leverages pretrained audio embeddings and temporal modeling to accurately distinguish gunshot events from environmental sounds under noisy, real-world conditions.

The project emphasizes low-latency inference, compact model design, and deployment feasibility, making it suitable for smart-city surveillance, campus safety, and critical infrastructure monitoring.

## Key Contributions

* **Embedding-based Audio Classification:** Utilizes **CNN14 (PANNs)** for robust feature extraction.
* **Temporal Modeling:** Implements **BiLSTM + Attention** mechanisms to capture temporal dependencies in audio streams.
* **Edge Optimization:** Focuses on pruning and dynamic quantization for efficient inference on edge devices (e.g., Raspberry Pi).
* **FPGA-Ready Architecture:** Designed with hardware acceleration in mind for future High-Level Synthesis (HLS) integration.
* **Robust Evaluation:** Comprehensive testing on a balanced, multi-source dataset to ensure reliability in diverse environments.

## System Architecture

The high-level pipeline consists of the following stages:

1.  **Audio Acquisition:** Continuous audio capture and segmentation.
2.  **Preprocessing:** RMS filtering and log-Mel spectrogram extraction.
3.  **Feature Extraction:** CNN14 pretrained embedding extractor.
4.  **Classification:** BiLSTM + Attention layers feeding into a Dense head.
5.  **Deployment:** Pipelines optimized for Raspberry Pi and FPGA implementation.



## Dataset Summary

* **Total Samples:** ~17,746 audio clips
* **Class Balance:** Gunshot / Non-Gunshot (Balanced)
* **Data Sources:**
    * *Gunshot:* Kaggle, Mendeley, Zenodo, MAD
    * *Non-Gunshot:* UrbanSound8K, ESC-50

## Results

| Model | Accuracy | Deployment Suitability |
| :--- | :--- | :--- |
| **MFCC + LSTM** | ~96.0% | Medium |
| **YAMNet Transfer** | ~98.5% | High |
| **CNN14 + BiLSTM + Attention (Proposed)** | **~97.6%** | **Edge & FPGA Ready** |

* **Latency:** ~100–200 ms per 1-second window (Raspberry Pi)
* **Noise Robustness:** High

## Repository Structure

```text
├── Bi-LSTM Model/           # Training and inference code (Bi-LSTM + Attention)
├── Deployment On FPGA/      # FPGA design files, HLS work, and bitstreams
├── Plans/                   # Project roadmaps and advancement plans
├── Presentations/           # Project slides and final report presentations
├── MFCC_YamNet/                 # Metrics, plots, and experimental results
├── .settings/               # IDE configuration files
├── GitHub_Paper.txt         # IEEE paper draft or references
└── README.md                # Project documentation
```
---

## 📅 Enhanced Actionable Plan & Roadmap

This project follows a structured, phased approach to move from model training to FPGA deployment.

### Phase 1: Dataset & Model Training (2–4 weeks)
* **Goal:** Create a quantized, exportable model with high recall.
* **Actions:**
    * [ ] Data collection & labeling (gunshot, near-miss, background, impulsive noises).
    * [ ] Build augmentation pipeline (pitch, time, noise, reverb, mic-array delays).
    * [ ] Develop feature extraction scripts (MFCC, Mel-spec, ZCR) and save as `.npy/.npz`.
    * [ ] Train baseline models (Tiny CNN, XGBoost) and quantize to 8-bit/fixed-point.
* **Deliverables:** Cleaned dataset, augmentation scripts, trained `.tflite/.onnx` models, C-header weight files.
* **Acceptance Criteria:** Recall ≥ 0.95, FPR ≤ target, model size fits FPGA constraints.

### Phase 2: Multi-Mic Localization (1–2 weeks)
* **Goal:** Implement direction-of-arrival (DOA) estimation.
* **Actions:**
    * [ ] Simulate circular 8-mic array with configurable radius/sample rate.
    * [ ] Generate TDOA patterns and add noise/attenuation.
    * [ ] Implement GCC-PHAT TDOA and delay-and-sum sector scoring.
* **Deliverables:** Signal generator, GCC-PHAT reference implementation, test vectors for 8 sectors.
* **Acceptance Criteria:** Angular error ≤ threshold for tested SNR scenarios.

### Phase 3: Thermal IR Simulation & Fusion (1 week)
* **Goal:** Integrate visual confirmation to reduce false positives.
* **Actions:**
    * [ ] Simulate low-res thermal frames with configurable "hot blobs."
    * [ ] Implement fusion logic: *Classifier Confidence + Localization + Thermal Confirmation → Final Decision*.
* **Deliverables:** Thermal simulator, blob detector, fusion logic unit tests.
* **Acceptance Criteria:** Fusion reduces false positives compared to audio-only baseline.



### Phase 4: FPGA Module Design (2–4 weeks per block)
* **Goal:** Translate Python logic to Hardware Description Language (HLS/RTL).
* **Actions:**
    * [ ] **Audio Preproc:** Windowing, fixed-point MFCC.
    * [ ] **Classifier:** Fixed-point CNN or lookup-based model.
    * [ ] **Localization:** TDOA-lite / beamformer.
    * [ ] **Fusion & UART:** FSM for decision logic and packet formatting.
* **Deliverables:** HLS/RTL block stubs, I/O specs, bit-accurate simulations.
* **Acceptance Criteria:** Bit-accurate outputs vs. Python reference; resource estimates (LUT/BRAM) within limits.

### Phase 5: Simulation & Verification (2–3 weeks)
* **Goal:** End-to-end system verification.
* **Actions:**
    * [ ] Generate comprehensive test vectors (various SNRs, multiple events).
    * [ ] Simulate blocks via Verilator/Vivado; compare with "Golden" Python outputs.
* **Deliverables:** Testbench, automated comparison reports, CI scripts.
* **Acceptance Criteria:** Module outputs match reference; timing constraints met.

### Phase 6: Optimization & Resource Estimation (1–2 weeks)
* **Goal:** Fit the design onto the target board.
* **Actions:**
    * [ ] Prune and compact model; tune HLS pragmas (pipelining, unrolling).
    * [ ] Run synthesis to collect accurate power and resource usage data.
* **Deliverables:** Optimized model, resource/latency tables.
* **Acceptance Criteria:** Design fits target FPGA with margin; real-time throughput achieved.

### Phase 7: Communication Simulation (1 week)
* **Goal:** Validate external connectivity.
* **Actions:**
    * [ ] Define minimal packet structure (Start Byte, Event Code, Sector, Checksum).
    * [ ] Simulate TX/RX to validate integrity and latency.
* **Deliverables:** UART/SPI simulator, receiver emulator.
* **Acceptance Criteria:** Packets received correctly across simulated noise channels.

### Phase 8: Documentation & Delivery (1 week)
* **Goal:** Handover and knowledge transfer.
* **Actions:**
    * [ ] Create block diagrams, timing diagrams, and "How-to-Run" guides.
* **Deliverables:** Final repo structure, deployment checklist.
* **Acceptance Criteria:** Reproducible simulation and verification steps.

---

### 🛡️ Mission & Maintenance
**Contributing to National Defense:**
We deeply respect and encourage contributions that help lead Indian defense technology forward. This project is dedicated to strengthening indigenous security solutions and fostering innovation in edge-AI for defense.

**Project Status:**
* **Active/Periodic:** Development on this repository occurs periodically as I balance this work with other ongoing research projects. Community contributions and pull requests are highly encouraged to keep the momentum going.

---

### 🔮 Future Improvements & Research Directions
* **FPGA Acceleration:** Full HLS-based implementation of CNN14 embedding and BiLSTM inference.
* **Multi-Class Firearm Detection:** Expansion of the model to identify specific firearm categories (e.g., handgun vs. rifle).
* **Event Localization:** Real-time hardware estimation of gunshot direction and distance.
* **Model Compression:** Advanced knowledge distillation and mixed-precision inference for ultra-low-power devices.
* **Dataset Expansion:** Gathering more real-world, outdoor, and long-range recordings to reduce dataset bias.