# EntropyX – Quantum-Inspired Cyber Forensics Toolkit
  EntropyX is a Python tool that analyzes and visualizes entropy patterns in files to detect hidden encrypted data, obfuscation, and anomalies. It combines classical Shannon entropy with quantum-inspired estimation, supports chunk-based and sliding window analysis, provides interactive plots and heatmaps, and exports results for forensic reporting. Fully offline and lightweight, EntropyX bridges quantum concepts and digital forensics in a practical, hands-on toolkit

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Qiskit](https://img.shields.io/badge/Qiskit-Optional-purple)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📘 Table of Contents

1. [Overview](#-project-overview)
2. [Features](#-features)
3. [Quantum Simulation](#-quantum-simulation)
4. [Visualization](#-visualization)
5. [Use Case: Cyber Forensics](#-use-case-cyber-forensics)
6. [Technologies Used](#-technologies-used)
7. [Installation & Setup](#-installation--setup)
8. [File Structure](#-file-structure)
9. [Screenshots](#-screenshots)
10. [Forensic Value](#-forensic-value)
11. [Future Improvements](#-future-improvements)
12. [Author](#-author)
13. [License](#-license)

---

## 🔍 Project Overview

*EntropyX* is a Python + Streamlit toolkit for *cyber forensic investigations, designed to profile **entropy distribution* across files using both *classical* and *quantum-inspired* methods.

It helps detect suspicious file regions that may indicate:
- Encryption or compression  
- Obfuscation or steganography  
- Data tampering or hidden payloads  

> 🧠 Built for education and demonstration — showing how quantum-inspired algorithms can enhance modern digital forensics.

---

## 🚀 Features

### ✅ Core Functionality

- Upload any *binary or text file* and scan for entropy anomalies  
- *Chunk-based entropy analysis* with adjustable segment size  
- Entropy estimation modes:
  - 📘 *Classical Shannon Entropy*
  - 🧪 *Quantum-inspired Entropy Proxy* (Qiskit or fallback mode)
- 📊 Interactive line plots and heatmaps  
- ⚠ *Anomaly detection*: highlights entropy spikes  
- 💾 Save/load profiles in *SQLite database*  
- 📤 Export results as *JSON forensic reports*

---

## 🧪 Quantum Simulation

When available, *Qiskit* powers a simulated quantum backend that:
- Encodes file segments into qubit superpositions  
- Applies *Hadamard* transformations to extract interference-based entropy  
- Produces a *quantum-derived entropy signature* for each file region  

> 🔁 If Qiskit isn’t installed, EntropyX automatically falls back to a *pseudo-quantum estimator* using randomized classical approximations.

---

## 📊 Visualization

- Comparative *entropy line plot* (classical vs quantum)  
- *Heatmap* visualization of entropy across file regions  
- Adjustable *thresholds* for custom anomaly sensitivity  

---

## 🔄 Sliding Window Analysis

- Supports *sliding window sampling* (overlapping chunks)  
- Detects subtle, localized entropy variations  
- Ideal for identifying *partial encryptions* or *hidden payloads*

---

## 💼 Use Case: Cyber Forensics

EntropyX assists analysts in identifying and visualizing hidden data zones in digital evidence.

| Scenario | Application |
|-----------|--------------|
| 🔐 Hidden Payloads | Detect encrypted or obfuscated file regions |
| 🦠 Malware Analysis | Locate polymorphic or packed binaries |
| 🧩 Data Tampering | Highlight irregular entropy distributions |
| 🧾 Evidence Profiling | Export entropy maps for forensic documentation |

---

## 🧠 Technologies Used

| Technology | Purpose |
|-------------|----------|
| *Python* | Core language |
| *Streamlit* | Web-based interface |
| *Qiskit* | Quantum circuit simulator (optional) |
| *Numpy* | Numerical computation |
| *Matplotlib* | Entropy visualization |
| *SQLite* | Persistent profile database |
| *JSON* | Forensic export format |

---
## 🛠 Installation & Setup

### 🔧 Requirements

- Python 3.8+  
- Virtual environment recommended  

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
````

If issues arise with Qiskit or Numpy:

```bash
pip install --only-binary=:all: numpy
pip install qiskit-terra qiskit-aer
```

### ▶ Run EntropyX

```bash
streamlit run app.py
```

---

## 🗂 File Structure

```
EntropyX/
├── app.py                # Streamlit UI
├── quantum_entropy.py    # Entropy analysis logic
├── profiles.db           # SQLite database (auto-created)
├── requirements.txt
└── README.md
```

---

## 🛡 Forensic Value

| Feature                 | Benefit                              |
| ----------------------- | ------------------------------------ |
| Quantum Proxy Entropy   | Simulated next-gen forensic analysis |
| Sliding Window Sampling | Detects fine-grained payloads        |
| Local Database          | Case management and storage          |
| JSON Export             | Portable, auditable evidence         |
| Offline Operation       | 100% local and secure                |

---

## 🧭 Future Improvements

* 🔍 Integrate YARA pattern matching
* 🧠 Connect to real IBM Quantum hardware
* 🧠 Add memory dump analysis support
* 📝 Include chain-of-custody metadata
* 📄 Auto-generate PDF forensic reports

---

## 👨‍💻 Author

**Adhiyaman Babu**

Cybersecurity & Quantum Computing Enthusiast
🌐 [GitHub](#) · 💼 [LinkedIn](#)

---

## 📜 License

This project is open source under the **MIT License**.
