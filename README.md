#  Sanskrit Speech Recognition  
### Character-Level ASR using CNN–BiLSTM + CTC–Attention Hybrid Model  

This project implements an **end-to-end Automatic Speech Recognition (ASR)** system for **Sanskrit**, a low-resource and morphologically rich language.  
The model combines **Convolutional** and **Bidirectional LSTM** layers with a **hybrid CTC + Attention** decoding approach for robust alignment and contextual understanding.

---

##  Overview  
- **Architecture:** CNN + BiLSTM Encoder with CTC + Attention Decoder  
- **Features:** 240-dim log-Mel spectrograms (with delta + delta-delta)  
- **Augmentation:** SpecAugment (time & frequency masking)  
- **Loss Function:** Weighted combination of CTC and Cross-Entropy  
- **Dataset:** 3,600+ aligned Sanskrit audio–text pairs  
- **Metrics:** Word Error Rate (WER), Character Error Rate (CER)

---

##  Model Highlights  
| Component | Description |
|------------|-------------|
| **Encoder** | 1D CNN frontend + 2-layer BiLSTM |
| **Decoder** | Attention-based LSTM |
| **Loss** | Hybrid (α·CTC + (1−α)·CE) |
| **Optimization** | AdamW + gradient clipping + mixed precision |
| **Framework** | PyTorch (GPU-optimized) |

---

##  Results  
| Metric | Score |
|---------|--------|
| **Word Error Rate (WER)** | **0.69** |
| **Character Error Rate (CER)** | **0.155** |

> Achieved strong recognition accuracy on a **low-resource Sanskrit dataset** with limited speaker diversity.

---

##  Key Features  
✅ End-to-end **CTC + Attention** hybrid training  
✅ **Character-level decoding** for better generalization  
✅ **SpecAugment** for noise and time variation robustness  
✅ **Greedy decoding** for fast inference  
✅ Extensible to **beam search + language model (KenLM)**  

---

## 📂 Directory Structure  
