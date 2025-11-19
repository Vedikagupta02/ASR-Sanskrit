# Sanskrit Speech Recognition
## Character-Level ASR using CNN–BiLSTM + Hybrid CTC–Attention Model

This repository presents an end-to-end Automatic Speech Recognition (ASR) system for Sanskrit, a low-resource and highly inflectional language.
The model integrates a CNN–BiLSTM encoder with a hybrid CTC + Attention decoder, enabling both strong alignment (CTC) and contextual understanding (Attention).

---

## Overview

* **Architecture:** CNN + BiLSTM Encoder with CTC + Attention Decoder
* **Acoustic Features:** 240-dim log-Mel spectrograms
* **Augmentation:** SpecAugment (time & frequency masking)
* **Decoding:** Beam Search + 3-gram KenLM Language Model
* **Loss Function:** Hybrid $\alpha \cdot \text{CTC} + (1 - \alpha) \cdot \text{Cross-Entropy}$
* **Dataset:** 3,600+ Sanskrit audio–text pairs
* **Metrics:** Word Error Rate (WER), Character Error Rate (CER)

---

## Model Architecture

| Component | Description |
| :--- | :--- |
| Encoder | CNN frontend + 2-layer BiLSTM |
| Decoder | Attention-based LSTM |
| Loss | Hybrid CTC + Cross-Entropy |
| Language Model | 3-gram KenLM integrated into beam search |
| Optimization | AdamW, Gradient Clipping, Mixed Precision |
| Framework | PyTorch (GPU-accelerated) |

---

## Results (Beam Search + LM)

| Metric | Score |
| :--- | :--- |
| Word Error Rate (WER) | 0.418 |
| Character Error Rate (CER) | 0.103 |

> Using a character-level 3-gram Language Model significantly improved decoding stability and reduced substitution errors.

---

## Key Features

* Hybrid CTC + Attention end-to-end training
* Character-level decoding -> strong generalization on low-resource languages
* SpecAugment for time & frequency robustness
* Beam Search + KenLM LM for improved sequence decoding
* Extensible to BPE / subword tokenization or transformer-based encoders
* Clean, modular PyTorch implementation
