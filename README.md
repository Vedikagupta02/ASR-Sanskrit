Sanskrit Speech Recognition
Character-Level ASR using CNN–BiLSTM + Hybrid CTC–Attention Model

This repository presents an end-to-end Automatic Speech Recognition (ASR) system for Sanskrit, a low-resource and highly inflectional language.
The model integrates a CNN–BiLSTM encoder with a Hybrid CTC + Attention decoder, enabling both strong alignment (CTC) and contextual understanding (Attention).

🚀 Overview

Architecture: CNN + BiLSTM Encoder with CTC + Attention Decoder

Acoustic Features: 240-dim log-Mel spectrograms

Augmentation: SpecAugment (time & frequency masking)

Decoding: Beam Search + 3-gram KenLM Language Model

Loss Function: Hybrid α·CTC + (1−α)·Cross-Entropy

Dataset: 3,600+ Sanskrit audio–text pairs

Metrics: Word Error Rate (WER), Character Error Rate (CER)

🧠 Model Architecture
Component	Description
Encoder	CNN frontend + 2-layer BiLSTM
Decoder	Attention-based LSTM
Loss	Hybrid CTC + Cross-Entropy
Language Model	3-gram KenLM integrated into beam search
Optimization	AdamW, Gradient Clipping, Mixed Precision
Framework	PyTorch (GPU-accelerated)
📊 Results (Beam Search + LM)
Metric	Score
Word Error Rate (WER)	0.418
Character Error Rate (CER)	0.103

Using a character-level 3-gram LM significantly improved decoding stability and reduced substitution errors.

🔑 Key Features

✅ Hybrid CTC + Attention end-to-end training

✅ Character-level decoding → strong generalization for low-resource languages

✅ SpecAugment for robustness

✅ Beam Search + KenLM for improved decoding

✅ Extensible to BPE/subwords and Transformer/Conformer encoders

✅ Clean, modular PyTorch implementation
