---
title: "Whisper: Code architecture"
parent: "086 Whisper"
nav_order: 1
---
# Code Architecture — Whisper Notebook

## Overview

The notebook implements a miniature Whisper-style encoder-decoder Transformer for automatic speech recognition (ASR) in PyTorch. It trains on the Free Spoken Digit Dataset (FSDD, ~1500 clips of spoken digits 0-9), evaluates with greedy decoding, then loads the pretrained OpenAI whisper-tiny model for zero-shot comparison and a noise robustness sweep. The from-scratch model reaches 98.3% exact-match accuracy (WER 0.017) on 600 test samples; the pretrained whisper-tiny achieves 80% zero-shot on 20 samples. Under noise, both models degrade, with the tiny model maintaining higher accuracy (80% vs 67% at 40 dB SNR).

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines key constants:
- `N_MELS = 80`, `N_FFT = 400`, `HOP_LENGTH = 160` — Mel spectrogram parameters matching Whisper Section 2.2
- `D_MODEL = 128`, `N_HEAD = 4`, `N_ENC_LAYERS = 2`, `N_DEC_LAYERS = 2`, `D_FF = 512` — tiny architecture
- `EPOCHS = 30`, `LR = 3e-4`, `BATCH_SIZE = 32`, `WARMUP_STEPS = 200`

### 2. Data

Downloads FSDD from GitHub. Parses filenames (`{digit}_{speaker}_{iteration}.wav`), resamples to 16 kHz, pads/crops to 2 seconds, computes 80-channel log-Mel spectrograms with per-sample normalization.

**Class: `SpokenDigitDataset`** — returns `(log_mel, tokens, word)` where log_mel is (80, T) and tokens are character-level indices with special tokens prepended.

Splits 80/20 train/test. Builds a character-level tokenizer with 4 special tokens (`<pad>`, `<sos>`, `<eos>`, `<transcribe>`) + 24 character tokens = 28 total vocabulary.

### 3. Model

**Class: `TinyWhisper`** (about 1.04M parameters (1,041,555))

- `WhisperEncoder`: Conv1d(80, 128, k=3) + GELU, Conv1d(128, 128, k=3, stride=2) + GELU, sinusoidal positional encoding, 2 Transformer encoder layers (norm_first=True, GELU), LayerNorm. Mirrors Whisper Section 2.2.
- `WhisperDecoder`: Embedding(28, 128) with tied output weights, learned positional embeddings (Embedding(256, 128)), 2 Transformer decoder layers with self-attention + cross-attention, LayerNorm, Linear output projection tied to embedding weights.

**Data flow:**
```
Input: (B, 80, T)  Mel spectrogram
  -> Conv1d stem -> (B, 128, T//2)
  -> transpose -> (B, T//2, 128)
  -> Transformer encoder -> (B, T//2, 128)  [memory]
  -> cross-attention with decoder tokens -> (B, seq_len, 28)  [logits]
```

### 4. Loss Function

Standard cross-entropy loss with `ignore_index=PAD_IDX`. The decoder is trained with teacher forcing: input is tokens[:-1], target is tokens[1:]. AdamW optimizer (lr=3e-4, betas=(0.9, 0.98), weight_decay=0.01) with linear warmup (200 steps) then linear decay to 0.1, gradient norm clipping at 1.0.

### 5. Training Loop

30 epochs, full-batch gradient updates. Logs loss and token accuracy every 5 epochs. No data augmentation or regularization (matching Whisper Section 2.4, which relies on dataset diversity instead).

### 6. Training Curves

Two subplots: training loss (decreasing) and token accuracy (increasing) over 30 epochs.

### 7. Final Evaluation

Greedy autoregressive decoding on the test set. Reports exact-match accuracy and WER (1 - exact_match). Prints 10 sample predictions with match/no-match indicators.

### 8. Probing Pretrained Whisper

Installs and loads `openai-whisper`, loads whisper-tiny (39M parameters), runs zero-shot transcription on 20 FSDD test clips. Reports exact-match accuracy and sample predictions.

### 9. Noise Robustness Comparison

Adds Gaussian white noise at SNR levels [40, 20, 10, 5, 0, -5] dB to 30 test samples. Compares tiny model vs pretrained whisper-tiny at each level. Plots accuracy vs SNR for both models.

### 10. Confusion Analysis

Counts which digit words the tiny model confuses at the worst SNR (-5 dB), using a Counter of (reference, prediction) pairs.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `SpokenDigitDataset` | Loads FSDD audio, computes log-Mel spectrograms, returns (mel, tokens, word) |
| `PositionalEncoding` | Sinusoidal positional embeddings for the encoder |
| `WhisperEncoder` | Conv1d stem + Transformer encoder blocks for audio |
| `WhisperDecoder` | Embedding + Transformer decoder with cross-attention to encoder |
| `TinyWhisper` | Full encoder-decoder model |
| `text_to_tokens` / `tokens_to_text` | Character-level tokenization with special tokens |
| `greedy_decode` | Autoregressive greedy decoding for inference |
| `add_noise` | Adds Gaussian noise at a specified SNR (dB) |
| `compute_accuracy` | Token-level accuracy ignoring padding |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Dataset | 680,000 hours web audio | ~1500 FSDD spoken-digit clips |
| Model size | 39M-1550M parameters | ~1.04M parameters |
| Layers | 4-32 encoder/decoder | 2 encoder + 2 decoder |
| Vocabulary | GPT-2 byte-level BPE | 28 character-level tokens |
| Tasks | Transcription, translation, language ID, VAD | Transcription only |
| Decoding | Beam search + temperature fallback | Greedy decoding |
| Training | 220 updates, batch 256, FP16 | 30 epochs, batch 32, FP32 |
