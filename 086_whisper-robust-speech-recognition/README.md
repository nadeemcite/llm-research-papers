---
title: "086 Whisper"
nav_order: 87
permalink: "/086_whisper-robust-speech-recognition/"
---
# Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/whisper-robust-speech-recognition)

- **Paper:** [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- **Authors:** Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever
- **Lab, year:** OpenAI, San Francisco, 2022 (ICML 2023)

## Summary

Whisper is a speech recognition system trained on 680,000 hours of audio paired with transcripts scraped from the internet. Unlike prior approaches that used self-supervised pre-training on unlabeled audio and then required fine-tuning on a specific dataset, Whisper simply learns to predict transcripts directly from audio. The transcripts are noisy (many come from YouTube auto-captions and similar sources), but the sheer scale compensates for the noise.

The result is a model that works well on new datasets without any fine-tuning — a property called zero-shot transfer. On standard benchmarks, Whisper's large model achieves a word error rate (WER, the fraction of words a recognizer gets wrong) of 2.5% on LibriSpeech test-clean, matching human performance, and 12.8% average across 12 other datasets — far better than LibriSpeech-trained models that reach 29.3% average on the same out-of-distribution sets (Table 2). The paper also shows Whisper approaches human-level accuracy and robustness, especially under additive noise.

This repo's notebook builds a tiny Whisper-style encoder-decoder Transformer from scratch, trains it on the Free Spoken Digit Dataset, and probes the pretrained whisper-tiny model for zero-shot comparison and noise robustness.

## What Problem Does It Solve

Imagine you learned to recognize speech only from crystal-clear audiobook recordings. You'd do great in a quiet library but stumble in a noisy cafe, on a phone call, or with someone who has a different accent. That's exactly what happened to speech recognition models trained on clean, curated datasets like LibriSpeech (1,000 hours of read audiobooks): they hit "superhuman" accuracy on their training distribution but fell apart on real-world audio.

Whisper's insight is simple: instead of obsessing over data quality, use quantity. The paper collects 680,000 hours of audio-transcript pairs from across the internet — noisy, diverse, multilingual — and trains a standard encoder-decoder Transformer to predict transcripts. The diversity of the data, not clever architectural tricks, is what makes the model robust to accents, background noise, and recording conditions it never saw during training.

## Core Idea

- **Weak supervision at scale:** Instead of 1,000 hours of carefully labeled audio, use 680,000 hours of noisily labeled audio from the web. The labels are imperfect, but the volume and diversity teach the model to generalize.
- **Off-the-shelf architecture:** Whisper deliberately uses a standard encoder-decoder Transformer (Vaswani et al., 2017) with no novel architectural components, to avoid confounding the study of scale with architecture improvements.
- **Multitask format:** A single model handles transcription, translation (X to English), language identification, and voice activity detection via special tokens prepended to the decoder output, replacing the multi-stage pipeline of traditional speech systems.

## Key Method Details

- **Architecture:** Encoder-decoder Transformer. Audio is converted to 80-channel log-Mel spectrograms (25ms windows, 10ms stride), processed through a 2-layer Conv1d stem (GELU, stride-2), then Transformer encoder blocks. The decoder uses learned positional embeddings and tied input-output embeddings (Section 2.2).
- **Model sizes:** Tiny (39M, 4 layers), Base (74M, 6 layers), Small (244M, 12 layers), Medium (769M, 24 layers), Large (1550M, 32 layers) (Table 1).
- **Training:** AdamW with linear warmup (2048 steps) then decay to zero, batch size 256, FP16 with dynamic loss scaling, 220 updates (~2-3 epochs), gradient norm clipping (Section 2.4).
- **Dataset:** 680,000 hours total: 438,000 hours English transcription, 117,000 hours multilingual (96 other languages), 125,000 hours X-to-English translation (Section 2.1).
- **Results:** Large-v2 achieves 2.5% WER on LibriSpeech clean-test (zero-shot), 12.8% average WER across 12 other datasets (Table 2), and 29.1 BLEU on CoVoST2 X-to-English translation (Table 4).

### Simplifications in This Notebook

- Smaller model: 2 encoder + 2 decoder layers, D_MODEL=128, about 1.04M parameters (paper's Tiny is 39M)
- Dataset: ~1500 Free Spoken Digit Dataset clips instead of 680,000 hours of web audio
- Character-level vocabulary (28 tokens) instead of GPT-2 byte-level BPE
- Transcription only, not the full multitask format (no translation, language ID, or VAD)
- Greedy decoding only, no beam search or temperature fallback scheduling
- 30 epochs with AdamW (lr 3e-4, warmup 200 steps, grad clip 1.0)

### What the Notebook Result Shows

The tiny from-scratch model (1,041,555 parameters) trained on 2,400 FSDD spoken-digit clips achieves 98.3% exact-match accuracy (WER 0.017) on 600 held-out test samples. The pretrained whisper-tiny (39M parameters, zero-shot, no fine-tuning) transcribes 20 FSDD samples with 80% exact match. Under additive Gaussian noise from 40 dB to -5 dB SNR, both models degrade, with the tiny trained model maintaining higher accuracy at high noise levels (80% vs 67% at 40 dB, 13% vs 0% at -5 dB) — likely because FSDD's clean, single-speaker-digit audio is simpler to recognize under noise than Whisper's diverse web-scale training distribution. These results are not comparable to the paper's benchmarks (different dataset, model size, and evaluation protocol).

## Influence

Whisper was released as open-source models and inference code, quickly becoming a foundation for speech recognition applications and research. It was published at ICML 2023. It influenced subsequent work on multilingual ASR, speech translation, and robust speech processing. The open-source whisper package on GitHub has been widely adopted for transcription pipelines. About 5,900 citations ([OpenAlex](https://openalex.org/works/W4311000453), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/086_whisper-robust-speech-recognition/solution.ipynb) | From-scratch PyTorch tiny Whisper encoder-decoder ASR trained on FSDD, plus pretrained whisper-tiny probing and noise robustness comparison |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
