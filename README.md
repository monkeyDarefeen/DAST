# DAST

Implementation of **DAST: A Dual-Stream Voice Anonymization Attacker with Staged Training**.

This repository provides the **inference pipeline** for computing speaker similarity scores between two audio files using a pre-trained DAST (Deep ASV) speaker embedding model — an ECAPA-TDNN classifier built on top of WavLM-Large self-supervised features (192-dim embeddings).

For training a DAST model from scratch (or fine-tuning a pre-trained checkpoint), see the **[Training Guide](training/README.md)** in the `training/` directory.

---

## Prerequisites

### 1. Install Dependencies

```bash
pip install numpy torch soundfile scipy speechbrain transformers joblib
```

### 2. Download the WavLM-Large Checkpoint

The DAST model requires the pre-trained **WavLM-Large** feature extractor. Download it from the official [Microsoft UniLM repository](https://github.com/microsoft/unilm/blob/master/wavlm/README.md) and place `WavLM-Large.pt` in your model directory alongside the DAST checkpoint (`embedding_model.ckpt`).

Your model directory should look like this:

```
your-model-dir/
├── embedding_model.ckpt    # DAST ECAPA-TDNN weights
└── WavLM-Large.pt          # WavLM-Large feature extractor (download from UniLM)
```

### 3. Pre-trained Checkpoints for Fine-tuning

The pre-trained checkpoint folder includes **Stage II** checkpoints, which can be used as a starting point for further fine-tuning in **Stage III** of the DAST training pipeline. Two Stage II checkpoints are available — choose either to continue training on your own data.

---

## Running Inference

### Basic Usage — Cosine Similarity

Compute the cosine similarity between two audio files:

```bash
python inference/dast_inference.py \
  --audio1 /path/to/enrollment.wav \
  --audio2 /path/to/probe.wav \
  --model-dir /path/to/your-model-dir
```

**Output (JSON to stdout):**

```json
{
  "audio1": "/path/to/enrollment.wav",
  "audio2": "/path/to/probe.wav",
  "score_type": "cosine_similarity",
  "score": 0.823456,
  "embedding_dim": 192,
  "elapsed_seconds": 3.42
}
```

### With QMF Calibration (Optional)

For calibrated probability output `P(same speaker)` in [0, 1], provide a trained QMF model and SNR metadata:

```bash
python inference/dast_inference.py \
  --audio1 /path/to/enrollment.wav \
  --audio2 /path/to/probe.wav \
  --model-dir /path/to/your-model-dir \
  --qmf-model /path/to/qmf_model.pkl \
  --snr1 15.3 \
  --snr2 12.7
```

### CPU Mode

By default the script uses CUDA. To run on CPU:

```bash
python inference/dast_inference.py \
  --audio1 /path/to/enrollment.wav \
  --audio2 /path/to/probe.wav \
  --model-dir /path/to/your-model-dir \
  --device cpu
```

### Save Output to File

```bash
python inference/dast_inference.py \
  --audio1 /path/to/enrollment.wav \
  --audio2 /path/to/probe.wav \
  --model-dir /path/to/your-model-dir \
  --output result.json
```

---

## Command-Line Arguments

| Argument      | Required | Description                                                                             |
| ------------- | -------- | --------------------------------------------------------------------------------------- |
| `--audio1`    | Yes      | Path to the first audio file (enrollment/reference)                                     |
| `--audio2`    | Yes      | Path to the second audio file (test/probe)                                              |
| `--model-dir` | Yes      | Path to DAST model directory (must contain `embedding_model.ckpt` and `WavLM-Large.pt`) |
| `--device`    | No       | Compute device: `cuda` (default) or `cpu`                                               |
| `--qmf-model` | No       | Path to trained QMF calibration model (`.pkl`). Output becomes a calibrated probability |
| `--snr1`      | No       | SNR in dB of audio1 (default: 50.0, used with `--qmf-model`)                            |
| `--snr2`      | No       | SNR in dB of audio2 (default: 50.0, used with `--qmf-model`)                            |
| `--dur1`      | No       | Duration of audio1 in seconds (auto-detected if omitted)                                |
| `--dur2`      | No       | Duration of audio2 in seconds (auto-detected if omitted)                                |
| `--output`    | No       | Write result to file instead of stdout (JSON format)                                    |

Supported audio formats: WAV, FLAC, OGG (anything `soundfile` can decode).

---

## Interpreting the Score

### Cosine Similarity (no QMF)

| Score Range | Interpretation               |
| ----------- | ---------------------------- |
| > 0.8       | Very likely the same speaker |
| 0.4 – 0.8   | Ambiguous / borderline       |
| < 0.4       | Likely different speakers    |

### QMF-Calibrated Probability

| Score Range | Interpretation                          |
| ----------- | --------------------------------------- |
| > 0.9       | High confidence: same speaker (genuine) |
| 0.5 – 0.9   | Moderate confidence                     |
| < 0.5       | More likely an impostor pair            |

---

## Folder Structure

```
DAST/
├── README.md
├── inference/
│   ├── dast_inference.py      # Main inference script
│   ├── dast_model.py          # DAST ECAPA-TDNN model definition (24-layer)
│   ├── modules.py             # Shared neural network modules
│   └── WavLM.py               # WavLM feature extractor wrapper
├── training/
│   ├── README.md              # Training guide with architecture, config, and troubleshooting
│   ├── dast_training.py       # Main training entry point
│   ├── dast_model.py          # Dual-stream ECAPA-TDNN model definition
│   ├── training_config.yaml   # Top-level training configuration (paths, hyperparameters)
│   ├── libri_prepare.py       # Data preparation: Kaldi I/O, train/dev split, CSV generation
│   ├── asv_dataset.py         # SpeechBrain dataset wrapper: audio loading, label encoding
│   ├── WavLM.py               # WavLM model architecture
│   ├── modules.py             # WavLM internals: attention, activations
│   ├── hparams/
│   │   └── train_ecapa_tdnn_small.yaml  # Inner SpeechBrain hyperparameters
│   ├── Muon/                  # Vendored Muon optimizer (optional)
│   └── utils/                 # Helpers: Kaldi I/O, path management, logging, result conversion
```

---

## VoicePrivacy 2026 Challenge

This DAST model was used as the **Attacker Model** in the **[Voice Privacy Challenge 2026](https://www.voiceprivacychallenge.org/vp2026/#welcome2026)**, evaluating the effectiveness of participants' voice anonymization techniques against speaker recognition attacks.

**GitHub:** [Voice-Privacy-Challenge/Voice-Privacy-Challenge-2026](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2026)

---

## References

- **WavLM-Large checkpoint:** [Microsoft UniLM / WavLM](https://github.com/microsoft/unilm/blob/master/wavlm/README.md)
- **Voice Privacy Challenge 2026:** <https://www.voiceprivacychallenge.org/vp2026/#welcome2026>

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{arefeen2026dastdualstreamvoiceanonymization,
      title={DAST: A Dual-Stream Voice Anonymization Attacker with Staged Training},
      author={Ridwan Arefeen and Xiaoxiao Miao and Rong Tong and Aik Beng Ng and Simon See and Timothy Liu},
      year={2026},
      eprint={2603.12840},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2603.12840},
}
```
