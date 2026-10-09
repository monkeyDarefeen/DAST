# DAST: Dual-Stream Voice Anonymization Attacker

This repository provides the implementation, training resources, pretrained checkpoints, and inference pipeline for **DAST**, a speaker verification attacker based on WavLM-Large features and an ECAPA-TDNN speaker embedding model (192-dimensional embeddings).

**This repository accompanies two related research papers:** the original DAST work and a follow-up study of acoustic- and content-oriented speaker verification attacks against multilingual voice anonymization. The second paper builds on the DAST attacker and introduces MultiVC-based adaptation for multilingual and cross-lingual attack experiments.

## Papers Associated with This Repository

### Paper 1 — DAST: Dual-Stream Voice Anonymization Attacker with Staged Training

**Ridwan Arefeen, Xiaoxiao Miao, Rong Tong, Aik Beng Ng, Simon See, and Timothy Liu (2026)**  
[Paper (arXiv:2603.12840)](https://arxiv.org/abs/2603.12840)

This is the **foundational DAST paper** describing the attacker and its staged training approach. The core model implementation, training pipeline, and original pretrained DAST checkpoints are associated with this work.

### Paper 2 — Exploiting Acoustic and Content-Oriented Speaker Verification Attacks Against Multilingual Voice Anonymization

**Ridwan Arefeen, Ze Li, Rong Tong, Ming Li, and Xiaoxiao Miao (2026)**  
[Paper (arXiv:2610.08107)](https://arxiv.org/abs/2610.08107)

This **follow-up paper** examines acoustic- and content-oriented speaker verification attacks against multilingual voice anonymization. In this repository, it is associated with the **MultiVC-adapted DAST acoustic attacker** and experiments involving multilingual and cross-lingual voice anonymization. MultiVC is a multilingual voice-converted corpus created using seven voice-conversion systems.

**How the papers relate:** Paper 1 provides the underlying DAST model and training framework; Paper 2 investigates an extended multilingual attack setting and uses a DAST checkpoint further adapted with MultiVC. They are complementary contributions, not two unrelated implementations.

For training DAST from scratch or fine-tuning a checkpoint, see the **[Training Guide](training/README.md)**.

## Checkpoints and Research Scope

| Resource | Description | Associated paper |
| --- | --- | --- |
| Original DAST Stage II checkpoints | Pretrained DAST checkpoints that can initialize further Stage III fine-tuning | Paper 1 — DAST |
| `pretrained_ckpt_multiVC/` | DAST / WavLM–ECAPA acoustic attacker initialized from pretrained DAST and further adapted using MultiVC | Paper 2 — Multilingual attacks (building on Paper 1) |
| Inference and training code | Shared DAST model implementation and workflows | Both papers |

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

### 3. Pretrained DAST Checkpoints for Fine-tuning

The original pretrained checkpoint resources include **two Stage II checkpoints**, either of which can be used to initialize **Stage III** training as described in the [Training Guide](training/README.md). These are associated with **Paper 1**.

### 4. MultiVC-adapted Checkpoint

The `pretrained_ckpt_multiVC/` directory contains the DAST acoustic attacker further adapted on **MultiVC**, as discussed in **Paper 2**. For inference, point `--model-dir` to a model directory containing the appropriate `embedding_model.ckpt` and `WavLM-Large.pt` files. The WavLM-Large checkpoint must be downloaded separately as described above.

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
├── pretrained_ckpt_multiVC/  # DAST checkpoint fine-tuned on the MultiVC dataset
```

---

## VoicePrivacy 2026 Challenge

This DAST model was used as the **Attacker Model** in the **[Voice Privacy Challenge 2026](https://www.voiceprivacychallenge.org/vp2026/#welcome2026)**, evaluating the effectiveness of participants' voice anonymization techniques against speaker recognition attacks.

**GitHub:** [Voice-Privacy-Challenge/Voice-Privacy-Challenge-2026](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2026)

---

## References

- **Paper 1 (original DAST):** [*DAST: A Dual-Stream Voice Anonymization Attacker with Staged Training*](https://arxiv.org/abs/2603.12840)
- **Paper 2 (multilingual voice anonymization attacks):** [*Exploiting Acoustic and Content-Oriented Speaker Verification Attacks Against Multilingual Voice Anonymization*](https://arxiv.org/abs/2610.08107)
- **WavLM-Large checkpoint:** [Microsoft UniLM / WavLM](https://github.com/microsoft/unilm/blob/master/wavlm/README.md)
- **Voice Privacy Challenge 2026:** <https://www.voiceprivacychallenge.org/vp2026/#welcome2026>

---

## Contact

For questions, email **rarefeen14@gmail.com** or reach out via [LinkedIn](https://www.linkedin.com/in/ridwan-arefeen-0950941a0/).

## Citation

This repository supports **both papers**. If you use DAST, the source code, any pretrained or MultiVC-adapted checkpoint, or the associated data/resources in your research, **please cite both papers** to acknowledge the original attacker and the follow-up multilingual attack study.

### Paper 1 — DAST

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

### Paper 2 — Multilingual Voice Anonymization Attacks

```bibtex
@misc{arefeen2026exploitingacousticcontentorientedspeaker,
      title={Exploiting Acoustic and Content-Oriented Speaker Verification Attacks Against Multilingual Voice Anonymization},
      author={Ridwan Arefeen and Ze Li and Rong Tong and Ming Li and Xiaoxiao Miao},
      year={2026},
      eprint={2610.08107},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2610.08107},
}
```

