# DAST Training: Speaker Embedding with ECAPA-TDNN + WavLM Fusion

Train an **ECAPA-TDNN** speaker embedding model augmented with **WavLM** multi-layer features, using Kaldi-style LibriSpeech data and SpeechBrain.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Data Format & Folder Structure](#data-format--folder-structure)
- [Configuration](#configuration)
- [Running the Training](#running-the-training)
- [Output Artifacts](#output-artifacts)
- [Code Walkthrough](#code-walkthrough)
  - [File Index](#file-index)
  - [Execution Flow](#execution-flow)
  - [Data Preparation Pipeline](#data-preparation-pipeline)
  - [Model Architecture](#model-architecture-details)
  - [Training Loop](#training-loop)
- [Multi-GPU Training](#multi-gpu-training)
- [Troubleshooting](#troubleshooting)

---

## Overview

This codebase trains a speaker identification model that fuses two complementary feature streams:

1. **Fbank features** (80-dimensional mel-filterbank) processed through an ECAPA-TDNN backbone
2. **WavLM multi-layer representations** (24 transformer layers from WavLM-Large) fused via a learnable weighted sum

The model uses **Additive Angular Margin (AAM)** loss with cosine similarity scoring, trained on LibriSpeech train-clean-360 (921 speakers, ~98K utterances).

### Key papers

- [ECAPA-TDNN](https://arxiv.org/abs/2005.07143) -- Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification
- [WavLM](https://arxiv.org/abs/2110.13900) -- Large-Scale Self-Supervised Pre-training for Full Stack Speech Processing

---

## Architecture

```
Input Audio (16kHz wav)
    |
    +-----> Fbank Features (80-dim) ----+
    |                                    v
    |                              ECAPA-TDNN Blocks
    |                              (TDNN + SE-Res2Net)
    |                                    v
    |                              Multi-Layer Feature Aggregation (MFA)
    |                                    |
    +-----> WavLM-Large                  |
              |                          |
              v                          |
       24 Transformer Layers             |
              |                          |
              v                          |
       WeightedLayerSum (learnable)      |
              |                          |
              v                          |
       SERes2Net Processing Blocks       |
              |                          |
              +----------+---------------+
                         |
                         v
                 Element-wise Multiply
                         |
                         v
            Attentive Statistical Pooling (ASP)
                         |
                         v
                  192-dim Embedding
                         |
                         v
                  Speaker Classifier
                (Linear -> 921 speakers)
```

---

## Prerequisites

### Python Environment

```bash
# Core dependencies
pip install torch torchaudio speechbrain pandas soundfile numpy hyperpyyaml tqdm

# Muon optimizer (optional -- set optimizer_type: muon in config)
# The Muon package is included in this repo under ./Muon/
```


### Required External Files

| File | Description | Default Path in Config |
|------|-------------|----------------------|
| `WavLM-Large.pt` | Pretrained WavLM checkpoint from Microsoft | Set via `wavlm_model_path` in `training_config.yaml` |
| LibriSpeech data | train-clean-360 subset in Kaldi format | Set via `data_dir` in `training_config.yaml` |

---

## Data Format & Folder Structure

The code expects **two** data sources:

### 1. Training Data (Kaldi Format)

This is the primary training corpus used to build CSV annotations and train the ECAPA-TDNN model. The folder must contain these Kaldi-style files:

```
data_dir/                          # e.g. /app/datasets/vpc/T12-5/data/train-clean-360
├── wav/                           # Directory containing .wav audio files
│   ├── 100-121669-0000.wav
│   ├── 100-121669-0001.wav
│   ├── 100-121669-0002.wav
│   └── ...                        # ~98K files for LibriSpeech train-clean-360
├── wav.scp                        # Kaldi wav.scp: one line per utterance
├── utt2spk                        # Kaldi utt2spk: utterance -> speaker mapping
├── spk2utt                        # Kaldi spk2utt: speaker -> utterances mapping
└── (optional) text, utt2dur, spk2gender
```

#### Required Kaldi Files (example)

**`wav.scp`** -- Maps each utterance ID to its absolute WAV file path:
```
100-121669-0000 /app/datasets/vpc/T12-5/data/train-clean-360//wav/100-121669-0000.wav
100-121669-0001 /app/datasets/vpc/T12-5/data/train-clean-360//wav/100-121669-0001.wav
```
Format: `<utt_id> <absolute_path_to_wav>` (tab or space separated)

**`utt2spk`** -- Maps each utterance to its speaker ID:
```
100-121669-0000 100
100-121669-0001 100
100-121669-0002 100
```
Format: `<utt_id> <speaker_id>`

**`spk2utt`** -- Maps each speaker to all their utterance IDs (space-separated on one line):
```
100 100-121669-0000 100-121669-0001 100-121669-0002 ... 100-121674-0030
1001 1001-134707-0000 1001-134707-0001 ...
```
Format: `<speaker_id> <utt_id_1> <utt_id_2> ...`

#### Filename Convention

WAV filenames follow the LibriSpeech convention: `<speaker_id>-<session_id>-<utterance_index>.wav`

The code extracts speaker/session/utterance IDs by splitting on `-`:
```python
# From "100-121669-0000.wav":
spk_id = "100"      # last-3 segment
sess_id = "121669"  # last-2 segment
utt_id = "0000"     # last-1 segment (before .wav)
```

### 2. Input Audio Folders (WavLM Feature Lookup)

This is a separate directory containing `.wav` files that the **WavLM feature extractor** looks up by utterance ID during training:

```
input_audio_folders[0]/            # e.g. /app/...../wav
├── E100-2315-000181.wav
├── E100-2315-000258.wav
├── E10001-8844-000002.wav
└── ...                           
```

During training, when the model processes a batch with utterance IDs like `["100--121669--0000_0_3"]`, the WavLM extractor:
1. Strips the chunk timestamp suffix (`_0_3`)
2. Collapses double-dashes: `100--121669--0000` -> `100-121669-0000`
3. Appends `.wav` and searches each folder in `input_audio_folders` for the file

**Important**: The utterance IDs in this folder must match those derived from the training data's CSV annotations after normalization (see `normalize_uttid()` in `dast_training.py`).

### Complete Directory Tree Example

```
/app/datasets/vpc/
├── T12-5/data/train-clean-360/       # <-- data_dir (Kaldi format)
│   ├── wav/
│   │   ├── 100-121669-0000.wav
│   │   └── ...
│   ├── wav.scp
│   ├── utt2spk
│   └── spk2utt
│
└── vpc26test/multilantest/Voice-Privacy-Challenge-2026/data/merged_trainin_BM1/wav/
    ├── E100-2315-000181.wav          # <-- input_audio_folders[0] (WavLM lookup)
    └── ...
```

---

## Configuration

All settings are in `training_config.yaml`. The file uses **HyperPyYAML** with `!ref` cross-references for derived paths.

### Required Settings

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `data_dir` | `str` | Path to Kaldi-format training data directory | `/app/datasets/vpc/T12-5/data/train-clean-360` |
| `exp_dir` | `str` | Experiment output directory (relative or absolute) | `dast_test` |
| `wavlm_model_path` | `str` | Path to pretrained WavLM-Large checkpoint | `/app/vpc/wavlm/WavLM-Large.pt` |
| `input_audio_folders` | `list[str]` | List of directories for WavLM audio file lookup | `[/app/datasets/.../wav]` |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | `7` | Number of training epochs |
| `batch_size` | `64` | Batch size (effective per-GPU in multi-GPU mode) |
| `lr` | `0.01` | Learning rate (max for CyclicLR scheduler) |
| `optimizer_type` | `adam` | Optimizer: `"adam"` or `"muon"` |
| `muon_lr` | `0.02` | Learning rate for Muon optimizer hidden layers (if using Muon) |
| `num_workers` | `10` | DataLoader worker processes |

### Data Selection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_spk` | `ALL` | Number of speakers: `"ALL"` or an integer |
| `num_utt` | `ALL` | Utterances per speaker: `"ALL"` or an integer |
| `utt_selection` | `spk-diverse-sess` | Selection strategy (see below) |

**Utterance selection strategies:**

- `spk-sess`: Pick N utterances per speaker-session pair
- `spk-random`: Pick N utterances randomly across all sessions for each speaker
- `spk-diverse-sess`: Distribute N utterances evenly across sessions for each speaker (default)

### Other Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | `1986` | Random seed |
| `pretrained_model` | `null` | Path to pretrained ECAPA model checkpoint (optional) |
| `train_config` | `./hparams/train_ecapa_tdnn_small.yaml` | Inner SpeechBrain hyperparameter file |
| `finetuning` | `false` | Whether fine-tuning mode is enabled |
| `retrain` | `true` | Force retraining even if model exists |

### How `!ref` Works in Config

HyperPyYAML `!ref` creates cross-references between parameters. For example:

```yaml
data_dir: /app/datasets/vpc/T12-5/data/train-clean-360
exp_dir: dast_test
train_data_name: ""

model_dir: !ref <exp_dir>/asv_anon<anon_data_suffix>
train_data_dir: !ref <data_dir>/<train_data_name>
```

Resolves to:
- `model_dir` -> `dast_test/asv_anon`
- `train_data_dir` -> `/app/datasets/vpc/T12-5/data/train-clean-360/`

---

## Running the Training

### Single GPU

```bash
cd /app/training
python dast_training.py training_config.yaml
```

The default config is used if no argument is provided:
```bash
python dast_training.py
```

### Multi-GPU (DDP)

Use SpeechBrain's built-in DDP launcher via the `--n_gpus` flag:

```bash
# 2 GPUs
python dast_training.py training_config.yaml --n_gpus 2

# 4 GPUs
python dast_training.py training_config.yaml --n_gpus 4
```

SpeechBrain handles process group initialization and device assignment automatically. Each GPU runs a full copy of the pipeline with its own dataloader workers.

### With Pretrained Model

Set `pretrained_model` in `training_config.yaml`:
```yaml
pretrained_model: /path/to/checkpoint.pt
```

The checkpoint should contain either a top-level `state_dict` key or be a raw state dict file. Keys prefixed with `module.` are automatically stripped for compatibility.

---

## Output Artifacts

All outputs go to `<exp_dir>/asv_anon/` (e.g., `dast_test/asv_anon/`):

```
dast_test/asv_anon/
├── hyperparams.yaml          # Saved copy of training config
├── env.log                   # Environment diagnostics
├── log.txt                   # Training log with epoch stats
├── train_log.txt             # Per-epoch loss and error rate
├── train.csv                 # Generated training CSV annotation
├── dev.csv                   # Generated validation CSV annotation
├── train_client.csv          # Shuffled copy of train.csv for client use
├── opt_libri_prepare.pkl     # Data prep options (used to skip re-preparation)
├── label_encoder.txt         # Speaker ID -> integer encoding
├── epoch_1_latest.pt         # Latest checkpoint per epoch
├── epoch_2_latest.pt
├── ...
└── epoch_<N>_latest.pt       # Best checkpoints (top 7 by error rate)
```

### CSV Annotation Format

The generated `train.csv` and `dev.csv` have columns:

| Column | Description | Example |
|--------|-------------|---------|
| `ID` | Chunk ID: `<spk>--<sess>--<utt>_<start>_<end>` | `100--121669--0000_0_3` |
| `duration` | Full audio duration in seconds | `4.52` |
| `wav` | Absolute path to WAV file | `/app/.../wav/100-121669-0000.wav` |
| `start` | Start sample index (at 16kHz) | `0` |
| `stop` | End sample index (at 16kHz) | `72320` |
| `spk_id` | Speaker ID | `100` |

Each audio file is split into 3-second chunks (configurable via `sentence_len` in `train_ecapa_tdnn_small.yaml`). For a 4.52-second file, this produces 1 chunk (`int(4.52 / 3.0) = 1`).

---

## Code Walkthrough

### File Index

| File | Purpose |
|------|---------|
| `README.md` | This training guide |
| `dast_training.py` | **Main entry point** -- config loading, model init, training loop, WavLM extractor |
| `training_config.yaml` | Top-level configuration (paths, hyperparameters) |
| `libri_prepare.py` | Data preparation: reads Kaldi files, splits train/dev, generates CSV annotations |
| `asv_dataset.py` | SpeechBrain dataset wrapper: audio loading pipeline, label encoding |
| `dast_model.py` | Model definitions: `ECAPA_TDNN_test`, `WeightedLayerSum`, `resample_tensor` |
| `modules.py` | WavLM internals: `MultiheadAttention`, activation functions, quantization noise |
| `WavLM.py` | WavLM model architecture (based on Microsoft's implementation) |
| `hparams/train_ecapa_tdnn_small.yaml` | Inner SpeechBrain config: feature extraction, optimizer, loss, augmentations |
| `Muon/` | Vendored Muon optimizer package (optional — not installed via pip; referenced directly from `./Muon/`) |
| `utils/` | Helpers: Kaldi format I/O (`data_io.py`), path management (`path_management.py`), logging (`logger.py`), SCP path conversion (`relative_scp_to_abs.py`), result formatting (`convert_results.py`, `prepare_results_in_kaldi_format.py`) |

### Execution Flow

```
dast_training.py (__main__)
    |
    +-- 1. Load training_config.yaml -> train_params
    |
    +-- 2. Build overrides_dict from train_params
    |       (maps top-level config to inner SpeechBrain hparams)
    |
    +-- 3. Load hparams/train_ecapa_tdnn_small.yaml with overrides -> hparams
    |
    +-- 4. Create experiment directory (exp_dir/asv_anon/)
    |
    +-- 5. Initialize DDP (SpeechBrain ddp_init_group)
    |
    +-- 6. Instantiate ECAPA_TDNN_test model on GPU
    |
    +-- 7. Load pretrained weights (if pretrained_model is set)
    |
    +-- 8. Replace hparams embedding_model with twin, configure checkpointer
    |
    +-- 9. Create SpeakerBrain training class
    |
    +-- 10. Load WavLM-Large onto GPU -> WavLMFeatureExtractor
    |        (audio lookup from input_audio_folders)
    |
    +-- 11. dataio_prep(hparams):
    |         |-- prepare_libri() [run_on_main only]
    |         |     |-- Read Kaldi wav.scp, utt2spk, spk2utt
    |         |     |-- Split train/dev (95/5)
    |         |     |-- Generate train.csv, dev.csv
    |         |
    |         |-- Shuffle train.csv -> train_client.csv
    |         |-- ASVDatasetGenerator.dataio_prep()
    |               |-- DynamicItemDataset from CSV
    |               |-- Audio pipeline (torchaudio.load)
    |               |-- Label encoder (speaker ID -> int)
    |
    +-- 12. ECAPATDNNClient.fit() -> model.fit(epochs)
```

### Data Preparation Pipeline

#### Step 1: Read Kaldi Files (`_get_utt_split_lists`)

```python
# From wav.scp: utterance_id -> [wav_path_list]
spk_files = read_kaldi_format(data_folder / 'wav.scp')

# From utt2spk: utterance_id -> speaker_id
utt2spk = read_kaldi_format(data_folder / 'utt2spk')

# From spk2utt: speaker_id -> [utterance_id_list]
spk2utt = read_kaldi_format(data_folder / 'spk2utt')
```

#### Step 2: Select Speakers and Utterances

Based on `num_spk`, `num_utt`, and `utt_selection`:

- **`num_spk == "ALL"`**: Use all speakers (921 for LibriSpeech train-clean-360)
- **`num_spk == N`**: Randomly sample N speakers
- **`num_utt == "ALL"`**: Use all utterances for selected speakers, deduplicate paths
- **`num_utt == N`**: Select N utterances per speaker using the `utt_selection` strategy

#### Step 3: Train/Dev Split

The selected utterance list is shuffled and split at the ratio defined in config (95% train / 5% dev).

#### Step 4: CSV Generation (`prepare_csv`)

For each WAV file in the split:
1. Parse speaker/session/utterance IDs from filename
2. Read audio duration via `sf.info()`
3. If `random_segment=False` (default): split into fixed-length chunks of `sentence_len` seconds
4. For each chunk, compute start/stop sample indices and optionally filter by amplitude threshold
5. Write row to CSV

The process uses a **single-threaded** loop for reliability. A commented-out parallel version exists in the file but is disabled due to CUDA fork-safety issues.

### Model Architecture Details

#### ECAPA-TDNN_test (`dast_model.py`)

A dual-stream ECAPA-TDNN variant:

**Stream 1 (Fbank):**
- TDNN block (input_size=80 -> 1024 channels)
- 3 SE-Res2Net blocks (1024 -> 1024, with scale=8)
- Outputs: list of intermediate representations `xl[1:]`

**Stream 2 (WavLM):**
- `WeightedLayerSum`: Learnable softmax-weighted combination of 24 WavLM transformer layers
  - Input shape: `[24, T, B, 1024]` -> Output: `[T, B, 1024]`
- 4 SE-Res2Net blocks processing the fused WavLM features
- Outputs: list of intermediate representations `xl_features[1:]`

**Fusion:**
- Concatenate intermediates from each stream separately
- Resample WavLM stream temporally to match Fbank stream length
- Element-wise multiply the two streams
- Pass through MFA (Multi-Layer Feature Aggregation) TDNN block

**Pooling & Classification:**
- Attentive Statistical Pooling (ASP) -> mean + std concatenation
- BatchNorm1d -> Linear projection to 192-dim embedding
- Classifier head: Linear(192 -> num_speakers)

#### WeightedLayerSum

```python
# 24 learnable weights, normalized via softmax
self.layer_weights = nn.Parameter(torch.ones(24))

def forward(self, layer_results):
    # layer_results shape: [24, T, B, 1024]
    weights = softmax(self.layer_weights)          # [24]
    weights = weights.view(24, 1, 1, 1)           # broadcastable
    return (layer_results * weights).sum(dim=0)   # [T, B, 1024]
```

Initially all weights are equal (uniform attention). During training, the model learns to emphasize layers that are most informative for speaker discrimination.

#### resample_tensor

Aligns the temporal dimension of WavLM features to match Fbank features:

- **Downsampling** (`T > target`): Average pooling with proportional bin sizes
- **Upsampling** (`T < target`): Cyclical repetition (tiling) + remainder fill

### Training Loop

The `SpeakerBrain` class (extends `speechbrain.core.Brain`) defines the training loop:

#### Forward Pass (`compute_forward`)

```python
def compute_forward(self, batch, stage):
    wavs, lens = batch.sig

    # Audio augmentation (train only)
    if stage == TRAIN and has_augment:
        wavs, lens = wav_augment(wavs, lens)

    # Fbank features + normalization
    feats = compute_features(wavs)
    feats = mean_var_norm(feats, lens)

    # WavLM features (by utterance ID lookup)
    uttid = normalize_uttid(batch.id)
    wavlm_feats = self.wavlm_extractor.extract_features(uttid)

    # Dual-stream model
    embeddings = embedding_model(feats, wavlm_feats)
    outputs = classifier(embeddings)

    return outputs, lens
```

#### Loss (`compute_objectives`)

**Additive Angular Margin (AAM)** loss:
- Margin: 0.2
- Scale: 30
- Combined with LogSoftmax

#### Optimizer

- **Adam** (default): `lr=0.01`, `weight_decay=2e-6`
- **Muon**: Two-param-group optimizer -- Muon for hidden >=2D layers (`lr=0.02`), AdamW for others (`lr=0.01`)

#### Learning Rate Schedule

**CyclicLR**: Oscillates between `base_lr=1e-8` and `max_lr=0.01` every 65,000 steps.

#### Augmentations (Train Only)

Applied via SpeechBrain's `Augmenter` with 4 augmentations per sample:
- **DropChunk**: Randomly drop 1-5 temporal chunks (1000-2000 samples each)
- **DropFreq**: Randomly zero out 1-3 frequency bands (5% width each)

#### Checkpointing

- Saves top 7 checkpoints by validation error rate
- Saves `epoch_<N>_latest.pt` at end of each epoch
- Grad accumulation factor: 8 (effective batch size = `batch_size * 8`)

---

## Multi-GPU Training

### How It Works

SpeechBrain's `ddp_init_group()` initializes PyTorch DDP with NCCL backend. Each GPU gets its own:
- Model copy (replicated)
- WavLM extractor (loaded independently per GPU)
- DataLoader workers (`num_workers` per GPU)

The `run_on_main()` wrapper ensures data preparation (`prepare_libri`) runs only on rank 0, then a DDP barrier synchronizes all ranks before proceeding.

### Known Issues

**Memory**: Each GPU loads its own copy of WavLM-Large (~1.5GB). For 8 GPUs, ensure sufficient VRAM per-GPU.

---

## Troubleshooting

### Training Gets Stuck at "Processing N files with M workers"

**Symptom**: Progress bar freezes during `prepare_csv`, many processes appear in `ps aux`.

**Cause**: CUDA fork-safety issue. When `ProcessPoolExecutor` forks children from a parent with active CUDA context, all children contend on CUDA runtime locks (futex deadlock).

**Fixes applied**:
1. Worker count capped at 32 (`min(32, os.cpu_count())`)
2. Single-threaded CSV generation is the default active code path

**If still stuck**: Ensure the parallel CSV code block remains commented out, or use `spawn` start method if you enable it.

### "Audio file not found in any predefined folder"

The WavLM extractor searches `input_audio_folders` for `<normalized_utt_id>.wav`. Verify:
1. The folders listed in `input_audio_folders` exist and contain `.wav` files
2. Utterance IDs match after normalization (strip timestamps, collapse `--` to `-`)
3. Files have the `.wav` extension (not `.flac` or other)

### Out of Memory

- Reduce `batch_size` in `training_config.yaml`
- Reduce `num_workers` if DataLoader workers consume too much CPU memory
- Use gradient accumulation (`grad_accumulation_factor: 8`) to maintain effective batch size with smaller per-GPU batches

### "Malformed path" warnings during data prep

WAV filenames must have at least 3 hyphen-separated segments (e.g., `100-121669-0000.wav`). Files that don't match this pattern are skipped. Check your `wav.scp` entries for consistency.

### Skipping Data Preparation

If `train.csv`, `dev.csv`, and `opt_libri_prepare.pkl` already exist in the output folder with matching options, preparation is automatically skipped. To force re-preparation:
- Delete the CSV files and `.pkl` file from the output folder, or
- Change a parameter that affects data selection (`num_spk`, `num_utt`, etc.)
