#!/usr/bin/env python3
"""
===============================================================================
DAST SIMILARITY SCORER — Standalone Two-Audio Comparison
===============================================================================

OVERVIEW
--------
This script computes a speaker similarity score between TWO audio files using
a pre-trained Deep ASV (DAST) speaker embedding model. It extracts fixed-
dimensional speaker embeddings from each audio file, then returns the cosine
similarity between them as a raw score in [-1, 1].

When a trained QMF (Quality-Metric Function) calibration model is also provided
along with audio duration/SNR metadata, the raw cosine score is calibrated into
a well-behaved probability P(same speaker) in [0, 1] via a logistic-regression
transform that accounts for audio quality (SNR), utterance length, and embedding
magnitude.

HOW IT WORKS (step-by-step)
----------------------------
  1. Load the DAST speaker embedding model from --model-dir.
     The model is an ECAPA-TDNN classifier on top of WavLM-Large SSL features
     (192-dim embeddings). Requires two checkpoint files inside --model-dir:
       * embedding_model.ckpt  (ECAPA-TDNN weights)
       * WavLM-Large.pt        (WavLM feature extractor)

  2. Read the two audio files (--audio1, --audio2) using soundfile.
     Supports WAV, FLAC, OGG — anything your backend can decode.

  3. Extract a speaker embedding vector from each file by forwarding the
     waveform through the loaded model (GPU or CPU).

  4. Compute cosine similarity between the two embedding vectors.
     This is the raw similarity score:
       - +1  = identical speaker direction (very likely same person)
       -  0  = orthogonal (unrelated)
       - -1  = opposite (definitely different)

  5. (Optional) If --qmf-model is given, calibrate the raw cosine score into
     a probability using a pre-trained logistic-regression QMF model. The
     calibration uses 8 features per utterance pair:
       [cosine_score, snr_enrol_norm, snr_test_norm,
        log_dur_enrol, log_dur_test, mag_enrol, mag_test, log_dur_total]
     Output is P(genuine) in [0, 1].

INPUTS
------
Required:
  --audio1 PATH        Path to the first audio file (enrollment/reference)
  --audio2 PATH        Path to the second audio file (test/probe)
  --model-dir PATH     Path to the DAST model directory (see HOW IT WORKS #1)

Optional:
  --device STR         Compute device: 'cuda' (default) or 'cpu'
  --qmf-model PATH     Path to a trained QMF calibration model (.pkl file).
                       If provided, the output is a calibrated probability
                       instead of raw cosine similarity.
  --snr1 FLOAT         SNR in dB of audio1 (used only with --qmf-model)
  --snr2 FLOAT         SNR in dB of audio2 (used only with --qmf-model)
  --dur1 FLOAT         Duration in seconds of audio1 (auto-detected if omitted)
  --dur2 FLOAT         Duration in seconds of audio2 (auto-detected if omitted)
  --output FILE        Write result to file instead of stdout (JSON format)

OUTPUT
------
Prints a JSON object to stdout (or writes to --output file):

  Without QMF:
  {
    "audio1": "/path/to/audio1.wav",
    "audio2": "/path/to/audio2.wav",
    "score_type": "cosine_similarity",
    "score": 0.823456,
    "embedding_dim": 192
  }

  With QMF:
  {
    "audio1": "/path/to/audio1.wav",
    "audio2": "/path/to/audio2.wav",
    "cosine_similarity": 0.781234,
    "score_type": "qmf_calibrated_probability",
    "score": 0.912345,
    "embedding_dim": 192
  }

INTERPRETING THE SCORE
-----------------------
  Cosine similarity (no QMF):
    > 0.8  -> Very likely the same speaker
    0.4-0.8 -> Ambiguous / borderline
    < 0.4  -> Likely different speakers

  QMF-calibrated probability:
    > 0.9  -> High confidence: same speaker (genuine)
    0.5-0.9 -> Moderate confidence
    < 0.5  -> More likely an impostor pair


USAGE EXAMPLES
--------------

  1) Basic cosine similarity:

     python3 dast_inference.py \
       --audio1 /data/enroll/speaker001_utt001.wav \
       --audio2 /data/test/probe_042.wav \
       --model-dir /app/datasets/ecapa_24L/.../asv_anon/CKPT+2

  2) With QMF calibration (returns probability):

     python3 dast_inference.py \
       --audio1 utterance_A.wav \
       --audio2 utterance_B.wav \
       --model-dir /app/datasets/ecapa_24L/.../CKPT+2 \
       --qmf-model ./qmf_lr_model.pkl \
       --snr1 15.3 --snr2 12.7

  3) From another script / programmatic use:

     import subprocess, json
     result = subprocess.run([
         'python3', 'dast_inference.py',
         '--audio1', 'a.wav', '--audio2', 'b.wav',
         '--model-dir', '/path/to/model'
     ], capture_output=True, text=True)
     data = json.loads(result.stdout)
     print(f"Similarity: {data['score']:.4f}")

  4) For GPT / AI agents — minimal call:

     python3 dast_inference.py \
       --audio1 <path_to_first_audio> \
       --audio2 <path_to_second_audio> \
       --model-dir <path_to_dast_model_dir>

     The stdout is valid JSON with a "score" field. Parse it directly.


DEPENDENCIES
------------
  numpy, torch, soundfile, scipy, speechbrain, transformers (for WavLM)
  For QMF: joblib (ships with scikit-learn)

Install: pip install numpy torch soundfile scipy speechbrain transformers joblib

AUTHOR / CONTEXT
----------------
Part of the Voice Privacy Challenge 2026 score normalization pipeline.
Based on the embedding extraction and scoring infrastructure in this project.
===============================================================================
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Compute speaker similarity between two audio files '
                    'using a DAST (Deep ASV) embedding model.'
    )
    # Required inputs
    p.add_argument('--audio1', required=True,
                   help='Path to the first audio file (enrollment/reference)')
    p.add_argument('--audio2', required=True,
                   help='Path to the second audio file (test/probe)')
    p.add_argument('--model-dir', required=True,
                   help='Path to the DAST model directory. '
                        'Must contain embedding_model.ckpt and WavLM-Large.pt.')

    # Model configuration
    p.add_argument('--device', default='cuda',
                   help='Compute device: cuda (default) or cpu')

    # QMF calibration (optional)
    p.add_argument('--qmf-model', default=None,
                   help='Path to trained QMF LR model (.pkl). '
                        'If given, output is a calibrated probability [0,1].')
    p.add_argument('--snr1', type=float, default=50.0,
                   help='SNR in dB of audio1 (default: 50.0, used only with --qmf-model)')
    p.add_argument('--snr2', type=float, default=50.0,
                   help='SNR in dB of audio2 (default: 50.0, used only with --qmf-model)')
    p.add_argument('--dur1', type=float, default=None,
                   help='Duration of audio1 in seconds (auto-detected if omitted)')
    p.add_argument('--dur2', type=float, default=None,
                   help='Duration of audio2 in seconds (auto-detected if omitted)')

    # Output
    p.add_argument('--output', default=None,
                   help='Output file path (JSON). Default: stdout.')

    return p.parse_args()


# =============================================================================
# Audio utilities
# =============================================================================

def get_audio_duration(wav_path):
    """Get audio duration in seconds without loading the full waveform."""
    import soundfile as sf
    info = sf.info(wav_path)
    return info.duration


def load_audio_tensor(wav_path, target_sr=None):
    """Load an audio file and return (torch.Tensor of shape (T,), sample_rate)."""
    import soundfile as sf
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # downmix to mono
    tensor = torch.from_numpy(audio).float()
    return tensor, sr


# =============================================================================
# Model loaders — thin wrappers around the existing extractors
# =============================================================================

def load_dast_extractor(model_dir, device):
    """
    Load ECAPA-TDNN + WavLM-Large DAST extractor (192-dim embeddings).

    model_dir must contain:
      - embedding_model.ckpt  (ECAPA-TDNN state dict)
      - WavLM-Large.pt        (WavLM config + weights)
    """
    # Ensure project-level imports are resolvable
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    sim_scorer_dir = str(Path(__file__).resolve().parent)  # sim_scorer/ — for local copies of WavLM, modules, ecapa_model
    if sim_scorer_dir not in sys.path:
        sys.path.insert(0, sim_scorer_dir)

    from speechbrain_vectors_ori import SpeechBrainVectors
    extractor = SpeechBrainVectors(
        vec_type='ecapa_ssl',
        device=device,
        model_path=Path(model_dir)
    )
    return extractor


# =============================================================================
# Embedding extraction
# =============================================================================

def extract_embedding(extractor, wav_path, device):
    """
    Extract a single speaker embedding from an audio file.

    Returns numpy array (embedding_dim,) — L2-normalized.
    """
    audio_tensor, sr = load_audio_tensor(wav_path)

    with torch.no_grad():
        vec = extractor.extract_vector(
            audio=audio_tensor,
            sr=sr,
            wav_path=wav_path
        )

    # Ensure 1-D numpy
    if isinstance(vec, torch.Tensor):
        vec = vec.cpu().numpy()
    if vec.ndim == 2:
        vec = vec[0, :]
    return vec


# =============================================================================
# Scoring
# =============================================================================

def cosine_similarity(a, b):
    """Compute cosine similarity between two 1-D numpy vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def build_qmf_feature(cosine_score, emb1, emb2, snr1, snr2, dur1, dur2,
                       snr_min, snr_max):
    """
    Build the 8-dimensional feature vector for QMF calibration.

    Features: [cosine, snr_enrol_norm, snr_test_norm,
               log_dur_enrol, log_dur_test,
               mag_enrol, mag_test, log_dur_total]
    """
    # Normalize SNR to [0, 1]
    denom = snr_max - snr_min if snr_max > snr_min else 1.0
    snr1_norm = (snr1 - snr_min) / denom
    snr2_norm = (snr2 - snr_min) / denom

    # Duration features
    log_dur1 = math.log(max(dur1, 0.01))
    log_dur2 = math.log(max(dur2, 0.01))
    log_dur_total = math.log(max(dur1 + dur2, 0.01))

    # Embedding L2 norms
    mag1 = float(np.linalg.norm(emb1))
    mag2 = float(np.linalg.norm(emb2))

    return np.array([
        cosine_score, snr1_norm, snr2_norm,
        log_dur1, log_dur2,
        mag1, mag2, log_dur_total
    ], dtype=np.float32).reshape(1, -1)


def qmf_calibrate(feature_matrix, clf):
    """Apply logistic regression QMF model. Returns P(genuine) in [0, 1]."""
    logit = feature_matrix @ clf.coef_.T + clf.intercept_
    prob = 1.0 / (1.0 + np.exp(-logit.ravel()))
    return float(prob[0])


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    t_start = time.time()

    # ── Validate inputs ──────────────────────────────────────────────
    if not os.path.isfile(args.audio1):
        print(f"Error: audio1 not found: {args.audio1}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.audio2):
        print(f"Error: audio2 not found: {args.audio2}", file=sys.stderr)
        sys.exit(1)
    if args.qmf_model and not os.path.isfile(args.qmf_model):
        print(f"Error: qmf-model not found: {args.qmf_model}", file=sys.stderr)
        sys.exit(1)

    # ── Load model ───────────────────────────────────────────────────
    print(f'Loading DAST model from {args.model_dir} ...', flush=True)
    t0 = time.time()

    extractor = load_dast_extractor(args.model_dir, args.device)

    print(f'  Model loaded in {time.time() - t0:.1f}s', flush=True)

    # ── Extract embeddings ───────────────────────────────────────────
    print(f'Extracting embedding for audio1: {args.audio1}', flush=True)
    t0 = time.time()
    emb1 = extract_embedding(extractor, args.audio1, args.device)
    print(f'  Done in {time.time() - t0:.2f}s (dim={emb1.shape})', flush=True)

    print(f'Extracting embedding for audio2: {args.audio2}', flush=True)
    t0 = time.time()
    emb2 = extract_embedding(extractor, args.audio2, args.device)
    print(f'  Done in {time.time() - t0:.2f}s (dim={emb2.shape})', flush=True)

    # ── Cosine similarity ────────────────────────────────────────────
    cos_score = cosine_similarity(emb1, emb2)
    print(f'\nCosine similarity: {cos_score:.6f}', flush=True)

    # ── QMF calibration (optional) ───────────────────────────────────
    final_score = cos_score
    score_type = 'cosine_similarity'

    if args.qmf_model:
        print(f'\nApplying QMF calibration from {args.qmf_model} ...', flush=True)
        import joblib
        qmf = joblib.load(args.qmf_model)
        clf = qmf['model']
        snr_min = qmf['snr_min']
        snr_max = qmf['snr_max']

        # Get durations (auto-detect if not provided)
        dur1 = args.dur1 if args.dur1 is not None else get_audio_duration(args.audio1)
        dur2 = args.dur2 if args.dur2 is not None else get_audio_duration(args.audio2)

        X = build_qmf_feature(
            cos_score, emb1, emb2,
            args.snr1, args.snr2, dur1, dur2,
            snr_min, snr_max
        )
        final_score = qmf_calibrate(X, clf)
        score_type = 'qmf_calibrated_probability'
        print(f'  Calibrated probability: {final_score:.6f}', flush=True)

    # ── Build result ─────────────────────────────────────────────────
    elapsed = time.time() - t_start
    result = {
        'audio1': str(args.audio1),
        'audio2': str(args.audio2),
        'score_type': score_type,
        'score': round(final_score, 6),
        'embedding_dim': int(emb1.shape[0]),
        'elapsed_seconds': round(elapsed, 2)
    }

    if args.qmf_model:
        result['cosine_similarity'] = round(cos_score, 6)

    # ── Output ───────────────────────────────────────────────────────
    output_json = json.dumps(result, indent=2)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json + '\n')
        print(f'\nResult written to {args.output}', flush=True)
    else:
        print('\n' + output_json, flush=True)


if __name__ == '__main__':
    main()
