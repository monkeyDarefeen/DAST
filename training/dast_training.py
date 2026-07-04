import os
import csv
import time
import torch
import shutil
import copy
from typing import List, Tuple, Dict, Union,Optional
import multiprocessing
from speechbrain.lobes.models.ECAPA_TDNN import *
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main
import pandas as pd
from libri_prepare import prepare_libri  # noqa
from asv_dataset import ASVDatasetGenerator
import logging
from pathlib import Path
from datetime import datetime
from dast_model import ECAPA_TDNN_test
from typing import Union, List
import sys
import numpy as np

try:
    from Muon.muon import MuonWithAuxAdam
except ImportError:
    MuonWithAuxAdam = None

config_path = "training_config.yaml"

import os
import torch
import soundfile as sf
from WavLM import WavLM, WavLMConfig
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List, Union, Tuple, Dict
import gc
from collections import defaultdict
import os
import shutil

def limit_utts_per_speaker(folder, max_utts_per_spk):
    wav_scp = os.path.join(folder, "wav.scp")
    wav_backup_scp = os.path.join(folder, "wav_backup.scp")

    if not os.path.isfile(wav_scp):
        raise FileNotFoundError(f"wav.scp not found in {folder}")

    if not os.path.isfile(wav_backup_scp):
        shutil.copy(wav_scp, wav_backup_scp)
        print(f"Created backup: {wav_backup_scp}")
    else:
        print(f"Using existing backup: {wav_backup_scp}")

    spk2utts = defaultdict(list)

    with open(wav_backup_scp, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            utt_id, wav_path = line.split(maxsplit=1)
            spk_id = utt_id.split("-")[0]
            spk2utts[spk_id].append((utt_id, wav_path))

    with open(wav_scp, "w") as f:
        for spk_id, utts in spk2utts.items():
            total_utts = len(utts)

            if total_utts < max_utts_per_spk:
                print(f"Speaker {spk_id}: only {total_utts} utterances, using all")

            for utt_id, wav_path in utts[:max_utts_per_spk]:
                f.write(f"{utt_id} {wav_path}\n")

    print(
        f"Saved filtered wav.scp in {folder} "
        f"(max {max_utts_per_spk} utterances per speaker)"
    )


class WavLMFeatureExtractor:
    """
    WavLM feature extractor that searches for audio files across
    predefined folders and performs true batched inference.
    """

    def __init__(self, model_path: str, device: str = None, INPUT_FOLDERS: list =None, max_length: float = 10.0):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.INPUT_FOLDERS = INPUT_FOLDERS
        print(f"Loading WavLM model to {self.device}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.cfg = WavLMConfig(checkpoint['cfg'])
        self.max_length = max_length 
        self.model = WavLM(self.cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()
        print("Extractor is ready.")
    def _crop_or_pad(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Crop to max_length if longer (random segment), pad if shorter."""
        max_samples = int(self.max_length * sr)
        n = audio.shape[0]
        if n > max_samples:
            start = torch.randint(0, n - max_samples + 1, (1,)).item()
            return audio[start : start + max_samples]
        elif n < max_samples:
            return F.pad(audio, (0, max_samples - n), mode="constant", value=0)
        return audio
    def _find_audio_path(self, audio_name: str) -> str:
        """Searches predefined folders for the audio file."""
        file_name = f"{audio_name}.wav"
        for folder in self.INPUT_FOLDERS:
            full_path = os.path.join(folder, file_name)
            if os.path.isfile(full_path):
                
                return full_path
        raise FileNotFoundError(f"Audio file '{file_name}' not found in any predefined folder.")

    def _load_audio(self, file_path: str):
        """Loads and pre-processes a single audio file."""
        audio_input, sr = sf.read(file_path)
        audio_input = torch.from_numpy(audio_input).float()

        if audio_input.dim() > 1:
            audio_input = audio_input[0, :]

        if self.cfg.normalize:
            audio_input = F.layer_norm(audio_input, audio_input.shape)
        return audio_input, sr

    def extract_features(
        self, 
        audio_names: Union[str, List[str], Tuple[str, ...]]
    ) -> Dict[str, Union[List[torch.Tensor], List[str]]]:
        if isinstance(audio_names, str):
            audio_names = [audio_names]

        audio_inputs = []
        for name in audio_names:
            path = self._find_audio_path(name)
            audio, sr = self._load_audio(path)
            audio = self._crop_or_pad(audio, sr)
            audio_inputs.append(audio)
        padded_batch = pad_sequence(audio_inputs, batch_first=True)
        padded_batch = padded_batch.to(self.device)

        with torch.no_grad():
            final_features, layer_results_list = self.model.extract_features(
                padded_batch,
                output_layer=self.model.cfg.encoder_layers,
                ret_layer_results=True
            )
        layer_reps = [x for x, _ in final_features[1][1:]] 
        layer_reps = torch.stack(layer_reps, dim=0)
        
        del final_features, layer_results_list, padded_batch,audio_inputs
        return layer_reps
    
    
class WavLMFeatureExtractor_updated:
    INPUT_FOLDERS = [
        "/app/datasets/vpc/sstc/combined/data/train-clean-360/wav"
    ]
    def __init__(
        self,
        old_model_path: str,
        new_model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\n--- Processing OLD Model: {os.path.basename(old_model_path)} ---")
        old_ckpt = torch.load(old_model_path, map_location="cpu")
        self.cfg = WavLMConfig(old_ckpt.get("cfg", {}))
        self.model_old = WavLM(self.cfg).to(self.device)

        old_sd = old_ckpt.get("model", old_ckpt)
        self._load_and_check_init(self.model_old, old_sd, "OLD")

        self.model_new = None
        if new_model_path:
            print(f"\n--- Processing NEW Model: {os.path.basename(new_model_path)} ---")
            new_ckpt = torch.load(new_model_path, map_location="cpu")

            raw_new_sd = new_ckpt.get("model", new_ckpt)
            prefix = "feature_extract.model."

            new_sd = {
                k[len(prefix):]: v for k, v in raw_new_sd.items() 
                if k.startswith(prefix)
            }

            if not new_sd:
                print(f"CRITICAL: No keys starting with '{prefix}' found in NEW model!")

            self.model_new = WavLM(self.cfg).to(self.device)
            self._load_and_check_init(self.model_new, new_sd, "NEW")

            self._compare_keys(old_sd, new_sd)
            if self.model_new is not None:
                self.unload_model("old")
            else:
                print("No new model provided; keeping old model.")
    
    def unload_model(self, model_to_remove: str = "old"):
        """Removes a model and its weight dictionaries from memory."""
        if model_to_remove == "old":
            print("Unloading OLD model to free memory...")
            if hasattr(self, 'model_old'):
                del self.model_old
                self.model_old = None

            if hasattr(self, 'old_model_dict'):
                del self.old_model_dict
                self.old_model_dict = None

        elif model_to_remove == "new":
            print("Unloading NEW model to free memory...")
            if hasattr(self, 'model_new'):
                del self.model_new
                self.model_new = None

            if hasattr(self, 'new_model_dict'):
                del self.new_model_dict
                self.new_model_dict = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"Successfully unloaded {model_to_remove} model.")
    def _load_and_check_init(self, model: torch.nn.Module, state_dict: dict, label: str):
        """Loads state_dict and reports exactly which parameters are random."""
        load_result = model.load_state_dict(state_dict, strict=False)
        
        missing = load_result.missing_keys
        unexpected = load_result.unexpected_keys

        if not missing:
            print(f"✅ [{label}] All parameters successfully loaded from checkpoint. None are random.")
        else:
            print(f"⚠️ [{label}] WARNING: {len(missing)} parameters were NOT found and are RANDOMLY INITIALIZED:")
            for k in missing[:15]:
                print(f"   - {k}")
            if len(missing) > 15:
                print(f"   ... and {len(missing)-15} more.")

        if unexpected:
            print(f"ℹ️ [{label}] Skipped {len(unexpected)} extra keys in checkpoint (not needed for WavLM).")

    def _compare_keys(self, old_sd: dict, new_sd: dict):
        old_keys = set(old_sd.keys())
        new_keys = set(new_sd.keys())
        missing_in_new = sorted(old_keys - new_keys)
        
        print(f"\n--- Checkpoint Sync Check ---")
        if missing_in_new:
            print(f"❌ NEW model is missing {len(missing_in_new)} keys that existed in OLD model.")
        else:
            print("✅ NEW model contains all WavLM keys found in the OLD model.")

    def _unwrap_model_dict(self, checkpoint: dict) -> Dict[str, torch.Tensor]:
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}

    def _report_differences(self, old_dict: dict, new_dict: dict):
        old_keys = set(old_dict.keys())
        new_keys = set(new_dict.keys())
        
        missing = sorted(old_keys - new_keys)
        extra = sorted(new_keys - old_keys)

        print("\n--- Key Comparison (Relative to WavLM structure) ---")
        if missing:
            print(f"MISSING in NEW model ({len(missing)}):")
            for k in missing[:10]: print(f"  - {k}")
            if len(missing) > 10: print(f"  ... and {len(missing)-10} more.")
        else:
            print("No WavLM keys are missing in the NEW model.")

        if extra:
            print(f"EXTRA in NEW model WavLM section ({len(extra)}):")
            for k in extra[:10]: print(f"  + {k}")
        print("----------------------------------------------------\n")


    def _find_audio_path(self, audio_name: str) -> str:
        """Searches predefined folders for the audio file."""
        file_name = f"{audio_name}.wav"
        for folder in self.INPUT_FOLDERS:
            full_path = os.path.join(folder, file_name)
            if os.path.isfile(full_path):
                
                return full_path
        raise FileNotFoundError(f"Audio file '{file_name}' not found in any predefined folder.")

    def _load_audio(self, file_path: str):
        """Loads and pre-processes a single audio file."""
        audio_input, sr = sf.read(file_path)
        audio_input = torch.from_numpy(audio_input).float()

        if audio_input.dim() > 1:
            audio_input = audio_input[0, :]

        if self.cfg.normalize:
            audio_input = F.layer_norm(audio_input, audio_input.shape)
        return audio_input, sr

    def extract_features(
        self, 
        audio_names: Union[str, List[str], Tuple[str, ...]],
         model_type: str = "new"
    ) -> Dict[str, Union[List[torch.Tensor], List[str]]]:
        if isinstance(audio_names, str):
            audio_names = [audio_names]

        audio_inputs = []
        for name in audio_names:
            path = self._find_audio_path(name)
            audio, _ = self._load_audio(path)
            audio_inputs.append(audio)
            
        if model_type == "new" and self.model_new is not None:
            model = self.model_new
        elif self.model_old is not None:
            model = self.model_old
        else:
            raise RuntimeError("The requested model has been unloaded or was never loaded.")
        
        padded_batch = pad_sequence(audio_inputs, batch_first=True)
        padded_batch = padded_batch.to(self.device)

        with torch.no_grad():
            final_features, layer_results_list = model.extract_features(
                padded_batch,
                output_layer=model.cfg.encoder_layers,
                ret_layer_results=True
            )
        layer_reps = [x for x, _ in final_features[1][1:]] 
        layer_reps = torch.stack(layer_reps, dim=0)
        
        del final_features, layer_results_list, padded_batch,audio_inputs
        return layer_reps

def normalize_uttid(uttids: Union[str, List[str]]) -> Union[str, List[str]]:
    """Normalize utterance IDs by stripping timestamps and collapsing double dashes."""
    if isinstance(uttids, str):
        base = uttids.split("_")[0]
        return base.replace("--", "-")
    elif isinstance(uttids, list):
        return [normalize_uttid(u) for u in uttids]
    else:
        raise TypeError(f"Expected str or list of str, got {type(uttids)}")("--", "-")




class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training."""

    def init_optimizers(self):
        """Initialize optimizer based on config: 'muon' or 'adam'."""
        opt_type = getattr(self.hparams, "optimizer_type", "adam").lower()

        if opt_type == "muon":
            if MuonWithAuxAdam is None:
                raise ImportError(
                    "optimizer_type is 'muon' but MuonWithAuxAdam could not be imported. "
                    "Make sure the Muon package is installed."
                )
            print("Initializing MuonWithAuxAdam Optimizer...")

            hidden_params = []
            other_params = []

            for name, param in self.modules.named_parameters():
                if not param.requires_grad:
                    continue

                if "embedding_model" in name and param.ndim >= 2:
                    hidden_params.append(param)
                else:
                    other_params.append(param)

            print(f"Muon Params (Hidden >= 2D): {len(hidden_params)}")
            print(f"AdamW Params (Others):      {len(other_params)}")

            muon_lr = getattr(self.hparams, "muon_lr", 0.02)
            adamw_lr = self.hparams.lr
            param_groups = [
                {
                    "params": hidden_params,
                    "use_muon": True,
                    "lr": muon_lr,
                    "momentum": 0.95,
                    "weight_decay": 0.01,
                },
                {
                    "params": other_params,
                    "use_muon": False,
                    "lr": adamw_lr,
                    "betas": (0.9, 0.95),
                    "weight_decay": 0.01,
                },
            ]

            self.optimizer = MuonWithAuxAdam(param_groups)

        else:
            print("Initializing Adam Optimizer...")
            import torch.optim as optim

            optimizer_cls = getattr(self.hparams, "opt_class", optim.Adam)
            aam = optimizer_cls(
                self.modules.parameters(),
                lr=self.hparams.lr,
                weight_decay=0.000002,
            )
            self.optimizer = aam

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def frame_shuffle_fast(self, wavs, frame_len=400, frame_shift=160):
        B, T = wavs.shape
        num_frames = 1 + (T - frame_len) // frame_shift

        framed = wavs.unfold(dimension=1, size=frame_len, step=frame_shift)

        torch.manual_seed(42)
        perm = torch.randperm(num_frames)
        framed_shuffled = framed[:, perm, :]

        recon = torch.zeros((B, T), device=self.device)
        ones = torch.ones((B, num_frames, frame_len), device=self.device)

        idx = (
            torch.arange(0, num_frames * frame_shift, frame_shift, device=self.device)
            .unsqueeze(1) + torch.arange(frame_len, device=self.device)
        )
        idx = idx.unsqueeze(0).expand(B, -1, -1)

        recon = recon.scatter_add(1, idx.reshape(B, -1), framed_shuffled.reshape(B, -1))

        overlap = torch.zeros((B, T), device=self.device)
        overlap = overlap.scatter_add(1, idx.reshape(B, -1), ones.reshape(B, -1))
        recon = recon / torch.clamp(overlap, min=1.0)

        return recon

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig
       
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
           wavs, lens = self.hparams.wav_augment(wavs, lens)

        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        uttid = normalize_uttid(batch.id)
        get_wav2vec2features = self.wavlm_extractor.extract_features(uttid)
        embeddings = self.modules.embedding_model(feats, get_wav2vec2features)
        self.emb = embeddings
        outputs = self.modules.classifier(embeddings)
        
        return outputs, lens
    
    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label."""
        predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            spkid = self.hparams.wav_augment.replicate_labels(spkid)

        loss_aam = self.hparams.compute_cost(predictions, spkid, lens)
        loss =  loss_aam

        if stage == sb.Stage.TRAIN and hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss
    
    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        for module in [self.modules.compute_features, self.modules.mean_var_norm,
                           self.modules.embedding_model, self.modules.classifier]:
            for p in module.parameters():
                p.requires_grad = True

        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                num_to_keep=7,
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
                name=epoch
            )
            self.checkpointer.save_checkpoint(
                meta={"epoch": epoch, "loss": stage_loss},
                name=f"epoch_{epoch}_latest",
            )
            
            
    def fit_batch(self, batch):
        """Fit one batch."""
        should_step = (self.step % self.grad_accumulation_factor) == 0
        self.on_fit_batch_start(batch, should_step)

        with self.no_sync(not should_step):
            with self.training_ctx:
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            scaled_loss = self.scaler.scale(
                loss / self.grad_accumulation_factor
            )
            self.check_loss_isfinite(scaled_loss)
            scaled_loss.backward()

        if should_step:
            self.optimizers_step()

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()


def dataio_prep(hparams):
    print("=== [Running prepare_libri ===")
    run_on_main(
        prepare_libri,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": ["train", "dev"],
            "split_ratio": [95, 5],
            "num_utt": hparams["num_utt"],
            "num_spk": hparams["num_spk"],
            "seg_dur": hparams["sentence_len"],
            "skip_prep": hparams["skip_prep"],
            "utt_selected_ways": hparams["utt_selected_ways"],
        },
    )
    
    full_df = pd.read_csv(hparams['train_annotation'])
    full_df = full_df.sample(frac=1.0, random_state=24)

    client_csv_path = os.path.join(hparams['save_folder'], f'train_client.csv')
    full_df.to_csv(client_csv_path, index=False)

    asv_dataset_gen = ASVDatasetGenerator(hparams, client_csv_path)
    client_train, valid_data = asv_dataset_gen.dataio_prep()

    return client_train, valid_data, None



class ECAPATDNNClient():
    def __init__(self, model: SpeakerBrain, train_data, valid_data):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        print(f"Train data size: {len(self.train_data)}")

    def fit(self):
        train_kwargs = self.model.hparams.dataloader_options
        valid_kwargs = self.model.hparams.dataloader_options
        train_kwargs["drop_last"] = True
        print(f"Starting training with {len(self.train_data)} samples")
        print(f"Batch size: {self.model.hparams.dataloader_options.get('batch_size', 'N/A')}")

        self.model.fit(
            self.model.hparams.epoch_counter,
            self.train_data,
            self.valid_data,
            train_loader_kwargs=train_kwargs,
            valid_loader_kwargs=valid_kwargs,
        )


    def evaluate(self):
        valid_kwargs = self.model.hparams.dataloader_options

        loss = self.model.evaluate(
            self.valid_data,
            valid_loader_kwargs=valid_kwargs,
        )
        return float(loss), len(self.valid_data), {}


if __name__ == "__main__":
    default_config = config_path

    if len(sys.argv) == 1 or sys.argv[1].startswith("-"):
        print(f"No param_file provided, defaulting to: {default_config}")
        sys.argv.insert(1, default_config)

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file, encoding="utf-8") as f:
        train_params = load_hyperpyyaml(f)

    overrides_dict = {
        'pretrained_path': str(train_params['pretrained_model']),
        'batch_size': train_params['batch_size'],
        'lr': train_params['lr'],
        'optimizer_type': train_params['optimizer_type'],
        'muon_lr': train_params['muon_lr'],
        'num_utt': train_params['num_utt'],
        'num_spk': train_params['num_spk'],
        'utt_selected_ways': train_params['utt_selection'],
        'number_of_epochs': train_params['epochs'],
        'data_folder': str(train_params['train_data_dir']),
        'output_folder': str(train_params['model_dir']),
        'num_workers': train_params['num_workers'],
    }

    overrides_dict['out_n_neurons'] = (
        6020 if train_params['num_spk'] == 'ALL' else int(train_params['num_spk'])
    )

    hparam_file_inner = train_params['train_config']
    with open(hparam_file_inner, encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f, overrides_dict)

    print("Data Prep Done")

    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=None,
    )


    sb.utils.distributed.ddp_init_group(run_opts)

    run_opts["find_unused_parameters"] = True

    device = run_opts.get("device", "cpu")
    twin = ECAPA_TDNN_test().to(device)

    pretrained_path = train_params.get('pretrained_model')

    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)
            state_dict = checkpoint.get('state_dict', checkpoint)
            state_dict = { k[len('module.'):] if k.startswith('module.') else k : v
                        for k, v in state_dict.items() }

            missing, unexpected = twin.load_state_dict(state_dict, strict=True)
            print(f"Pretrained model loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            if len(missing) > 0:
                print(f"Missing keys details: {missing}")
        except Exception as e:
            print(f"ERROR: Failed to load pretrained model: {e}")
    else:
        if pretrained_path:
            print(f"WARNING: Pretrained path '{pretrained_path}' not found. Training from random initialization.")
        else:
            print("No pretrained_model set in config. Training from random initialization.")

    print("\n=== Debug: Layer Frozen Status ===")
    frozen_count = 0
    trainable_count = 0
    for name, param in twin.named_parameters():
        if not param.requires_grad:
            print(f"[FROZEN] {name}")
            frozen_count += 1
        else:
            trainable_count += 1
    print(f"Total Frozen Layers: {frozen_count}")
    print(f"Total Trainable Layers: {trainable_count}")
    print("==================================\n")

    hparams['modules']['embedding_model'] = twin
    hparams['checkpointer'].add_recoverable('embedding_model', twin)
    run_opts["grad_accumulation_factor"] = 8
    speaker = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    INPUT_FOLDERS = train_params.get("input_audio_folders", [])
    wavlm_path = train_params.get("wavlm_model_path", "wavlm/WavLM-Large.pt")
    wavlm_ = WavLMFeatureExtractor(wavlm_path, device=device, INPUT_FOLDERS=INPUT_FOLDERS)
    speaker.wavlm_extractor = wavlm_

    train_ds, valid_ds, _ = dataio_prep(hparams)

    ecapa_train = ECAPATDNNClient(speaker, train_ds, valid_ds)
    ecapa_train.fit()
