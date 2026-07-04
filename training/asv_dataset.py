import os
import random
import torchaudio
import torch
import pandas as pd
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset


class ASVDatasetGenerator:
    def __init__(self, hparams, train_csv=None):
        self.sample_rate = hparams['sample_rate']
        self.data_folder = hparams['data_folder']
        self.save_folder = hparams['save_folder']
        self.dev_annotation = hparams['valid_annotation']
        self.train_annotation = train_csv if train_csv is not None else hparams['train_annotation']
        self.lab_enc_file = os.path.join(self.save_folder, 'label_encoder.txt')

    def dataio_prep(self):
        # 1. Load datasets
        train_dataset = DynamicItemDataset.from_csv(
            csv_path=self.train_annotation,
            replacements={'data_root': self.data_folder}
        )

        dev_dataset = DynamicItemDataset.from_csv(
            csv_path=self.dev_annotation,
            replacements={'data_root': self.data_folder}
        )

        datasets = [train_dataset, dev_dataset]

        # 2. Audio pipeline (assumes utterances already segmented during prepare_libri)
        @sb.utils.data_pipeline.takes('wav')
        @sb.utils.data_pipeline.provides('sig')
        def audio_pipeline(wav):
            sig, fs = torchaudio.load(wav)
            return sig.squeeze(0)

        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

        # 3. Label pipeline
        label_encoder = sb.dataio.encoder.CategoricalEncoder()
        label_encoder.ignore_len()

        @sb.utils.data_pipeline.takes('spk_id')
        @sb.utils.data_pipeline.provides('spk_id', 'spk_id_encoded')
        def label_pipeline(spk_id):
            yield spk_id
            encoded = label_encoder.encode_sequence_torch([spk_id])
            yield encoded

        sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

        # 4. Fit or load encoder
        label_encoder.load_or_create(
            path=self.lab_enc_file,
            from_didatasets=[train_dataset],
            output_key='spk_id'
        )

        # 5. Output keys
        sb.dataio.dataset.set_output_keys(datasets, ['id', 'sig', 'spk_id_encoded'])

        return train_dataset, dev_dataset
