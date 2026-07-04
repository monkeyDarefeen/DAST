# This code is based on
# https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/voxceleb_prepare.py
import csv
import os
import logging
import random
from pathlib import Path
import numpy as np
import torch
import torchaudio
import soundfile as sf
from tqdm.contrib import tqdm
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)

from utils import read_kaldi_format

logger = logging.getLogger(__name__)
OPT_FILE = "opt_libri_prepare.pkl"
TRAIN_CSV = "train.csv"
DEV_CSV = "dev.csv"
ENROL_CSV = "enrol.csv"
SAMPLERATE = 16000

def prepare_libri(
    data_folder,
    save_folder,
    splits=["train", "dev"],
    split_ratio=[90, 10],
    seg_dur=3.0,
    amp_th=5e-04,
    num_utt=None,
    num_spk=None,
    random_segment=False,
    skip_prep=False,
    utt_selected_ways="spk-random",
):
    """
    Prepares the csv files for the libri datasets.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original libri  dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    verification_pairs_file : str
        txt file containing the verification split.
    splits : list
        List of splits to prepare from ['train', 'dev']
    split_ratio : list
        List if int for train and validation splits
    seg_dur : int
        Segment duration of a chunk in seconds (e.g., 3.0 seconds).
    amp_th : float
        removes segments whose average amplitude is below the
        given threshold.
    source : str
        Path to the folder where the VoxCeleb dataset source is stored.
    num_utt: float
        How many utterances for each speaker used for training
    num_spk: float
        How many speakers used for training
    random_segment : bool
        Train random segments
    skip_prep: Bool
        If True, skip preparation.

    Example
    -------
    >>> from libri_prepare import prepare_libri
    >>> data_folder = 'LibriSpeech/train-clean-360/'
    >>> save_folder = 'libri/'
    >>> splits = ['train', 'dev']
    >>> split_ratio = [90, 10]
    >>> prepare_libri(data_folder, save_folder, splits, split_ratio)
    """
    
    if skip_prep:
        return

    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True, parents=True)

    # Setting ouput files
    save_opt = save_folder / OPT_FILE
    save_csv_train = save_folder / TRAIN_CSV
    save_csv_dev = save_folder / DEV_CSV

    conf = locals()
    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, locals()):
        logger.info("Skipping preparation, completed in previous run.")
        return

    # Additional checks to make sure the data folder contains VoxCeleb data
    if "," in data_folder:
        data_folder = [Path(dir) for dir in data_folder.replace(" ", "").split(",")]
    else:
        data_folder = [Path(data_folder)]

    logger.info("Creating csv file for the Libri Dataset...")

    # Split data into 90% train and 10% validation (verification split)
    wav_lst_train, wav_lst_dev, utt2spk = _get_utt_split_lists(
        data_folder, split_ratio, num_utt, num_spk, utt_selected_ways
    )

    # Creating csv file for training data
    if "train" in splits:
        prepare_csv(
            seg_dur, wav_lst_train, utt2spk, save_csv_train, random_segment, amp_th
        )

    if "dev" in splits:
        prepare_csv(seg_dur, wav_lst_dev, utt2spk, save_csv_dev, random_segment, amp_th)


    # Saving options (useful to skip this phase when already done)
    save_pkl(conf, str(save_opt))


# Used for verification split
def _get_utt_split_lists(
    data_folders, split_ratio, num_utt='ALL', num_spk='ALL', utt_selected_ways="spk-random"
):
    """
    Tot. number of speakers libri-360=921
    Splits the audio file list into train and dev.
    """
    train_lst = []
    dev_lst = []

    logger.debug("Getting file list...")

    out_utt2spk = []
    for data_folder in data_folders:
        spk2utt = read_kaldi_format(data_folder / 'spk2utt')
        utt2spk = read_kaldi_format(data_folder / 'utt2spk')
        spk_files = read_kaldi_format(data_folder / 'wav.scp')
        out_utt2spk += utt2spk
        spks_pure = spk2utt.keys()
        full_utt = len(spks_pure)

        selected_list = []
        selected_spk = {}
        #select the number of speakers
        if num_spk != 'ALL':
            logger.info(f"selected {num_spk} speakers for training")
            selected_spks_pure = random.sample(spks_pure, int(num_spk))
            for k,v in spk_files.items():
                if utt2spk[k] in selected_spks_pure:
                    selected_spk[k] = v
        elif num_spk == 'ALL':
            logger.info(f"selected {len(utt2spk)} (all) speakers speakers for training")
            selected_spk = spk_files
        else:
            raise ValueError(f"invalid {num_spk} value")

            # select the number of utterances for each speaker-sess-id
        if num_utt != 'ALL':
            # select the number of utterances for each speaker-sess-id
            if utt_selected_ways == 'spk-sess':
                logger.info(f"selected {num_utt} utterances for each selected speaker-sess-id")
                for spk in selected_spk:
                    if len(selected_spk[spk]) >= int(num_utt):
                        selected_list.extend(random.sample(selected_spk[spk], int(num_utt)))
                    else:
                        selected_list.extend(selected_spk[spk])
            elif utt_selected_ways == 'spk-random':
                logger.info(f"randomly selected {num_utt} utterances for each selected speaker-id")
                selected_spks_pure = {}
                for k, v in selected_spk.items():
                    spk_pure = utt2spk[k]
                    if spk_pure not in selected_spks_pure:
                        selected_spks_pure[spk_pure] = []
                    selected_spks_pure[spk_pure].extend(v)
                selected_spk = selected_spks_pure
                for spk in selected_spk:
                    if len(selected_spk[spk]) >= int(num_utt):
                        selected_list.extend(random.sample(selected_spk[spk], int(num_utt)))
                    else:
                        selected_list.extend(selected_spk[spk])
            elif utt_selected_ways == 'spk-diverse-sess':
                logger.info(f"diversely selected {num_utt} utterances for each selected speaker-id" % num_utt)
                selected_spks_pure = {}
                for k, v in selected_spk.items():
                    spk_pure = utt2spk[k]
                    if spk_pure not in selected_spks_pure:
                        selected_spks_pure[spk_pure] = []
                    selected_spks_pure[spk_pure].append(v)
                selected_spk = selected_spks_pure
                for spk in selected_spk:
                    num_each_sess = round(num_utt / len(selected_spk[spk]))  # rounded up
                    for utts in selected_spk[spk]:
                        if len(utts) >= int(num_each_sess):
                            selected_list.extend(random.sample(utts, int(num_each_sess)))
                        else:
                            selected_list.extend(selected_spk[spk])
        elif num_utt == 'ALL':
            logger.info("selected all utterances for each selected speaker")
            
            # 1. Use a Set for O(1) lookup of items to ignore
            ignore_tokens = {"flac", "-c", "-d", "-s", "|"}
            
            # 2. Use a Set to automatically handle deduplication
            unique_items = set()

            for value in selected_spk.values():
                # Handle both lists and single items uniformly
                # Avoid string parsing types; use isinstance
                items = value if isinstance(value, list) else [value]
                
                for item in items:
                    # Check against ignore set and add to unique set
                    if item not in ignore_tokens:
                        unique_items.add(item)

            # 3. Convert back to list and shuffle
            selected_list = list(unique_items)
            random.shuffle(selected_list)

        else:
            raise ValueError(f"invalid {num_utt} value")

        logger.info(f'Full training set:{full_utt}')
        logger.debug(f'Used for training:{len(selected_list)}')

        split = int(0.01 * split_ratio[0] * len(selected_list))
        train_snts = selected_list[:split]
        dev_snts = selected_list[split:]

        train_lst.extend(train_snts)
        if os.getenv('VPC_TEST_TOOLS', 'False').lower() == "true": # For testing only!
            logger.critical("Train list contains dev speakers (Use this only for testing purposes)!!")
            train_lst.extend(dev_snts)
        dev_lst.extend(dev_snts)

    return train_lst, dev_lst, out_utt2spk


def _get_chunks(seg_dur, audio_id, audio_duration):
    """
    Returns list of chunks
    """
    num_chunks = int(audio_duration / seg_dur)  # all in milliseconds
    chunk_lst = [
        audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
        for i in range(num_chunks)
    ]
    return chunk_lst


def prepare_csv(seg_dur, wav_lst, utt2spk, csv_file, random_segment=False, amp_th=0):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    csv_file : str
        The path of the output csv file
    random_segment: bool
        Read random segments
    amp_th: float
        Threshold on the average amplitude on the chunk.
        If under this threshold, the chunk is discarded.

    Returns
    -------
    None
    """
    
    logger.info(f"Creating csv lists in {csv_file}...")

    csv_output = [["ID", "duration", "wav", "start", "stop", "spk_id"]]

    # For assigning unique ID to each chunk
    my_sep = "--"
    entry = []
    problematic_wavs = []
    # Processing all the wav files in the list
    for wav_file in tqdm(wav_lst, dynamic_ncols=True):
        # Getting sentence and speaker ids
        # TODO use utt2spk (but with the current impl, wav_file loses it's uniq id)
        try:
            temp = wav_file.split("/")[-1].split(".")[0]
            [spk_id, sess_id, utt_id] = temp.split('-')[-3:]
        except ValueError:
            logger.info(f"Malformed path: {wav_file}")
            continue
        audio_id = my_sep.join([spk_id, sess_id, utt_id.split(".")[0]])

        # Reading the signal (to retrieve duration in seconds)
        try:
            audio_duration = sf.info(wav_file).duration
            #signal, fs = torchaudio.load(wav_file)
        except RuntimeError:
            problematic_wavs.append(wav_file)
            continue
        #signal = signal.squeeze(0)

        if random_segment:
            #audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            #stop_sample = signal.shape[0]
            stop_sample = int(audio_duration * SAMPLERATE)

            # Composition of the csv_line
            csv_line = [
                audio_id,
                str(audio_duration),
                wav_file,
                start_sample,
                stop_sample,
                spk_id,
            ]
            entry.append(csv_line)
        else:
            #audio_duration = signal.shape[0] / SAMPLERATE
            signal, fs = torchaudio.load(wav_file)
            signal = signal.squeeze(0)

            uniq_chunks_list = _get_chunks(seg_dur, audio_id, audio_duration)
            for chunk in uniq_chunks_list:
                s, e = chunk.split("_")[-2:]
                start_sample = int(float(s) * SAMPLERATE)
                end_sample = int(float(e) * SAMPLERATE)

                #  Avoid chunks with very small energy
                mean_sig = torch.mean(np.abs(signal[start_sample:end_sample]))
                if mean_sig < amp_th:
                    continue

                # Composition of the csv_line
                csv_line = [
                    chunk,
                    str(audio_duration),
                    wav_file,
                    start_sample,
                    end_sample,
                    spk_id,
                ]
                entry.append(csv_line)

    logger.info(f'Skipped {len(problematic_wavs)} invalid audios')
    csv_output = csv_output + entry

    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)



# import csv
# import logging
# import os
# import soundfile as sf
# import torch
# import torchaudio
# import numpy as np
# from tqdm import tqdm
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import multiprocessing as mp

# # Use 'spawn' to avoid CUDA fork-safety issues (children start clean)
# # uncommment this for faster processing
# # try:
# #     mp.set_start_method("spawn", force=True)
# # except RuntimeError:
# #     pass  # already set

# # Setup logger
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # --- 1. Your Exact Chunk Logic (Reverted) ---
# def _get_chunks(seg_dur, audio_id, audio_duration):
#     num_chunks = int(audio_duration / seg_dur)
#     chunk_lst = [
#         # Reverted to str() to match your exact output format
#         audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
#         for i in range(num_chunks)
#     ]
#     return chunk_lst

# # --- 2. Worker Function ---
# def process_single_file(wav_file, random_segment, seg_dur, amp_th, sample_rate):
#     my_sep = "--"
#     rows = []
    
#     if not os.path.exists(wav_file):
#         return None

#     # --- LOGIC MATCH: ID Parsing ---
#     try:
#         # We use os.path.basename to be safe across OS, but keep your split logic
#         temp = os.path.basename(wav_file).replace('.wav', '').replace('.flac', '')
        
#         # STRICT: This attempts to unpack exactly 3 items. 
#         # If the filename doesn't have enough hyphens, this raises ValueError
#         # and goes to the except block (Skipping the file), matching your original code.
#         [spk_id, sess_id, utt_id] = temp.split('-')[-3:]
        
#         audio_id = my_sep.join([spk_id, sess_id, utt_id])
#     except ValueError:
#         # Returns None, effectively skipping the file (same as 'continue')
#         return None

#     # --- LOGIC MATCH: Duration Reading ---
#     try:
#         info = sf.info(wav_file)
#         audio_duration = info.duration
#     except Exception:
#         return None

#     if random_segment:
#         start_sample = 0
#         stop_sample = int(audio_duration * sample_rate)
        
#         rows.append([
#             audio_id, str(audio_duration), wav_file, 
#             start_sample, stop_sample, spk_id
#         ])
#     else:
#         # STRICT: Only load audio if amp_th is set, OR if you want to be 100% 
#         # sure the file isn't corrupt (since your original loaded it here).
#         # However, purely for logic, if amp_th=0, your original code
#         # loaded 'signal' but never used it for filtering.
#         # We optimize this by only loading if needed. 
        
#         signal = None
#         if amp_th > 0:
#             try:
#                 signal, fs = torchaudio.load(wav_file)
#                 signal = signal.squeeze(0)
#             except:
#                 return None # Skip if audio load fails

#         uniq_chunks_list = _get_chunks(seg_dur, audio_id, audio_duration)
        
#         for chunk in uniq_chunks_list:
#             s, e = chunk.split("_")[-2:]
            
#             # Use the PASSED sample_rate, not a hardcoded one
#             start_sample = int(float(s) * sample_rate)
#             end_sample = int(float(e) * sample_rate)

#             if amp_th > 0:
#                 # We already checked amp_th > 0 before loading signal
#                 if signal is not None:
#                     if start_sample < signal.shape[0]:
#                         actual_end = min(end_sample, signal.shape[0])
#                         seg_sig = signal[start_sample:actual_end]
#                         if seg_sig.numel() > 0:
#                             mean_sig = torch.mean(torch.abs(seg_sig))
#                             if mean_sig < amp_th:
#                                 continue
#                         else:
#                             continue

#             rows.append([
#                 chunk, str(audio_duration), wav_file,
#                 start_sample, end_sample, spk_id
#             ])
            
#     return rows

# # --- 3. Manager Function ---
# def prepare_csv(seg_dur, wav_lst, utt2spk, csv_file, random_segment=False, amp_th=0):
#     """
#     Added sample_rate argument to ensure calculations match your global SAMPLERATE.
#     """
#     logger.info(f"Creating csv lists in {csv_file}...")
    
#     header = ["ID", "duration", "wav", "start", "stop", "spk_id"]
#     with open(csv_file, mode="w", newline='') as csv_f:
#         csv_writer = csv.writer(csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         csv_writer.writerow(header)

#     max_workers = min(os.cpu_count(),32)
#     logger.info(f"Processing {len(wav_lst)} files with {max_workers} workers...")
    
#     # Counter for skipped files (for verification)
#     skipped_count = 0
    
#     with open(csv_file, mode="a", newline='') as csv_f:
#         csv_writer = csv.writer(csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             futures = []
#             for wav in wav_lst:
#                 # Pass sample_rate explicitly
#                 futures.append(executor.submit(
#                     process_single_file, wav, random_segment, seg_dur, amp_th, sample_rate=16000
#                 ))
            
#             for future in tqdm(as_completed(futures), total=len(wav_lst), dynamic_ncols=True):
#                 result_rows = future.result()
#                 if result_rows:
#                     for row in result_rows:
#                         csv_writer.writerow(row)
#                 else:
#                     skipped_count += 1

#     logger.info(f"CSV generation complete. Skipped {skipped_count} invalid/malformed files.")


def skip(splits, save_folder, conf):
    """
    Detects if the voxceleb data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking csv files
    skip = True

    split_files = {
        "train": TRAIN_CSV,
        "dev": DEV_CSV,
    }
    for split in splits:
        if not Path(save_folder, split_files[split]).is_file():
            skip = False
    #  Checking saved options
    save_opt = save_folder / OPT_FILE
    if skip is True:
        if save_opt.is_file():
            opts_old = load_pkl(str(save_opt))

            if opts_old.popitem() == conf.popitem():
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip
