import os.path as op
import mmap
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union, List

import numpy as np
import torchaudio

FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS = {".npy", ".wav", ".flac", ".ogg"}

def get_waveform(
    path_or_fp: Union[str, BinaryIO], normalization=True
) -> Tuple[np.ndarray, int]:
    """Get the waveform and sample rate of a 16-bit mono-channel WAV or FLAC.

    Args:
        path_or_fp (str or BinaryIO): the path or file-like object
        normalization (bool): Normalize values to [-1, 1] (Default: True)
    """
    ext = ".wav"
    if isinstance(path_or_fp, str):
        ext = op.splitext(op.basename(path_or_fp))[1]
        if ext not in {".flac", ".wav", ".mp3"}:
            raise ValueError(f"Unsupported audio format: {ext}")

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile to load WAV/FLAC file")
    if ext in {".flac", ".wav"}:
        waveform, sample_rate = sf.read(path_or_fp, dtype="float32")
    else: # mp3 file, eg. commonvoice
        waveform, sr = torchaudio.load(path_or_fp)
        if sr != 16000:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resample(waveform).view(-1).numpy()
            sample_rate = 16000
    if not normalization:
        waveform *= 2 ** 15  # denormalized to 16-bit signed integers
    return waveform, sample_rate


def get_segment_waveform(
        path_or_fp, offset, n_frames, normalization=True
):
    if isinstance(path_or_fp, str):
        ext = op.splitext(op.basename(path_or_fp))[1]
        if ext not in {".flac", ".wav"}:
            raise ValueError(f"Unsupported audio format: {ext}")

    waveform, sample_rate = torchaudio.load(path_or_fp, frame_offset=offset, num_frames=n_frames)
    if not normalization:
        waveform *= 2 ** 15
    return waveform, sample_rate


def _get_kaldi_fbank(waveform, sample_rate, n_bins=80) -> Optional[np.ndarray]:
    """Get mel-filter bank features via PyKaldi."""
    try:
        from kaldi.feat.mel import MelBanksOptions
        from kaldi.feat.fbank import FbankOptions, Fbank
        from kaldi.feat.window import FrameExtractionOptions
        from kaldi.matrix import Vector

        mel_opts = MelBanksOptions()
        mel_opts.num_bins = n_bins
        frame_opts = FrameExtractionOptions()
        frame_opts.samp_freq = sample_rate
        opts = FbankOptions()
        opts.mel_opts = mel_opts
        opts.frame_opts = frame_opts
        fbank = Fbank(opts=opts)
        features = fbank.compute(Vector(waveform), 1.0).numpy()
        return features
    except ImportError:
        return None


def _get_torchaudio_fbank(waveform, sample_rate, n_bins=80) -> Optional[np.ndarray]:
    """Get mel-filter bank features via TorchAudio."""
    try:
        import torch
        import torchaudio.compliance.kaldi as ta_kaldi

        waveform = torch.from_numpy(waveform).unsqueeze(0)
        features = ta_kaldi.fbank(
            waveform, num_mel_bins=n_bins, sample_frequency=sample_rate
        )
        return features.numpy()
    except ImportError:
        return None


def get_fbank(path_or_fp: Union[str, BinaryIO], n_bins=80) -> np.ndarray:
    """Get mel-filter bank features via PyKaldi or TorchAudio. Prefer PyKaldi
    (faster CPP implementation) to TorchAudio (Python implementation). Note that
    Kaldi/TorchAudio requires 16-bit signed integers as inputs and hence the
    waveform should not be normalized."""
    sound, sample_rate = get_waveform(path_or_fp, normalization=False)

    features = _get_kaldi_fbank(sound, sample_rate, n_bins)
    if features is None:
        features = _get_torchaudio_fbank(sound, sample_rate, n_bins)
    if features is None:
        raise ImportError(
            "Please install pyKaldi or torchaudio to enable "
            "online filterbank feature extraction"
        )

    return features

def mmap_read(path: str, offset: int, length: int) -> bytes:
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_o:
            data = mmap_o[offset : offset + length]
    return data

def read_from_stored_zip(zip_path: str, offset: int, length: int) -> bytes:
    return mmap_read(zip_path, offset, length)

def parse_path(path: str) -> Tuple[str, List[int]]:
    """Parse data path which is either a path to
    1. a .npy/.wav/.flac/.ogg file
    2. a stored ZIP file with slicing info: "[zip_path]:[offset]:[length]"

      Args:
          path (str): the data path to parse

      Returns:
          file_path (str): the file path
          slice_ptr (list of int): empty in case 1;
            byte offset and length for the slice in case 2
    """

    if Path(path).suffix in FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS:
        _path, slice_ptr = path, []
    else:
        _path, *slice_ptr = path.split(":")
        if not Path(_path).is_file():
            raise FileNotFoundError(f"File not found: {_path}")
    assert len(slice_ptr) in {0, 2}, f"Invalid path: {path}"
    slice_ptr = [int(i) for i in slice_ptr]
    return _path, slice_ptr