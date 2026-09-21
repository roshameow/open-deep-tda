"""Opt-in public dataset downloads with provenance; no remote model/data upload."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import numpy as np

HAR_URL = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
# Observed official-download digest, recorded for reproducibility, not a publisher signature.
HAR_SHA256 = 'c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031'
HAR_CITATION = 'Reyes-Ortiz et al. (2013), Human Activity Recognition Using Smartphones, UCI, DOI:10.24432/C54S4K'


def _validate_har(data):
    for split, n in [('train', 7352), ('test', 2947)]:
        X, y, subjects = data['X_'+split], data['y_'+split], data['subjects_'+split]
        if X.shape != (n, 561) or y.shape != (n,) or subjects.shape != (n,):
            raise ValueError('HAR shape does not match the original official split')
        if not np.isfinite(X).all() or set(np.unique(y)) != set(range(1, 7)):
            raise ValueError('HAR contains invalid feature values or activity labels')
        if np.any(subjects < 1) or np.any(subjects > 30):
            raise ValueError('invalid HAR subject IDs')
    a, b = set(data['subjects_train']), set(data['subjects_test'])
    if a & b or len(a) != 21 or len(b) != 9:
        raise ValueError('HAR official subject-disjoint split was not preserved')


def load_uci_har(cache_dir='data/uci_har', download=False):
    """Return the full 7352/2947 official subject-disjoint, 561-feature split.

    Labels and subjects are returned separately, never concatenated to features.
    Files are read by exact archive member name; zip paths are never extracted.
    Download is opt-in; cache excludes raw inertial signal tensors from parsing.
    """
    root = Path(cache_dir)
    archive = root / 'human_activity_recognition_using_smartphones.zip'
    cache = root / 'features.npz'
    if not archive.exists():
        if not download:
            raise FileNotFoundError(f'{archive} missing; pass download=True to fetch the public UCI data')
        root.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix('.download')
        try:
            with urllib.request.urlopen(HAR_URL, timeout=90) as response, temporary.open('wb') as out:
                while True:
                    block = response.read(1024*1024)
                    if not block:
                        break
                    out.write(block)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != HAR_SHA256:
                raise ValueError('official archive digest differs; inspect before changing the pinned checksum')
            temporary.replace(archive)
        finally:
            if temporary.exists():
                temporary.unlink()
    if hashlib.sha256(archive.read_bytes()).hexdigest() != HAR_SHA256:
        raise ValueError('HAR archive checksum mismatch')
    if cache.exists():
        with np.load(cache, allow_pickle=False) as values:
            data = {key: values[key] for key in values.files}
    else:
        data = {}
        with zipfile.ZipFile(archive) as outer:
            with zipfile.ZipFile(io.BytesIO(outer.read('UCI HAR Dataset.zip'))) as inner:
                for split in ('train', 'test'):
                    for key, filename, dtype in [('X', 'X', np.float32), ('y', 'y', np.int64),
                                                  ('subjects', 'subject', np.int64)]:
                        name = f'UCI HAR Dataset/{split}/{filename}_{split}.txt'
                        with inner.open(name) as stream:
                            data[key+'_'+split] = np.loadtxt(stream, dtype=dtype)
        _validate_har(data)
        np.savez_compressed(cache, **data)
    _validate_har(data)
    data['metadata'] = {'name': 'UCI HAR original 561-feature dataset', 'url': HAR_URL,
        'archive_sha256': HAR_SHA256, 'digest_kind': 'observed download checksum',
        'citation': HAR_CITATION, 'license': 'CC BY 4.0 per UCI dataset page',
        'train_subjects': sorted(int(v) for v in np.unique(data['subjects_train'])),
        'test_subjects': sorted(int(v) for v in np.unique(data['subjects_test'])),
        'split': 'official subject-disjoint split; not shuffled overlapping-window splitting'}
    return data
