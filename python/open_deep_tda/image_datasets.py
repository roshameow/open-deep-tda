"""Full, opt-in image datasets; uint8 only, with no torchvision dependency.

Only original archives are cached, never extracted or duplicated as float arrays.
Full official files were verified: COIL-20 has 1,440 grayscale 128 x 128 PNGs
(plus one directory entry); Fashion-MNIST has 60,000 train and 10,000 test rows.
Fashion-MNIST MD5 pins are publisher values from its README. COIL's SHA256 is
an observed official-download fingerprint, not a publisher signature. Dataset
terms are independent of this project's software license. Pillow is needed only
when loading COIL-20. Importing this module performs no I/O.
"""
import gzip
import hashlib
import io
import os
from pathlib import Path
import stat
import struct
import tempfile
import time
import urllib.request
import zipfile

import numpy as np


COIL20_PAGE = 'https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php'
COIL20_URL = ('https://www.cs.columbia.edu/CAVE/databases/'
              'SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip')
COIL20_SHA256 = '517c5594820eb40066ba0ff6842e7f09392bf7fde849bf9cb9c28445b0f29e88'
COIL20_CURRENT_PAGE = 'https://cave.cs.columbia.edu/repository/COIL-20'
# Both official pages were inspected, including the current page's JS content:
# they give downloads/citation/contact, but no explicit research-use license.
# Do not substitute another CAVE dataset's terms or this project's MIT license.
FASHION_BASE_URL = ('https://raw.githubusercontent.com/zalandoresearch/'
                    'fashion-mnist/master/data/fashion/')
FASHION_README = 'https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md'
FASHION_LICENSE = 'https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE'
# filename, output key, count, IDX magic, published compressed-file MD5
_FASHION_FILES = (
    ('train-images-idx3-ubyte.gz', 'train_images', 60000, 2051,
     '8d4fb7e6c68d591d4c3dfef9ec88bf0d'),
    ('train-labels-idx1-ubyte.gz', 'train_labels', 60000, 2049,
     '25c81989df183df01b3e8a0aad5dffbe'),
    ('t10k-images-idx3-ubyte.gz', 'test_images', 10000, 2051,
     'bef4ecab320f06d8554ea6380940ec79'),
    ('t10k-labels-idx1-ubyte.gz', 'test_labels', 10000, 2049,
     'bb300cfdad3c16e7a12a480ee83cd310'),
)
_MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
_DOWNLOAD_SECONDS = 600


def _checksums(path, max_bytes):
    """Stream bounded files, including cache hits; never trust a sidecar digest."""
    hashes = {name: hashlib.new(name) for name in ('md5', 'sha256')}
    size = 0
    with Path(path).open('rb') as stream:
        while True:
            block = stream.read(min(1024 * 1024, max_bytes - size + 1))
            if not block:
                break
            size += len(block)
            if size > max_bytes:
                raise ValueError(f'{path}: file exceeds byte limit')
            for digest in hashes.values():
                digest.update(block)
    return dict(size_bytes=size, **{name: h.hexdigest() for name, h in hashes.items()})


def _verify(path, algorithm, expected, max_bytes):
    result = _checksums(path, max_bytes)
    if result[algorithm] != expected:
        raise ValueError(f'{path}: {algorithm} checksum mismatch; remove corrupt file explicitly')
    return result


def _ensure_file(path, url, algorithm, expected, download, max_bytes=_MAX_ARCHIVE_BYTES):
    """Bounded GET into a same-directory temporary file, verify, then atomic rename.

    Existing corrupt files are errors, not silently replaced. Socket operations
    time out after 30 seconds and a transfer has a 600-second wall-clock budget
    (plus at most one socket timeout). No upload or archive extraction occurs.
    """
    path = Path(path)
    if path.exists():
        return _verify(path, algorithm, expected, max_bytes)
    if not download:
        raise FileNotFoundError(f'{path} missing; pass download=True to fetch public data')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.',
                                         suffix='.download', delete=False) as output:
            temporary = Path(output.name)
            start = time.monotonic()
            with urllib.request.urlopen(url, timeout=30) as response:
                length = response.headers.get('Content-Length')
                if length is not None and (int(length) < 0 or int(length) > max_bytes):
                    raise ValueError('download Content-Length exceeds byte limit')
                size = 0
                while True:
                    if time.monotonic() - start > _DOWNLOAD_SECONDS:
                        raise TimeoutError('download exceeded time limit')
                    # read1 avoids waiting to fill a block on a trickling server.
                    block = response.read1(min(64 * 1024, max_bytes - size + 1))
                    if not block:
                        break
                    size += len(block)
                    if size > max_bytes:
                        raise ValueError('download exceeds byte limit')
                    output.write(block)
                if length is not None and size != int(length):
                    raise ValueError('download Content-Length mismatch')
            output.flush()
            os.fsync(output.fileno())
        result = _verify(temporary, algorithm, expected, max_bytes)
        temporary.replace(path)
        return result
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_coil20(archive, image_size):
    """Read exact expected PNG members in numeric order, never extract paths."""
    from PIL import Image

    names = [f'coil-20-proc/obj{obj}__{view}.png'
             for obj in range(1, 21) for view in range(72)]
    expected = set(names)
    images = np.empty((1440, 1, image_size, image_size), dtype=np.uint8)
    native_sizes = set()
    with zipfile.ZipFile(archive) as source:
        seen = set()
        for member in source.infolist():
            if member.orig_filename != member.filename:
                raise ValueError('unexpected COIL-20 archive member spelling')
            # One harmless, explicit directory record is allowed; nothing extracted.
            if (member.filename == 'coil-20-proc/' and member.is_dir()
                    and member.file_size == 0):
                if member.filename in seen:
                    raise ValueError('duplicate COIL-20 archive member')
                seen.add(member.filename)
                continue
            if member.filename not in expected:
                raise ValueError(f'unexpected COIL-20 archive member: {member.filename!r}')
            if member.filename in seen:
                raise ValueError('duplicate COIL-20 view')
            seen.add(member.filename)
            mode = member.external_attr >> 16
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                raise ValueError('COIL-20 member is not a regular file')
            if member.flag_bits & 1 or not 0 < member.file_size <= 128 * 1024:
                raise ValueError('invalid COIL-20 member size or encryption')
        if seen - {'coil-20-proc/'} != expected:
            raise ValueError('missing expected COIL-20 views (need all 20 x 72)')
        for row, name in enumerate(names):
            payload = source.read(name)  # ZipFile checks CRC; bounded above.
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != 'PNG' or image.size != (128, 128) or image.mode != 'L':
                    raise ValueError('COIL-20 PNG must be native 128 x 128 grayscale')
                if getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('COIL-20 PNG must be a single frame')
                native_sizes.add(image.size)
                image.load()
                if image_size != 128:
                    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
                images[row, 0] = np.asarray(image, dtype=np.uint8)
    return images, sorted([list(size) for size in native_sizes])


def load_coil20(cache_dir='data/coil20', download=False, image_size=32):
    """Load all 1,440 processed COIL-20 views, object-major then view-major.

    IDs are 1..20; views 0..71 correspond to 0..355 degrees in five-degree steps.
    The default 32 x 32 is a deliberate fixed LANCZOS downsample, NOT native
    resolution. Request 128 to preserve native pixels (no upsampling allowed).
    Consult the Columbia terms before use; this is not MIT-licensed data.
    """
    if (isinstance(image_size, (bool, np.bool_)) or
            not isinstance(image_size, (int, np.integer)) or not 1 <= image_size <= 128):
        raise ValueError('image_size must be an integer between 1 and 128')
    image_size = int(image_size)
    path = Path(cache_dir) / 'coil-20-proc.zip'
    digest = _ensure_file(path, COIL20_URL, 'sha256', COIL20_SHA256, download)
    images, native_sizes = _read_coil20(path, image_size)
    return {
        'images': images,
        'object_ids': np.repeat(np.arange(1, 21, dtype=np.int64), 72),
        'angles_degrees': np.tile(np.arange(72, dtype=np.float64) * 5.0, 20),
        'metadata': {
            'name': 'COIL-20 processed', 'source_url': COIL20_URL,
            'source_page': COIL20_PAGE, 'files': {path.name: digest},
            'checksum_provenance': 'SHA256 observed from official archive; not publisher-signed',
            'license': 'No explicit license stated on checked official COIL-20 pages; not MIT',
            'license_url': COIL20_CURRENT_PAGE,
            'intended_use': 'research; confirm terms with Columbia; no commercial grant asserted',
            'permission_contact': 'webcave@lists.cs.columbia.edu',
            'citation': 'Nene, Nayar and Murase, COIL-20, CUCS-005-96, February 1996',
            'count': 1440, 'native_dimensions_observed': native_sizes,
            'image_dimensions': [image_size, image_size], 'dtype': 'uint8',
            'preprocessing': ('none' if image_size == 128 else
                              'fixed square downsample from 128 x 128 using Pillow LANCZOS'),
            'ordering': 'object 1..20, each view 0..71; angle = view * 5 degrees',
            'pillow_version': __import__('PIL').__version__,
        },
    }


def _read_idx_gzip(path, *, expected_count, magic):
    """Strict IDX byte/dimension/length validation with bounded decompression."""
    if magic not in (2049, 2051) or expected_count <= 0:
        raise ValueError('unsupported IDX specification')
    header_size = 16 if magic == 2051 else 8
    payload_size = expected_count * (28 * 28 if magic == 2051 else 1)
    total = header_size + payload_size
    # Read one excess byte to reject trailing data, also forcing gzip CRC/EOF checks.
    with gzip.open(path, 'rb') as stream:
        raw = stream.read(total + 1)
    if len(raw) != total:
        raise ValueError('IDX length mismatch (truncated or excess payload)')
    actual_magic, count = struct.unpack_from('>II', raw)
    if actual_magic != magic or count != expected_count:
        raise ValueError('IDX magic or count mismatch')
    if magic == 2051:
        if struct.unpack_from('>II', raw, 8) != (28, 28):
            raise ValueError('IDX image dimensions must be 28 x 28')
        return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, 1, 28, 28).copy()
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
    if np.any(labels > 9):
        raise ValueError('Fashion-MNIST labels must be in 0..9')
    return labels.astype(np.int64)


def load_fashion_mnist(cache_dir='data/fashion_mnist', download=False):
    """Load the complete official 60,000/10,000 split without normalization."""
    root = Path(cache_dir)
    result, files = {}, {}
    for filename, key, count, magic, md5 in _FASHION_FILES:
        path = root / filename
        digest = _ensure_file(path, FASHION_BASE_URL + filename, 'md5', md5, download)
        files[filename] = dict(digest, source_url=FASHION_BASE_URL + filename)
        result[key] = _read_idx_gzip(path, expected_count=count, magic=magic)
    result['metadata'] = {
        'name': 'Fashion-MNIST', 'source_url': FASHION_BASE_URL,
        'source_page': FASHION_README, 'files': files,
        'checksum_provenance': 'compressed-file MD5 published in official README; SHA256 observed',
        'license': 'MIT, Copyright 2017 Zalando SE', 'license_url': FASHION_LICENSE,
        'citation': 'Xiao, Rasul and Vollgraf (2017), Fashion-MNIST, arXiv:1708.07747',
        'train_count': 60000, 'test_count': 10000, 'split': 'official, unshuffled',
        'image_dimensions': [28, 28], 'dtype': 'uint8', 'preprocessing': 'none',
        'label_names': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
    }
    return result
