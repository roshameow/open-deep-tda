"""Offline regression tests; synthetic inputs are not substitutes for real data.

Verified workflow: full-download validation is a separate explicit loader call;
these tests must never contact dataset hosts. COIL object/view ordering must be
numeric (lexical ZIP order puts obj10 before obj2); IDX byte order is big-endian.
Separate real-file verification passed: all 1,440 COIL PNGs are native 128 x 128;
Fashion-MNIST has 60,000/10,000 rows with 6,000/1,000 examples per class. The
four published MD5s match, and COIL's full ZIP passed every member CRC check.
"""
import gzip
import hashlib
import importlib
import io
import json
import struct
import warnings
import zipfile

import numpy as np
import pytest
from PIL import Image

from open_deep_tda import image_datasets as datasets


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('tests must not access the network')
    monkeypatch.setattr(datasets.urllib.request, 'urlopen', forbidden)


def test_import_and_missing_cache_are_offline(tmp_path):
    importlib.reload(datasets)
    for loader in (datasets.load_coil20, datasets.load_fashion_mnist):
        with pytest.raises(FileNotFoundError, match='download=True'):
            loader(tmp_path / 'absent')
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('size', [0, -1, 129, 32.0, True, '32', None])
def test_bad_image_size(size, tmp_path):
    with pytest.raises(ValueError, match='image_size'):
        datasets.load_coil20(tmp_path, image_size=size)


def png(value=0, size=(128, 128), mode='L', fmt='PNG'):
    stream = io.BytesIO()
    Image.new(mode, size, value).save(stream, format=fmt)
    return stream.getvalue()


def coil_zip(path, *, omit=None, extra=None, replacement=None):
    # Deliberately reverse canonical ordering; all expected views still present.
    payloads = {obj: png(obj) for obj in range(1, 21)}
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('coil-20-proc/', b'')
        for obj in range(20, 0, -1):
            for view in range(71, -1, -1):
                name = f'coil-20-proc/obj{obj}__{view}.png'
                if name != omit:
                    data = replacement if replacement and obj == 1 and view == 0 else payloads[obj]
                    archive.writestr(name, data)
        if extra:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                archive.writestr(extra, png())


def test_complete_coil_api_order_dtype_metadata(tmp_path, monkeypatch):
    path = tmp_path / 'coil-20-proc.zip'
    coil_zip(path)
    monkeypatch.setattr(datasets, 'COIL20_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    for size in (32, 128):
        result = datasets.load_coil20(tmp_path, image_size=size)
        assert result['images'].shape == (1440, 1, size, size)
        assert result['images'].dtype == np.uint8
        assert result['object_ids'].dtype == np.int64
        assert result['angles_degrees'].dtype == np.float64
        np.testing.assert_array_equal(result['object_ids'], np.repeat(np.arange(1, 21), 72))
        np.testing.assert_array_equal(result['angles_degrees'], np.tile(np.arange(72) * 5., 20))
        np.testing.assert_array_equal(result['images'][:, 0, 0, 0], result['object_ids'])
        assert result['metadata']['native_dimensions_observed'] == [[128, 128]]
        assert result['metadata']['image_dimensions'] == [size, size]
        assert ('LANCZOS' in result['metadata']['preprocessing']) == (size != 128)
        json.dumps(result['metadata'])
    assert [p.name for p in tmp_path.iterdir()] == ['coil-20-proc.zip']


@pytest.mark.parametrize('extra', ['../escape.png', '/absolute.png',
                                  'coil-20-proc/../escape.png', 'coil-20-proc/obj1__72.png',
                                  'coil-20-proc/obj21__0.png', 'coil-20-proc/obj01__0.png',
                                  'coil-20-proc/obj1__0.png', 'README.txt'])
def test_coil_rejects_unexpected_or_duplicate_members(tmp_path, extra):
    path = tmp_path / 'bad.zip'
    coil_zip(path, extra=extra)
    with pytest.raises(ValueError, match='unexpected|duplicate'):
        datasets._read_coil20(path, 32)
    assert len(list(tmp_path.iterdir())) == 1


def test_coil_missing_view(tmp_path):
    path = tmp_path / 'bad.zip'
    coil_zip(path, omit='coil-20-proc/obj20__71.png')
    with pytest.raises(ValueError, match='missing expected'):
        datasets._read_coil20(path, 32)


@pytest.mark.parametrize('payload', [png(size=(32, 32)), png(mode='RGB'), png(fmt='BMP')])
def test_coil_rejects_wrong_native_png(tmp_path, payload):
    path = tmp_path / 'bad.zip'
    coil_zip(path, replacement=payload)
    with pytest.raises(ValueError, match='native 128'):
        datasets._read_coil20(path, 32)


def idx_bytes(magic=2051, count=2, rows=28, columns=28):
    header = struct.pack('>II', magic, count)
    if magic == 2051:
        size = count * rows * columns
        payload = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
        return header + struct.pack('>II', rows, columns) + payload
    return header + bytes(range(count))


@pytest.mark.parametrize('magic', [2049, 2051])
def test_idx_valid(tmp_path, magic):
    path = tmp_path / 'data.gz'
    path.write_bytes(gzip.compress(idx_bytes(magic)))
    result = datasets._read_idx_gzip(path, expected_count=2, magic=magic)
    assert result.shape == ((2, 1, 28, 28) if magic == 2051 else (2,))
    assert result.dtype == (np.uint8 if magic == 2051 else np.int64)


@pytest.mark.parametrize('mutation,match', [
    (lambda b: b[:3], 'length'),
    (lambda b: b[:-1], 'length'),
    (lambda b: b + b'x', 'length'),
    (lambda b: struct.pack('>I', 999) + b[4:], 'magic'),
    (lambda b: b[:4] + struct.pack('>I', 3) + b[8:], 'count'),
    (lambda b: b[:8] + struct.pack('>II', 14, 56) + b[16:], 'dimensions'),
])
def test_idx_rejects_malformed(tmp_path, mutation, match):
    path = tmp_path / 'bad.gz'
    path.write_bytes(gzip.compress(mutation(idx_bytes())))
    with pytest.raises(ValueError, match=match):
        datasets._read_idx_gzip(path, expected_count=2, magic=2051)


def test_idx_bad_label_and_corrupt_gzip(tmp_path):
    path = tmp_path / 'bad.gz'
    path.write_bytes(gzip.compress(struct.pack('>II', 2049, 2) + bytes([0, 10])))
    with pytest.raises(ValueError, match='labels'):
        datasets._read_idx_gzip(path, expected_count=2, magic=2049)
    raw = bytearray(gzip.compress(idx_bytes()))
    raw[-8] ^= 1
    path.write_bytes(raw)
    with pytest.raises(gzip.BadGzipFile):
        datasets._read_idx_gzip(path, expected_count=2, magic=2051)
    path.write_bytes(gzip.compress(idx_bytes())[:-4])
    with pytest.raises(EOFError):
        datasets._read_idx_gzip(path, expected_count=2, magic=2051)


def test_fashion_api_with_small_idx_fixtures(tmp_path, monkeypatch):
    specs = []
    for filename, key, _, magic, _ in datasets._FASHION_FILES:
        raw = gzip.compress(idx_bytes(magic))
        (tmp_path / filename).write_bytes(raw)
        specs.append((filename, key, 2, magic, hashlib.md5(raw).hexdigest()))
    monkeypatch.setattr(datasets, '_FASHION_FILES', specs)
    result = datasets.load_fashion_mnist(tmp_path)
    for split in ('train', 'test'):
        assert result[split + '_images'].shape == (2, 1, 28, 28)
        assert result[split + '_images'].dtype == np.uint8
        assert result[split + '_labels'].dtype == np.int64
    assert len(result['metadata']['files']) == 4
    json.dumps(result['metadata'])
    assert len(list(tmp_path.iterdir())) == 4


@pytest.mark.parametrize('loader,filename', [
    (datasets.load_coil20, 'coil-20-proc.zip'),
    (datasets.load_fashion_mnist, 'train-images-idx3-ubyte.gz'),
])
def test_cached_files_are_verified_even_with_download_true(tmp_path, loader, filename):
    (tmp_path / filename).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        loader(tmp_path, download=True)
    assert (tmp_path / filename).read_bytes() == b'corrupt'


class Response(io.BytesIO):
    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = {} if length is None else {'Content-Length': str(length)}


def test_atomic_download_and_cache_revalidation(tmp_path, monkeypatch):
    path = tmp_path / 'file'
    payload = b'verified data'
    expected = hashlib.sha256(payload).hexdigest()

    def get(url, timeout):
        assert url == 'https://example.test/data'
        assert timeout == 30
        assert not path.exists()
        return Response(payload, len(payload))

    monkeypatch.setattr(datasets.urllib.request, 'urlopen', get)
    digest = datasets._ensure_file(path, 'https://example.test/data', 'sha256', expected, True)
    assert digest['sha256'] == expected
    assert path.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [path]
    assert datasets._ensure_file(path, '', 'sha256', expected, False) == digest
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        datasets._ensure_file(path, '', 'sha256', expected, True)


@pytest.mark.parametrize('data,length,max_bytes,match', [
    (b'bad', 3, 10, 'checksum'),
    (b'x' * 11, None, 10, 'byte limit'),
    (b'x', 11, 10, 'byte limit'),
    (b'x', 2, 10, 'Content-Length mismatch'),
])
def test_download_failure_cleans_temporary(tmp_path, monkeypatch, data, length, max_bytes, match):
    monkeypatch.setattr(datasets.urllib.request, 'urlopen', lambda *a, **kw: Response(data, length))
    with pytest.raises(ValueError, match=match):
        datasets._ensure_file(tmp_path / 'file', 'https://example.test', 'sha256',
                              'wrong', True, max_bytes=max_bytes)
    assert list(tmp_path.iterdir()) == []


def test_download_timeout_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets.urllib.request, 'urlopen', lambda *a, **kw: Response(b'abc'))
    ticks = iter([0, datasets._DOWNLOAD_SECONDS + 1])
    monkeypatch.setattr(datasets.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(TimeoutError):
        datasets._ensure_file(tmp_path / 'file', '', 'sha256', '', True)
    assert list(tmp_path.iterdir()) == []


def test_coil_rejects_symlink_and_oversize_member(tmp_path):
    path = tmp_path / 'bad.zip'
    link = zipfile.ZipInfo('coil-20-proc/obj1__0.png')
    link.create_system = 3
    link.external_attr = 0o120777 << 16
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(link, b'/outside.png')
    with pytest.raises(ValueError, match='regular file'):
        datasets._read_coil20(path, 32)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('coil-20-proc/obj1__0.png', b'x' * (128 * 1024 + 1))
    with pytest.raises(ValueError, match='member size'):
        datasets._read_coil20(path, 32)


def test_cached_file_size_is_bounded(tmp_path):
    path = tmp_path / 'file'
    path.write_bytes(b'x' * 11)
    with pytest.raises(ValueError, match='byte limit'):
        datasets._ensure_file(path, '', 'sha256', '', False, max_bytes=10)


def test_network_exception_cleans_temporary(tmp_path, monkeypatch):
    def disconnected(*args, **kwargs):
        raise OSError('disconnected')
    monkeypatch.setattr(datasets.urllib.request, 'urlopen', disconnected)
    with pytest.raises(OSError, match='disconnected'):
        datasets._ensure_file(tmp_path / 'file', '', 'sha256', '', True)
    assert list(tmp_path.iterdir()) == []


def test_idx_rejects_extra_gzip_member(tmp_path):
    path = tmp_path / 'bad.gz'
    path.write_bytes(gzip.compress(idx_bytes()) + gzip.compress(b'x'))
    with pytest.raises(ValueError, match='length'):
        datasets._read_idx_gzip(path, expected_count=2, magic=2051)


def test_known_published_fashion_pins():
    assert datasets.COIL20_SHA256 == (
        '517c5594820eb40066ba0ff6842e7f09392bf7fde849bf9cb9c28445b0f29e88'
    )
    assert [spec[2] for spec in datasets._FASHION_FILES] == [60000, 60000, 10000, 10000]
    assert [spec[4] for spec in datasets._FASHION_FILES] == [
        '8d4fb7e6c68d591d4c3dfef9ec88bf0d', '25c81989df183df01b3e8a0aad5dffbe',
        'bef4ecab320f06d8554ea6380940ec79', 'bb300cfdad3c16e7a12a480ee83cd310',
    ]
