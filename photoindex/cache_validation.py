from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np

RUN_ID = 'chinese-clip-base-cpu-v2'
MODEL_ID = 'OFA-Sys/chinese-clip-vit-base-patch16'
MODEL_REVISION = '36e679e65c2a2fead755ae21162091293ad37834'
WEIGHTS_SHA256 = '7b7b583c210c867410bc6bdb8a55fe14eec62999e0a9ea31ff222dc501f9cfbe'
MANIFEST_SHA256 = '90ba5129aa1fd4b4ce462dbfd9043088bf235738ab1662edef37523803cdf496'
VECTORS_SHA256 = 'd1090bd421f1353b1bd5b4df4edcdfbee258b8d683fb3c2f692bd70389233806'

class CacheValidationError(RuntimeError):
    pass

@dataclass(frozen=True)
class VerifiedCache:
    run_id: str
    model_dir: Path
    asset_ids: tuple[int, ...]
    image_vectors: np.ndarray
    asset_paths: dict[int, tuple[str, str]]
    thumbnails: dict[int, tuple[Path, str]]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CacheValidationError(message)


def read_json(path: Path):
    require(path.is_file(), f'Missing file: {path}')
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CacheValidationError(f'Cannot read JSON: {path}') from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
    except OSError as exc:
        raise CacheValidationError(f'Cannot hash: {path}') from exc
    return digest.hexdigest()


def verified_hash(path: Path, expected: str) -> None:
    require(isinstance(expected, str) and len(expected) == 64,
            f'Invalid expected SHA-256 for {path.name}')
    require(sha256_file(path) == expected, f'SHA-256 mismatch: {path}')


def project_path(root: Path, value: str, boundary: Path) -> Path:
    require(isinstance(value, str) and bool(value), 'Invalid relative path')
    relative = Path(value)
    require(not relative.is_absolute(), f'Absolute path disallowed: {value}')
    resolved = (root / relative).resolve(strict=False)
    require(resolved.is_relative_to(boundary.resolve()),
            f'Path escapes expected directory: {value}')
    return resolved


def load_verified_v2_cache(project_root: Path) -> VerifiedCache:
    root = Path(project_root).resolve(strict=True)
    run = root / 'local-data/retrieval/runs' / RUN_ID
    retrieval_root = root / 'local-data/retrieval'
    thumbnail_root = retrieval_root / 'thumbnails'
    model_dir = root / 'local-data/models/chinese-clip-base'
    manifest_path = retrieval_root / 'manifest.json'
    images_path = retrieval_root / 'images.json'
    ids_path = run / 'asset_ids.json'
    metadata_path = run / 'run_metadata.json'
    vectors_path = run / 'image_vectors.npy'
    lock_path = root / 'local-data/models/chinese-clip-base-lock.json'

    verified_hash(manifest_path, MANIFEST_SHA256)
    verified_hash(vectors_path, VECTORS_SHA256)
    manifest = read_json(manifest_path)
    images = read_json(images_path)
    ids = read_json(ids_path)
    meta = read_json(metadata_path)
    lock = read_json(lock_path)

    require(isinstance(manifest, dict) and isinstance(images, dict)
            and isinstance(meta, dict) and isinstance(lock, dict),
            'Metadata must contain JSON objects')
    require(meta.get('run_id') == RUN_ID, 'Wrong run_id')
    require(meta.get('model_id') == MODEL_ID, 'Wrong model_id')
    require(meta.get('model_revision') == MODEL_REVISION, 'Wrong revision')
    require(lock.get('model_id') == MODEL_ID and lock.get('revision') == MODEL_REVISION,
            'Model lock identity mismatch')
    require(meta.get('dataset_id') == manifest.get('dataset_id'), 'Dataset ID mismatch')
    require(meta.get('manifest_sha256') == MANIFEST_SHA256, 'Manifest reference mismatch')
    require(meta.get('images_json_sha256') == sha256_file(images_path),
            'images.json reference mismatch')
    require(meta.get('image_vectors_sha256') == VECTORS_SHA256,
            'Vector reference mismatch')
    require(meta.get('device') == 'cpu' and meta.get('dtype') in ('float32', 'torch.float32'),
            'Unexpected device or dtype')
    require(meta.get('vector_shape') == [50, 512], 'Vector metadata shape mismatch')
    require(meta.get('model_weights_sha256') == WEIGHTS_SHA256 and
            lock.get('weight_sha256') == WEIGHTS_SHA256, 'Weight reference mismatch')

    model_hashes = meta.get('model_files_sha256')
    require(isinstance(model_hashes, dict), 'Missing model files hashes')
    for name in ('config.json', 'preprocessor_config.json', 'vocab.txt', 'pytorch_model.bin'):
        require(name in model_hashes, f'Missing model digest: {name}')
        verified_hash(model_dir / name, model_hashes[name])
    require(meta.get('processor_config_sha256') == sha256_file(model_dir / 'preprocessor_config.json'),
            'Processor config mismatch')
    require(sha256_file(model_dir / 'pytorch_model.bin') == WEIGHTS_SHA256,
            'Weight bytes mismatch')

    require(isinstance(ids, list) and len(ids) == 50 and
            all(type(i) is int and i > 0 for i in ids) and len(set(ids)) == 50,
            'Invalid asset IDs')
    require(isinstance(manifest.get("asset_ids"), list)
        and manifest["asset_ids"] == ids,
        "Asset row order differs from frozen manifest",
    )
    require(isinstance(images.get('images'), list) and len(images['images']) == 50,
            'Invalid image mapping')
    require(isinstance(meta.get('inputs'), list) and len(meta['inputs']) == 50,
            'Invalid input manifest')

    try:
        vectors = np.load(vectors_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CacheValidationError('Cannot load image vectors') from exc
    require(vectors.shape == (50, 512) and vectors.dtype == np.float32,
            'Invalid vector shape or dtype')
    require(bool(np.isfinite(vectors).all()), 'Nonfinite vectors')
    require(bool(np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5, rtol=0)),
            'Vectors not unit-normalized')
    vectors.setflags(write=False)

    by_id = {}
    for entry in images['images']:
        require(isinstance(entry, dict) and type(entry.get('asset_id')) is int
                and isinstance(entry.get('png_path'), str), 'Invalid images entry')
        asset_id = entry['asset_id']
        require(asset_id not in by_id and asset_id in ids, 'Duplicate/unknown image ID')
        by_id[asset_id] = entry['png_path']

    thumbnails = {}
    for entry in meta['inputs']:
        require(isinstance(entry, dict) and type(entry.get('asset_id')) is int
                and isinstance(entry.get('png_path'), str), 'Invalid input entry')
        asset_id = entry['asset_id']
        require(asset_id in by_id and asset_id not in thumbnails, 'Duplicate/unknown input ID')
        require(entry['png_path'] == by_id[asset_id], 'Input path does not match images.json')
        require(type(entry.get('width')) is int and entry['width'] > 0 and
                type(entry.get('height')) is int and entry['height'] > 0,
                'Invalid image dimensions')
        preview = project_path(root, entry['png_path'], thumbnail_root)
        require(preview.suffix.lower() == '.png', 'Thumbnail is not PNG')
        verified_hash(preview, entry.get('sha256'))
        thumbnails[asset_id] = (preview, entry['sha256'])
    require(set(thumbnails) == set(ids), 'Missing thumbnail mapping')
    require(
        len({path for path, _ in thumbnails.values()}) == len(ids),
        "Duplicate thumbnail paths",
    )

    db_path = root / 'local-data/photoindex.db'
    require(db_path.is_file(), 'Missing SQLite database')
    uri = 'file:' + quote(str(db_path), safe='/') + '?mode=ro'
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            conn.execute('PRAGMA query_only=ON')
            rows = conn.execute('''
                SELECT a.id, a.path, s.root_path, a.source_id
                FROM assets AS a JOIN sources AS s ON a.source_id = s.id
            ''').fetchall()
    except sqlite3.Error as exc:
        raise CacheValidationError('Cannot read Source/Asset mapping') from exc

    db_rows = {asset_id: (rel, source_root, source_id)
               for asset_id, rel, source_root, source_id in rows}
    require(len(db_rows) == len(rows), 'Duplicate Asset IDs in SQLite')
    require(set(ids).issubset(db_rows), 'Frozen asset missing from SQLite')
    require(manifest.get('source_id') is not None, 'Manifest has no source ID')

    asset_paths = {}
    for asset_id in ids:
        relative, source_string, source_id = db_rows[asset_id]
        require(source_id == manifest['source_id'], 'Wrong Source ID')
        require(isinstance(relative, str) and relative != '' and
                isinstance(source_string, str) and source_string != '',
                'Invalid source or relative path')
        source = Path(source_string)
        require(source.is_absolute(), 'Source root is not absolute')
        source = source.resolve(strict=False)
        require(not Path(relative).is_absolute(), 'Original path must be relative')
        original = (source / relative).resolve(strict=False)
        require(original.is_relative_to(source), 'Original path escapes Source root')
        asset_paths[asset_id] = (relative, str(original))

    return VerifiedCache(RUN_ID, model_dir, tuple(ids), vectors, asset_paths, thumbnails)
