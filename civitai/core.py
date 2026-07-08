import requests
import hashlib
import json
import sys
import os
import io

from typing import Dict, List, Optional
from pathlib import Path
from PIL import Image

from modules.shared import cmd_opts, opts
from modules.paths import models_path
from modules import sd_models
from modules import hashes as hmod

from .api import get_model
from .logger import log


IS_KAGGLE    = 'KAGGLE_URL_BASE' in os.environ
_preview_ext = '.preview.png'
_resources: List[Dict] = []

MODEL_EXTENSIONS = {'.safetensors', '.pt', '.ckpt', '.pth', '.th', '.vae', '.zip'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _suppress_output(func, *args):
    old_out = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return func(*args)
    finally:
        sys.stdout = old_out


def calc_sha256(path: str, silent: bool = True) -> str:
    """Compute SHA256 of a file using hashes module if available, else hashlib"""
    try:
        name = f"civitai/{Path(path).name}"
        if silent:
            return _suppress_output(hmod.sha256, path, name)
        return hmod.sha256(path, name)
    except ImportError:
        if silent:
            return _suppress_output(_sha256_file, path)
        return _sha256_file(path)


def _get_opt(name: str, fallback: Optional[str] = None, alt: Optional[str] = None) -> Optional[str]:
    val = getattr(cmd_opts, name, None) or (getattr(cmd_opts, alt, None) if alt else None)
    return str(val) if val else fallback


def get_model_dirs() -> Dict[str, Path]:
    """Resolve filesystem paths for each model type"""
    ckpt_dir = _get_opt('ckpt_dir', str(sd_models.model_path), 'ckpt_dirs')
    lora_dir = _get_opt('lora_dir', str(Path(models_path) / 'Lora'), 'lora_dirs')
    emb_dir  = _get_opt('embeddings_dir', str(Path(models_path) / 'embeddings'), 'embeddings_dirs')

    return {
        'Checkpoint':       Path(ckpt_dir),
        'LORA':             Path(lora_dir),
        'LoCon':            Path(lora_dir),
        'DoRA':             Path(lora_dir),
        'TextualInversion': Path(emb_dir),
    }


def has_preview(path: str) -> bool:
    return Path(path).with_suffix(_preview_ext).exists()


def has_info(path: str) -> bool:
    return Path(path).with_suffix('.json').exists()


def _scan_folder(file_type: str, folder: str, exts: set) -> List[Dict]:
    """Scan a single folder for model files matching given extensions"""
    folder_path = Path(folder).resolve()
    items = []

    for file_path in folder_path.rglob('*'):
        if file_path.is_dir() or file_path.suffix not in exts:
            continue

        path_str = str(file_path)
        items.append({
            'type':        file_type,
            'name':        file_path.stem,
            'hash':        calc_sha256(path_str, silent=True),
            'path':        path_str,
            'has_preview': has_preview(path_str),
            'has_info':    has_info(path_str),
        })
    return items


def scan_resources(types: List[str]) -> List[Dict]:
    """Scan all configured model directories for each given type"""
    global _resources

    if not types:
        return _resources

    old_no_hashing = _get_opt('no_hashing')
    if old_no_hashing is not None:
        cmd_opts.no_hashing = False

    try:
        dirs       = get_model_dirs()
        result     = []
        seen_paths = set()
        for type in types:
            folder = dirs.get(type)

            if folder and folder.exists():
                for item in _scan_folder(type, str(folder), MODEL_EXTENSIONS):
                    if item['path'] not in seen_paths:
                        seen_paths.add(item['path'])
                        result.append(item)

        _resources = result
        return result
    finally:
        if old_no_hashing is not None:
            cmd_opts.no_hashing = old_no_hashing


def save_json(version: Dict, resource_path: str, sha256: str, skip_existing: bool = True) -> bool:
    """Save model info JSON sidecar next to the model file"""
    dest = Path(resource_path).with_suffix('.json')
    if skip_existing and dest.exists():
        return False

    model_id    = version.get('modelId', 0)
    version_id  = version.get('id', 0),
    model       = version.get('model', {})
    description = model.get('description', '')

    if not description and model_id and getattr(opts, 'ce_save_description', False):
        try:
            model_data  = get_model(model_id)
            description = model_data.get('description', '')
            model       = model_data
        except Exception:
            pass

    data = {
        'modelName':    model.get('name', ''),
        'versionName':  version.get('name', ''),
        'modelId':      model_id,
        'versionId':    version_id,
        'contentType':  model.get('type', ''),
        'baseModel':    version.get('baseModel', ''),
        'trainedTags':  version.get('trainedWords', []),
        'description':  description,
        'modelPageURL': f"https://civitai.red/models/{model_id}?modelVersionId={version_id}",
        'sha256':       sha256.upper(),
    }

    dest.write_text(json.dumps(data, indent=4), encoding='utf-8')
    return True


def save_preview(url: str, resource_path: str, skip_existing: bool = True, resize: int = 512) -> bool:
    """Download and save a preview image, with optional resize and Kaggle encryption"""
    dest = Path(resource_path).with_suffix(_preview_ext)
    if skip_existing and dest.exists():
        return True

    try:
        resp = requests.get(url, stream=True, headers={'User-Agent': 'CivitaiLink:Automatic1111'})
        resp.raise_for_status()

        data = resp.content

        if resize > 0:
            img   = Image.open(io.BytesIO(data))
            w, h  = img.size
            scale = resize / max(w, h)
            img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf   = io.BytesIO()
            img.save(buf, format='PNG')
            data = buf.getvalue()

        if IS_KAGGLE:
            import sd_image_encryption
            img  = Image.open(io.BytesIO(data))
            info = img.info or {}
            if not all(k in info for k in ('Encrypt', 'EncryptPwdSha')):
                sd_image_encryption.EncryptedImage.from_image(img).save(dest)
            return True

        dest.write_bytes(data)
        return True
    except Exception as e:
        log.error(f"Preview save failed [{dest.name}]: {e}")
        if dest.exists():
            dest.unlink()
        return False
