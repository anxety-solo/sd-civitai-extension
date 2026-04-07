import requests
import json
import time
import sys
import os
import io
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from PIL import Image

from modules import shared, sd_models, sd_vae, hashes, ui_extra_networks, cache
from modules.paths import models_path

from .logger import log


# ~~ Config ~~

BASE_URL = 'https://civitai.com/api/v1'
USER_AGENT = 'CivitaiLink:Automatic1111'

IS_KAGGLE = 'KAGGLE_URL_BASE' in os.environ

_api_cache = cache.cache('civil_ai_api_sha256')
_resources: List[Dict[str, Any]] = []


# ~~ Helpers ~~

def _verbose() -> bool:
    """Get verbose logging setting"""
    return getattr(shared.opts, 'civitai_verbose', False)

def _suppress_output(func, *args, **kwargs):
    """Run func with both stdout and stderr suppressed"""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return func(*args, **kwargs)
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        os.close(devnull_fd)
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# ~~ API ~~

def req(endpoint: str, method: str = 'GET', data: Any = None, params: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """Make HTTP request to Civitai API"""
    if headers is None:
        headers = {}
    headers['User-Agent'] = USER_AGENT

    api_key = shared.opts.data.get('civitai_api_key')
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"

    if data is not None:
        headers['Content-Type'] = 'application/json'
        data = json.dumps(data)

    url = BASE_URL + ('/' + endpoint.lstrip('/'))
    response = requests.request(method, url, data=data, params=params or {}, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} {response.text}")
    return response.json()

def get_all_by_hash(file_hashes: List[str]) -> List[Dict[str, Any]]:
    """Get model information by hash list"""
    return req('/model-versions/by-hash', method='POST', data=file_hashes)

def get_all_by_hash_with_cache(file_hashes: List[str]) -> List[Dict[str, Any]]:
    """Get model information by hash with caching"""
    missing = [h for h in file_hashes if h not in _api_cache]
    new_results = []

    if missing:
        log.info(f"Fetching info for {len(missing)} missing hashes", _verbose())
        try:
            for i in range(0, len(missing), 100):
                new_results.extend(get_all_by_hash(missing[i:i + 100]))
        except Exception as e:
            log.error(f"Error fetching model info: {e}")
            raise

    new_results.sort(
        key=lambda x: datetime.fromisoformat(x['createdAt'].rstrip('Z')),
        reverse=True
    )

    found = set()
    for meta in new_results:
        for f in meta['files']:
            h = f['hashes']['SHA256'].lower()
            _api_cache[h] = meta
            found.add(h)

    for h in set(missing) - found:
        _api_cache[h] = None

    return [_api_cache[h] for h in file_hashes if _api_cache.get(h)]


# ~~ Directories ~~

def get_lora_dir() -> str:
    return (
        getattr(shared.cmd_opts, 'lora_dir', None)
        or getattr(shared.cmd_opts, 'lora_dirs', None)
    )

def get_locon_dir() -> str:
    try:
        return shared.cmd_opts.lyco_dir or get_lora_dir()
    except AttributeError:
        return get_lora_dir()

def get_model_dir() -> str:
    return (
        getattr(shared.cmd_opts, 'ckpt_dir', None)
        or getattr(shared.cmd_opts, 'ckpt_dirs', None)
        or sd_models.model_path
    )


# ~~ Resources ~~

def _has_preview(filename: str) -> bool:
    """Check if file has a preview image"""
    preview_exts = ui_extra_networks.allowed_preview_extensions()
    preview_exts = [*preview_exts, *[f"preview.{x}" for x in preview_exts]]
    return any(Path(filename).with_suffix(f".{ext}").exists() for ext in preview_exts)

def _has_info(filename: str) -> bool:
    """Check if file has an info JSON sidecar"""
    return Path(filename).with_suffix('.json').exists()

def _auto_name(file_type: str, filename: str, folder: str) -> str:
    """Get automatic name for file relative to folder"""
    path = Path(filename).resolve()
    try:
        rel = path.relative_to(Path(folder).resolve())
    except ValueError:
        rel = Path(path.name)
    return str(rel) if file_type == 'Checkpoint' else str(rel.with_suffix(''))

def _calc_hash(path: str, auto_type: str, auto_name: str) -> str:
    """Calculate SHA256 file hash, optionally suppressing all console output"""
    fn = lambda: hashes.sha256(path, f"{auto_type}/{auto_name}")
    if getattr(shared.opts, 'civitai_suppress_hash_output', True):
        return _suppress_output(fn)
    return fn()

def get_resources_in_folder(file_type: str, folder: str, exts: List[str] = None, exts_exclude: List[str] = None) -> List[Dict[str, Any]]:
    """Get resources from folder with specified extensions"""
    exts = exts or []
    exts_exclude = exts_exclude or []
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)

    auto_type = file_type.lower()
    candidates = [
        f for ext in exts for f in folder.rglob(f"*.{ext}")
        if not f.is_dir() and not any(str(f).endswith(e) for e in exts_exclude)
    ]

    result = []
    old_no_hashing = shared.cmd_opts.no_hashing
    shared.cmd_opts.no_hashing = False

    try:
        for f in sorted(candidates):
            auto_name = _auto_name(file_type, str(f), str(folder))
            result.append({
                'type': file_type,
                'name': f.stem,
                'hash': _calc_hash(str(f), auto_type, auto_name),
                'path': str(f),
                'hasPreview': _has_preview(str(f)),
                'hasInfo': _has_info(str(f)),
            })
    finally:
        shared.cmd_opts.no_hashing = old_no_hashing

    return result

def load_resource_list(types: List[str] = None) -> List[Dict[str, Any]]:
    """Load resource list from all configured folders"""
    global _resources

    if types is None:
        types = ['LORA', 'LoCon', 'TextualInversion', 'Checkpoint', 'VAE', 'Controlnet', 'Upscaler']

    res = [r for r in _resources if r['type'] not in types]

    lora_dir = Path(get_lora_dir())
    locon_dir = Path(get_locon_dir())
    model_dir = Path(get_model_dir())

    type_configs = [
        ('LORA', lora_dir, ['pt', 'safetensors', 'ckpt']),
        ('LoCon', locon_dir, ['pt', 'safetensors', 'ckpt']) if locon_dir != lora_dir else None,
        ('TextualInversion', Path(getattr(shared.cmd_opts, 'embeddings_dir', '')), ['pt', 'bin', 'safetensors']),
        ('Checkpoint', model_dir, ['safetensors', 'ckpt'], ['vae.safetensors', 'vae.ckpt']),
        ('Controlnet', Path(models_path) / 'ControlNet', ['safetensors', 'ckpt'], ['vae.safetensors', 'vae.ckpt']),
        ('Upscaler', Path(models_path) / 'ESRGAN', ['safetensors', 'ckpt', 'pt']),
    ]

    for cfg in type_configs:
        if cfg is None:
            continue
        t, folder, exts, *rest = cfg
        if t in types:
            log.info(f"Loading {t} resources from {folder}", _verbose())
            res += get_resources_in_folder(t, str(folder), exts, rest[0] if rest else [])

    # VAE handled separately — may span two distinct folders
    if 'VAE' in types:
        log.info('Loading VAE resources', _verbose())
        vae_exts = ['vae.pt', 'vae.safetensors', 'vae.ckpt']
        vae_path = Path(getattr(sd_vae, 'vae_path', ''))
        for folder in {model_dir, vae_path}:
            res += get_resources_in_folder('VAE', str(folder), vae_exts)

    _resources = res
    return res

def get_model_by_hash(file_hash: str) -> Optional[Any]:
    """Get checkpoint model by hash from the checkpoints list"""
    found = [
        info for info in sd_models.checkpoints_list.values()
        if file_hash in (info.sha256, info.shorthash, info.hash)
    ]
    return found[0] if found else None

def get_resource_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """Get resource by hash, excluding resources currently being downloaded"""
    found = [
        r for r in load_resource_list([])
        if file_hash.lower() == r['hash'] and not r.get('downloading')
    ]
    return found[0] if found else None


# ~~ Preview download ~~

def _resize_image_bytes(image_bytes: bytes, target_size: int = 512) -> io.BytesIO:
    """Resize image bytes to target_size on the longer side, keeping aspect ratio"""
    image = Image.open(io.BytesIO(image_bytes))
    w, h = image.size
    scale = target_size / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    output = io.BytesIO()
    image.resize(new_size, Image.LANCZOS).save(output, format='PNG')
    output.seek(0)
    return output

def download_preview(url: str, dest_path: str, on_progress=None) -> bool:
    """Download and resize preview image"""
    dest = Path(dest_path).expanduser()
    if dest.exists():
        return True

    log.info(f"Downloading preview: {url}", _verbose())
    response = requests.get(url, stream=True, headers={'User-Agent': USER_AGENT})
    total = int(response.headers.get('content-length', 0))
    start_time = time.time()

    try:
        data = bytearray()
        for chunk in response.iter_content(chunk_size=8192):
            data.extend(chunk)
            if on_progress and on_progress(len(data), total, start_time):
                raise Exception('Download cancelled')

        resized = _resize_image_bytes(bytes(data))

        if IS_KAGGLE:
            import sd_image_encryption
            img = Image.open(resized)
            imginfo = img.info or {}
            if not all(k in imginfo for k in ('Encrypt', 'EncryptPwdSha')):
                sd_image_encryption.EncryptedImage.from_image(img).save(dest)
        else:
            dest.write_bytes(resized.read())
        return True

    except Exception as e:
        log.error(f"Preview download failed [{dest}]: {e}")
        if dest.exists():
            dest.unlink()
        return False

def update_resource_preview(file_hash: str, preview_url: str) -> bool:
    """Update preview for resource with given hash"""
    success = False
    for res in [r for r in load_resource_list([]) if r['hash'] == file_hash.lower()]:
        preview_path = Path(res['path']).with_suffix('.preview.png')
        if download_preview(preview_url, str(preview_path)):
            success = True
    return success