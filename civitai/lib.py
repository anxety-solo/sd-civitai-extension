from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from PIL import Image
import requests
import json
import time
import sys
import os
import io

from modules import shared, sd_models, sd_vae, hashes, ui_extra_networks, cache
from modules.paths import models_path

from .logger import log_message


base_url = 'https://civitai.com/api/v1'
user_agent = 'CivitaiLink:Automatic1111'
civil_ai_api_cache = cache.cache('civil_ai_api_sha256')

IS_KAGGLE = 'KAGGLE_URL_BASE' in os.environ

resources = []


def get_verbose() -> bool:
    """Get verbose logging setting."""
    return getattr(shared.opts, 'civitai_verbose', False)

def req(endpoint: str, method: str = 'GET', data: Any = None, params: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """Make HTTP request to Civitai API."""
    if headers is None:
        headers = {}
    headers['User-Agent'] = user_agent

    api_key = shared.opts.data.get('civitai_api_key', None)
    if api_key is not None:
        headers['Authorization'] = f"Bearer {api_key}"

    if data is not None:
        headers['Content-Type'] = 'application/json'
        data = json.dumps(data)

    if not endpoint.startswith('/'):
        endpoint = '/' + endpoint

    if params is None:
        params = {}

    response = requests.request(method, base_url + endpoint, data=data, params=params, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} {response.text}")

    return response.json()

def get_all_by_hash(hashes: List[str]) -> List[Dict[str, Any]]:
    """Get model information by hash list."""
    response = req(f"/model-versions/by-hash", method='POST', data=hashes)
    return response

def get_lora_dir() -> str:
    """Get LoRA directory path."""
    return shared.cmd_opts.lora_dir

def get_locon_dir() -> str:
    """Get LoCon directory path."""
    try:
        return shared.cmd_opts.lyco_dir or get_lora_dir()
    except AttributeError:
        return get_lora_dir()

def get_model_dir() -> str:
    """Get model directory path."""
    return shared.cmd_opts.ckpt_dir or sd_models.model_path

def get_automatic_type(file_type: str) -> str:
    """Convert file type to automatic type."""
    if file_type == 'Hypernetwork':
        return 'hypernet'
    return file_type.lower()

def get_automatic_name(file_type: str, filename: str, folder: str) -> str:
    """Get automatic name for file."""
    path = Path(filename).resolve()
    folder_path = Path(folder).resolve()

    try:
        fullname = path.relative_to(folder_path)
    except ValueError:
        fullname = path.name

    if file_type == 'Checkpoint':
        return str(fullname)
    return str(fullname.with_suffix(''))

def has_preview(filename: str) -> bool:
    """Check if file has preview image."""
    preview_exts = ui_extra_networks.allowed_preview_extensions()
    preview_exts = [*preview_exts, *['preview.' + x for x in preview_exts]]
    for ext in preview_exts:
        if Path(filename).with_suffix(f".{ext}").exists():
            return True
    return False

def has_info(filename: str) -> bool:
    """Check if file has info JSON."""
    return Path(filename).with_suffix('.json').exists()

def calculate_file_hash(file_path: str, automatic_type: str, automatic_name: str) -> str:
    """Calculate file hash with optional output suppression."""
    suppress_hash_output = getattr(shared.opts, 'civitai_suppress_hash_output', True)

    if suppress_hash_output:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            file_hash = hashes.sha256(str(file_path), f"{automatic_type}/{automatic_name}")
        finally:
            sys.stdout = old_stdout
    else:
        file_hash = hashes.sha256(str(file_path), f"{automatic_type}/{automatic_name}")

    return file_hash

def get_resources_in_folder(file_type: str, folder: str, exts: List[str] = None, exts_exclude: List[str] = None) -> List[Dict[str, Any]]:
    """Get resources from folder with specified extensions."""
    exts = exts or []
    exts_exclude = exts_exclude or []
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)

    automatic_type = get_automatic_type(file_type)
    candidates = [f for ext in exts for f in folder.rglob(f"*.{ext}")
                 if not any(str(f).endswith(e) for e in exts_exclude)]

    _resources = []
    cmd_opts_no_hashing = shared.cmd_opts.no_hashing
    shared.cmd_opts.no_hashing = False

    try:
        for f in sorted(candidates):
            if f.is_dir():
                continue

            name = f.stem
            automatic_name = get_automatic_name(file_type, str(f), str(folder))
            file_hash = calculate_file_hash(str(f), automatic_type, automatic_name)

            _resources.append({
                'type': file_type,
                'name': name,
                'hash': file_hash,
                'path': str(f),
                'hasPreview': has_preview(str(f)),
                'hasInfo': has_info(str(f))
            })

    finally:
        shared.cmd_opts.no_hashing = cmd_opts_no_hashing

    return _resources

def get_all_by_hash_with_cache(file_hashes: List[str]) -> List[Dict[str, Any]]:
    """Get model information by hash with caching."""
    verbose = get_verbose()
    missing_info_hashes = [file_hash for file_hash in file_hashes if file_hash not in civil_ai_api_cache]
    new_results = []

    if missing_info_hashes:
        log_message(f"Fetching info for {len(missing_info_hashes)} missing hashes", status='info', verbose=verbose)

        try:
            for i in range(0, len(missing_info_hashes), 100):
                batch = missing_info_hashes[i:i + 100]
                new_results.extend(get_all_by_hash(batch))
        except Exception as e:
            log_message(f"Error fetching model info: {e}", status='error', verbose=verbose)
            raise e

    new_results = sorted(new_results, key=lambda x: datetime.fromisoformat(x['createdAt'].rstrip('Z')), reverse=True)

    found_info_hashes = set()

    for new_metadata in new_results:
        for file in new_metadata['files']:
            file_hash = file['hashes']['SHA256'].lower()
            civil_ai_api_cache[file_hash] = new_metadata
            found_info_hashes.add(file_hash)

    for file_hash in set(missing_info_hashes) - found_info_hashes:
        civil_ai_api_cache[file_hash] = None

    final_results = []

    for h in file_hashes:
        cached = civil_ai_api_cache.get(h)
        if cached:
            final_results.append(cached)

    return final_results

def load_resource_list(types: List[str] = None) -> List[Dict[str, Any]]:
    """Load resource list from all configured folders."""
    global resources
    verbose = get_verbose()

    if types is None:
        types = ['LORA', 'LoCon', 'Hypernetwork', 'TextualInversion', 'Checkpoint', 'VAE', 'Controlnet', 'Upscaler']

    res = [r for r in resources if r['type'] not in types]

    folders = {
        'LORA': Path(get_lora_dir()),
        'LoCon': Path(get_locon_dir()),
        'Hypernetwork': Path(getattr(shared.cmd_opts, 'hypernetwork_dir', '')),
        'TextualInversion': Path(getattr(shared.cmd_opts, 'embeddings_dir', '')),
        'Checkpoint': Path(get_model_dir()),
        'Controlnet': Path(models_path) / 'ControlNet',
        'Upscaler': Path(models_path) / 'ESRGAN',
        'VAE1': Path(get_model_dir()),
        'VAE2': Path(getattr(sd_vae, 'vae_path', '')),
    }

    # Load resources for each type
    type_configs = [
        ('LORA', folders['LORA'], ['pt', 'safetensors', 'ckpt']),
        ('LoCon', folders['LoCon'], ['pt', 'safetensors', 'ckpt']) if folders['LORA'] != folders['LoCon'] else None,
        ('Hypernetwork', folders['Hypernetwork'], ['pt', 'safetensors', 'ckpt']),
        ('TextualInversion', folders['TextualInversion'], ['pt', 'bin', 'safetensors']),
        ('Checkpoint', folders['Checkpoint'], ['safetensors', 'ckpt'], ['vae.safetensors', 'vae.ckpt']),
        ('Controlnet', folders['Controlnet'], ['safetensors', 'ckpt'], ['vae.safetensors', 'vae.ckpt']),
        ('Upscaler', folders['Upscaler'], ['safetensors', 'ckpt', 'pt']),
    ]

    for config in type_configs:
        if config is None:
            continue

        type_name, folder, exts = config[:3]
        exts_exclude = config[3] if len(config) > 3 else []

        if type_name in types:
            log_message(f"Loading {type_name} resources from {folder}", status='info', verbose=verbose)
            res += get_resources_in_folder(type_name, folder, exts, exts_exclude)

    # Handle VAE separately due to multiple folders
    if 'VAE' in types:
        log_message('Loading VAE resources', status='info', verbose=verbose)
        res += get_resources_in_folder('VAE', folders['VAE1'], ['vae.pt', 'vae.safetensors', 'vae.ckpt'])
        res += get_resources_in_folder('VAE', folders['VAE2'], ['pt', 'safetensors', 'ckpt'])

    resources = res
    return resources

def get_model_by_hash(file_hash: str) -> Optional[Any]:
    """Get model by hash from checkpoints list."""
    found = [info for info in sd_models.checkpoints_list.values()
             if file_hash == info.sha256 or file_hash == info.shorthash or file_hash == info.hash]
    return found[0] if found else None

def get_resource_by_hash(hash: str) -> Optional[Dict[str, Any]]:
    """Get resource by hash."""
    resources = load_resource_list([])
    found = [resource for resource in resources
             if hash.lower() == resource['hash'] and ('downloading' not in resource or resource['downloading'] != True)]
    return found[0] if found else None

def _resize_image_bytes(image_bytes, target_size=512):
    """Resize image bytes to target_size on the longer side, keeping aspect ratio."""
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size

    if width > height:
        new_size = (target_size, int(height * target_size / width))
    else:
        new_size = (int(width * target_size / height), target_size)

    output = io.BytesIO()
    image.resize(new_size, Image.LANCZOS).save(output, format='PNG')
    output.seek(0)
    return output

def download_preview(url: str, dest_path: str, on_progress: callable = None) -> None:
    """Download and resize preview image."""
    verbose = get_verbose()
    dest = Path(dest_path).expanduser()
    if dest.exists():
        return

    if verbose:
        log_message(f"Downloading preview: {url}", status='info', verbose=verbose)

    response = requests.get(url, stream=True, headers={'User-Agent': user_agent})
    total = int(response.headers.get('content-length', 0))
    start_time = time.time()

    try:
        image_data = bytearray()
        current = 0
        for data in response.iter_content(chunk_size=8192):
            image_data.extend(data)
            current += len(data)
            if on_progress is not None:
                should_stop = on_progress(current, total, start_time)
                if should_stop:
                    raise Exception('Download cancelled')

        resized_image = _resize_image_bytes(image_data)

        if IS_KAGGLE:
            import sd_encrypt_image     # Import Module for Encrypt Image

            img = Image.open(resized_image)
            imginfo = img.info or {}
            if not all(key in imginfo for key in ['Encrypt', 'EncryptPwdSha']):
                sd_encrypt_image.EncryptedImage.from_image(img).save(dest)
        else:
            dest.write_bytes(resized_image.read())

    except Exception as e:
        log_message(f"Preview download failed: {dest} : {e}", status='error', verbose=verbose)
        if dest.exists():
            dest.unlink()

def update_resource_preview(hash: str, preview_url: str) -> None:
    """Update preview for resource with given hash."""
    for res in [r for r in load_resource_list([]) if r['hash'] == hash.lower()]:
        preview_path = Path(res['path']).with_suffix('.preview.png')
        download_preview(preview_url, str(preview_path))