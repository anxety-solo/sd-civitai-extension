import threading
import json
import gradio as gr
from pathlib import Path

from modules.script_callbacks import on_app_started, on_ui_settings
from modules import shared

import civitai.lib as civitai
from civitai.lib import log, _verbose


VERSION = '3.0'
_lock = threading.Lock()
_TYPES = ['LORA', 'LoCon', 'TextualInversion', 'Checkpoint']

_BASE_MODELS = {
    'SD 1': 'SD1', 'SD 1.5': 'SD1', 'SD 2': 'SD2', 'SD 3': 'SD3',
    'SDXL': 'SDXL', 'Pony': 'SDXL', 'Illustrious': 'SDXL',
}


# ~~ Processing ~~

def _process_resources(label: str, missing_cond, process_fn):
    """Common function to process resources (info or preview)"""
    missing = [r for r in civitai.load_resource_list() if r['type'] in _TYPES and missing_cond(r)]
    if not missing:
        return

    hashes = [r['hash'] for r in missing]
    results = civitai.get_all_by_hash_with_cache(hashes)
    if not results:
        return

    log.info(f"Checking resources for missing {label}...")
    updated = 0
    processed = set()

    for meta in results:
        if meta is None:
            continue
        for f in meta['files']:
            if 'hashes' not in f or 'SHA256' not in f['hashes']:
                continue
            sha = f['hashes']['SHA256'].lower()
            if sha not in hashes:
                continue
            for res in [r for r in missing if r['hash'] == sha]:
                if process_fn(meta, res, processed):
                    updated += 1

    if updated:
        log.info(f"Updated {updated} {label} files")

def _process_info_file(meta: dict, res: dict, processed: set) -> bool:
    """Process info file for a resource"""
    path = Path(res['path']).with_suffix('.json')
    if str(path) in processed:
        return False

    sd_ver = next((v for k, v in _BASE_MODELS.items() if k in meta.get('baseModel', '')), '')
    path.write_text(json.dumps({
        'activation text': ', '.join(meta.get('trainedWords', [])),
        'sd version': sd_ver,
        'modelId': meta['modelId'],
        'modelVersionId': meta['id'],
        'sha256': meta['files'][0]['hashes']['SHA256'].upper(),
    }, indent=4), encoding='utf-8')

    if path.exists() and path.stat().st_size > 0:
        processed.add(str(path))
        log.success(f"Updated info for: {res['name']}", _verbose())
        return True

    log.error(f"Failed to write info for: {res['name']}")
    return False

def _process_preview_file(meta: dict, res: dict, processed: set) -> bool:
    """Process preview file for a resource"""
    images = meta.get('images', [])
    # Skip animated formats — prefer static images for previews
    preview = next((p for p in images if not p['url'].lower().endswith(('.mp4', '.gif'))), None)
    if not preview:
        return False

    preview_path = Path(res['path']).with_suffix('.preview.png')
    if str(preview_path) in processed:
        return False

    if civitai.update_resource_preview(res['hash'], preview['url']):
        processed.add(str(preview_path))
        log.success(f"Updated preview for: {res['name']}", _verbose())
        return True

    log.error(f"Failed to update preview for: {res['name']}")
    return False

def load_info():
    """Load missing info files for resources"""
    _process_resources('info', lambda r: not r['hasInfo'], _process_info_file)

def load_preview():
    """Load missing preview images for resources"""
    _process_resources('preview', lambda r: not r['hasPreview'], _process_preview_file)

def _run_with_lock(func, label: str):
    """Run function in thread-safe manner"""
    with _lock:
        try:
            func()
        except Exception as e:
            log.error(f"Error in {label}: {e}")


# ~~ App startup ~~

def app(_: gr.Blocks, app):
    """Initialize extension on app start"""
    log.info(f"Starting CivitAI-Extension \033[32mV{VERSION}\033[0m")

    threading.Thread(target=_run_with_lock, args=(load_info, 'info')).start()
    threading.Thread(target=_run_with_lock, args=(load_preview, 'preview')).start()


# ~~ Settings ~~

def on_settings():
    """Register extension settings in the WebUI settings panel"""
    section = ('civitai_extension', 'CivitAI')

    shared.opts.add_option(
        'civitai_api_key',
        shared.OptionInfo(
            default='',
            label='Your Civitai API Key',
            section=section
        ).info('You can find your API key in your CivitAI account settings')
    )
    shared.opts.add_option(
        'civitai_hashify_resources',
        shared.OptionInfo(
            default=True,
            label='Include resource hashes in image metadata (for resource auto-detection on Civitai)',
            section=section
        )
    )
    shared.opts.add_option(
        'civitai_folder_model',
        shared.OptionInfo(
            default='',
            label='Models directory (if not default)',
            section=section
        ).info('Specify a custom directory for models')
    )
    shared.opts.add_option(
        'civitai_folder_lora',
        shared.OptionInfo(
            default='',
            label='LoRA directory (if not default)',
            section=section
        ).info('Specify a custom directory for LoRA files')
    )
    shared.opts.add_option(
        'civitai_folder_lyco',
        shared.OptionInfo(
            default='',
            label='LyCORIS directory (if not default)',
            section=section
        ).info('Specify a custom directory for LyCORIS files')
    )
    shared.opts.add_option(
        'civitai_verbose',
        shared.OptionInfo(
            default=False,
            label='Enable verbose logging for Civitai extension',
            section=section
        ).info('Enable this option to see detailed log messages from the Civitai extension')
    )
    shared.opts.add_option(
        'civitai_suppress_hash_output',
        shared.OptionInfo(
            default=True,
            label='Suppress hash calculation output messages',
            section=section
        ).info('If enabled, hash calculation messages will not be shown in the console output')
    )


on_app_started(app)
on_ui_settings(on_settings)