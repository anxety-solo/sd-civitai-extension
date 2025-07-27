from pathlib import Path
import gradio as gr
import threading
import json

from modules.script_callbacks import on_app_started, on_ui_settings
from modules import shared

import civitai.lib as civitai
from civitai.logger import log_message


lock = threading.Lock()
types = ['LORA', 'LoCon', 'Hypernetwork', 'TextualInversion', 'Checkpoint']
VERSION = '2'

def get_verbose() -> bool:
    """Get verbose logging setting."""
    return getattr(shared.opts, 'civitai_verbose', False)

def load_info() -> None:
    """Load missing info files for resources."""
    verbose = get_verbose()
    missing = [r for r in civitai.load_resource_list() if r['type'] in types and not r['hasInfo']]
    hashes = [r['hash'] for r in missing]

    if not hashes:
        return

    results = civitai.get_all_by_hash_with_cache(hashes)
    if not results:
        return

    log_message('Checking resources for missing info files...', status='info', verbose=True)

    base_list = {
        'SD 1': 'SD1',
        'SD 1.5': 'SD1',
        'SD 2': 'SD2',
        'SD 3': 'SD3',
        'SDXL': 'SDXL',
        'Pony': 'SDXL',
        'Illustrious': 'SDXL',
    }

    updated_count = 0

    for r in results:
        if r is None:
            continue

        for f in r['files']:
            if 'hashes' not in f or 'SHA256' not in f['hashes']:
                continue

            sha256 = f['hashes']['SHA256']
            if sha256.lower() not in hashes:
                continue

            data = {
                'activation text': ', '.join(r.get('trainedWords', [])),
                'sd version': next((v for k, v in base_list.items() if k in r.get('baseModel', '')), ''),
                'modelId': r['modelId'],
                'modelVersionId': r['id'],
                'sha256': sha256.upper()
            }

            matching_resources = [r for r in missing if sha256.lower() == r['hash']]
            if not matching_resources:
                continue

            for resource in matching_resources:
                path = Path(resource['path']).with_suffix('.json')
                path.write_text(json.dumps(data, indent=4), encoding='utf-8')
                updated_count += 1
                if verbose:
                    log_message(f"Updated info for: {resource['name']}", status='success', verbose=verbose)

    if updated_count > 0:
        log_message(f"Updated {updated_count} info files", status='info', verbose=True)

def load_preview() -> None:
    """Load missing preview images for resources."""
    verbose = get_verbose()
    hashes = [r['hash'] for r in civitai.load_resource_list() if r['type'] in types and not r['hasPreview']]

    if not hashes:
        return

    results = civitai.get_all_by_hash_with_cache(hashes)
    if not results:
        return

    log_message('Checking resources for missing preview images...', status='info', verbose=True)

    updated_count = 0

    for r in results:
        if r is None:
            continue

        for f in r['files']:
            if 'hashes' not in f or 'SHA256' not in f['hashes']:
                continue

            sha256 = f['hashes']['SHA256']
            if sha256.lower() not in hashes:
                continue

            images = r.get('images', [])
            if not images:
                continue

            preview = next((p for p in images if not p['url'].lower().endswith(('.mp4', '.gif'))), None)
            if preview is None:
                continue

            civitai.update_resource_preview(sha256, preview['url'])
            updated_count += 1
            if verbose:
                log_message(f"Updated preview for: {r['name']}", status='success', verbose=verbose)

    if updated_count > 0:
        log_message(f"Updated {updated_count} preview images", status='info', verbose=True)

def run_load_info():
    """Run info loading in thread-safe manner."""
    verbose = get_verbose()
    with lock:
        try:
            load_info()
        except Exception as e:
            log_message(f"Error loading info: {e}", status='error', verbose=verbose)

def run_load_preview():
    """Run preview loading in thread-safe manner."""
    verbose = get_verbose()
    with lock:
        try:
            load_preview()
        except Exception as e:
            log_message(f"Error loading previews: {e}", status='error', verbose=verbose)

def app(_: gr.Blocks, app):
    """Initialize extension on app start."""
    log_message(f"Starting CivitAI extension \033[32mV{VERSION}\033[0m", status='info', verbose=True)

    info_thread = threading.Thread(target=run_load_info)
    preview_thread = threading.Thread(target=run_load_preview)

    info_thread.start()
    preview_thread.start()

def on_settings():
    """Register extension settings in the WebUI settings panel."""
    section = ('civitai_extension', 'CivitAI')

    # API configuration
    shared.opts.add_option(
        'civitai_api_key',
        shared.OptionInfo('', 'Your Civitai API Key', section=section)
    )

    # Resource metadata settings
    shared.opts.add_option(
        'civitai_hashify_resources',
        shared.OptionInfo(
            True,
            'Include resource hashes in image metadata (for resource auto-detection on Civitai)',
            section=section
        )
    )

    # Directory configuration
    shared.opts.add_option(
        'civitai_folder_model',
        shared.OptionInfo('', 'Models directory (if not default)', section=section)
    )
    shared.opts.add_option(
        'civitai_folder_lora',
        shared.OptionInfo('', 'LoRA directory (if not default)', section=section)
    )
    shared.opts.add_option(
        'civitai_folder_lyco',
        shared.OptionInfo('', 'LyCORIS directory (if not default)', section=section)
    )

    # Debug settings
    shared.opts.add_option(
        'civitai_verbose',
        shared.OptionInfo(False, 'Enable verbose logging for Civitai extension', section=section)
    )
    shared.opts.add_option(
        'civitai_suppress_hash_output',
        shared.OptionInfo(True, 'Suppress hash calculation output messages', section=section)
    )


on_app_started(app)
on_ui_settings(on_settings)