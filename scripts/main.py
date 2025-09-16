import threading
import json
import gradio as gr
from pathlib import Path

# ===  WebUI imports ===
from modules.script_callbacks import on_app_started, on_ui_settings
from modules import shared

# === Extension imports ===
import civitai.lib as civitai
from civitai.logger import log_message
from civitai.lib import get_verbose


lock = threading.Lock()
types = ['LORA', 'LoCon', 'TextualInversion', 'Checkpoint']
VERSION = '2.2'


def _process_resources(process_type: str, missing_condition: callable, process_func: callable) -> None:
    """Common function to process resources (info or preview)"""
    verbose = get_verbose()
    missing = [r for r in civitai.load_resource_list() if r['type'] in types and missing_condition(r)]
    hashes = [r['hash'] for r in missing]

    if not hashes:
        return

    results = civitai.get_all_by_hash_with_cache(hashes)
    if not results:
        return

    log_message(f'Checking resources for missing {process_type}...', status='info', verbose=True)

    updated_count = 0
    processed_paths = set()

    for r in results:
        if r is None:
            continue

        for f in r['files']:
            if 'hashes' not in f or 'SHA256' not in f['hashes']:
                continue

            sha256 = f['hashes']['SHA256'].lower()
            if sha256 not in hashes:
                continue

            matching_resources = [res for res in missing if sha256 == res['hash']]
            if not matching_resources:
                continue

            for resource in matching_resources:
                if process_func(r, resource, processed_paths, verbose):
                    updated_count += 1

    if updated_count > 0:
        log_message(f"Updated {updated_count} {process_type} files", status='info', verbose=True)

def _process_info_file(r: dict, resource: dict, processed_paths: set, verbose: bool) -> bool:
    """Process info file for a resource."""
    path = Path(resource['path']).with_suffix('.json')
    if str(path) in processed_paths:
        return False

    base_list = {
        'SD 1': 'SD1', 'SD 1.5': 'SD1', 'SD 2': 'SD2', 'SD 3': 'SD3',
        'SDXL': 'SDXL', 'Pony': 'SDXL', 'Illustrious': 'SDXL'
    }

    path.write_text(json.dumps({
        'activation text': ', '.join(r.get('trainedWords', [])),
        'sd version': next((v for k, v in base_list.items() if k in r.get('baseModel', '')), ''),
        'modelId': r['modelId'],
        'modelVersionId': r['id'],
        'sha256': r['files'][0]['hashes']['SHA256'].upper()
    }, indent=4), encoding='utf-8')

    if path.exists() and path.stat().st_size > 0:
        processed_paths.add(str(path))
        log_message(f"Updated info for: {resource['name']}", status='success', verbose=verbose)
        return True
    else:
        log_message(f"Failed to write info for: {resource['name']}", status='error', verbose=verbose)
        return False

def load_info() -> None:
    """Load missing info files for resources."""
    _process_resources('info', lambda r: not r['hasInfo'], _process_info_file)

def _process_preview_file(r: dict, resource: dict, processed_paths: set, verbose: bool) -> bool:
    """Process preview file for a resource."""
    images = r.get('images', [])
    if not images:
        return False

    preview = next((p for p in images if not p['url'].lower().endswith(('.mp4', '.gif'))), None)
    if preview is None:
        return False

    preview_path = Path(resource['path']).with_suffix('.preview.png')
    if str(preview_path) in processed_paths:
        return False

    if civitai.update_resource_preview(resource['hash'], preview['url']):
        processed_paths.add(str(preview_path))
        log_message(f"Updated preview for: {resource['name']}", status='success', verbose=verbose)
        return True
    else:
        log_message(f"Failed to update preview for: {resource['name']}", status='error', verbose=verbose)
        return False

def load_preview() -> None:
    """Load missing preview images for resources."""
    _process_resources('preview', lambda r: not r['hasPreview'], _process_preview_file)

def _run_with_lock(func: callable, error_msg: str) -> None:
    """Run function in thread-safe manner."""
    verbose = get_verbose()
    with lock:
        try:
            func()
        except Exception as e:
            log_message(f"{error_msg}: {e}", status='error', verbose=verbose)

def run_load_info():
    """Run info loading in thread-safe manner."""
    _run_with_lock(load_info, "Error loading info")

def run_load_preview():
    """Run preview loading in thread-safe manner."""
    _run_with_lock(load_preview, "Error loading previews")


def app(_: gr.Blocks, app):
    """Initialize extension on app start."""
    log_message(f"Starting CivitAI-Extension \033[32mV{VERSION}\033[0m", status='info', verbose=True)

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
        shared.OptionInfo(
            default='',
            label='Your Civitai API Key',
            section=section
        ).info('You can find your API key in your CivitAI account settings')
    )

    # Resource metadata settings
    shared.opts.add_option(
        'civitai_hashify_resources',
        shared.OptionInfo(
            default=True,
            label='Include resource hashes in image metadata (for resource auto-detection on Civitai)',
            section=section
        )
    )

    # Directory configuration
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

    # Debug settings
    shared.opts.add_option(
        'civitai_verbose',
        shared.OptionInfo(
            default=False,
            label='Enable verbose logging for Civitai extension',
            section=section
        ).info('Enable this option to see detailed log messages from the Civitai extension.')
    )
    shared.opts.add_option(
        'civitai_suppress_hash_output',
        shared.OptionInfo(
            default=True,
            label='Suppress hash calculation output messages',
            section=section
        ).info('If enabled, hash calculation messages will not be shown in the console output.')
    )


on_app_started(app)
on_ui_settings(on_settings)