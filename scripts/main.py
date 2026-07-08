import threading

import gradio as gr

from modules.script_callbacks import on_app_started, on_ui_settings
from modules.shared import opts, OptionInfo

import civitai.api as api

from civitai.core import scan_resources, save_json, save_preview
from civitai.logger import log


VERSION  = '4.2'
_lock    = threading.Lock()
_TYPES   = ['LORA', 'LoCon', 'DoRA', 'TextualInversion', 'Checkpoint']
_SECTION = ('ce_extension', 'CivitAI-Extension')


def _opt(key: str, default, label: str, *, info: str = None):
    """Register a CivitAI extension option in WebUI settings"""
    opt = OptionInfo(default=default, label=label, section=_SECTION)
    if info:
        opt.info(info)
    opts.add_option(key, opt)


def _process_resources(label: str, cond, save_fn):
    """Find resources matching a condition, look them up via API, and apply save_fn"""
    resources = scan_resources(_TYPES)
    missing   = [r for r in resources if r['type'] in _TYPES and cond(r)]
    if not missing:
        return

    hashes  = [r['hash'] for r in missing]
    results = api.get_by_hash_cached(hashes)
    if not results:
        return

    log.success(f"Updating {label} for {len(missing)} resources...")

    updated   = 0
    processed = set()

    for version in results:
        if version is None:
            continue
        for file in version.get('files', []):
            sha256 = file.get('hashes', {}).get('SHA256', '').lower()
            if not sha256:
                continue
            for res in missing:
                if res['hash'] == sha256:
                    if res['path'] in processed:
                        continue
                    if save_fn(version, res):
                        updated += 1
                        processed.add(res['path'])
                    break

    if updated:
        log.info(f"Updated {updated} {label} files")


def _process_info(version: dict, resource: dict) -> bool:
    skip = getattr(opts, 'ce_skip_existing', True)
    ok   = save_json(version, resource['path'], resource['hash'], skip_existing=skip)
    if ok:
        log.success(f"Saved info for: {resource['name']}")
    return ok


def _process_preview(version: dict, resource: dict) -> bool:
    images  = version.get('images', [])
    preview = next(
        (p for p in images if not p.get('url', '').lower().endswith(('.mp4', '.gif'))),
        None
    )
    if not preview:
        return False

    skip = getattr(opts, 'ce_skip_existing', True)
    ok   = save_preview(preview['url'], resource['path'], skip_existing=skip)
    if ok:
        log.success(f"Saved preview for: {resource['name']}")
    return ok


def load_info():
    _process_resources('info', lambda r: not r['has_info'], _process_info)


def load_preview():
    _process_resources('preview', lambda r: not r['has_preview'], _process_preview)


def _run_locked(label: str, fn):
    with _lock:
        try:
            fn()
        except Exception as e:
            log.error(f"Error in {label}: {e}")


def on_app_start(_: gr.Blocks, app):
    """Extension entry point — launches background threads for info/preview sync"""
    log.info(f"Starting \033[32mV{VERSION}\033[0m")

    threading.Thread(target=_run_locked, args=('info', load_info)).start()
    threading.Thread(target=_run_locked, args=('preview', load_preview)).start()


def setup_settings():
    """Register CivitAI extension options in the WebUI settings panel"""

    _opt('ce_api_key', '', 'Civitai API Key', info='Optional — for authenticated requests')
    _opt('ce_skip_existing', True, 'Skip existing preview/info files (no overwrite)')
    _opt('ce_save_description', False, 'Save description in JSON (requires extra API call)')
    _opt('ce_hashify_resources', False, 'Include resource hashes in image metadata')
    _opt('ce_suppress_hash_output', True, 'Suppress hash calculation output')
    _opt('ce_verbose', False, 'Verbose logging')


on_app_started(on_app_start)
on_ui_settings(setup_settings)
