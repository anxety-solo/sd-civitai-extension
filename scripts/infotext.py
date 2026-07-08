import json
import os
import re

from modules.shared import opts
from modules import sd_vae, processing as P

from civitai.core import scan_resources

_orig_create_infotext = P.create_infotext


def _patched_create_infotext(*args, **kwargs):
    """Monkey-patched create_infotext that injects resource hashes into generation metadata"""
    result = _orig_create_infotext(*args, **kwargs)
    if not isinstance(result, str):
        return result
    if getattr(opts, 'ce_hashify_resources', False):
        extra = _compute_hashes(result)
        if extra:
            result = _merge_hashes(result, extra)
    return result


P.create_infotext = _patched_create_infotext


def _merge_hashes(infotext: str, hashes: dict) -> str:
    """Merge a hash dict into existing Hashes JSON in the infotext"""
    match = re.search(r'Hashes:\s*(\{.*?\})', infotext)
    if match:
        try:
            existing = json.loads(match.group(1))
        except json.JSONDecodeError:
            existing = {}
        existing.update(hashes)
        merged = f"Hashes: {json.dumps(existing)}"
        return re.sub(r'Hashes:\s*\{.*?\}', merged, infotext)
    return f"{infotext}, Hashes: {json.dumps(hashes)}"


def _compute_hashes(infotext: str) -> dict:
    """Find resource hashes referenced in the infotext for Civitai auto-detection"""
    if not getattr(opts, 'ce_hashify_resources', False):
        return {}
    parts  = infotext.strip().split('\n', 1)
    prompt = parts[0].strip()
    rest   = parts[1] if len(parts) > 1 else ''

    negative_prompt   = ''
    generation_params = rest
    if rest.startswith('Negative prompt:'):
        neg_parts = rest.split('\n', 1)
        negative_prompt   = neg_parts[0][len('Negative prompt:'):].strip()
        generation_params = neg_parts[1] if len(neg_parts) > 1 else ''

    resources = scan_resources([])
    result    = {}

    if sd_vae.loaded_vae_file is not None:
        vae_name = os.path.splitext(sd_vae.get_filename(sd_vae.loaded_vae_file))[0]
        match = next(
            (r for r in resources if r['type'] == 'VAE' and r['name'] == vae_name),
            None
        )
        if match:
            result['vae'] = match['hash'][:10]

    for emb in (r for r in resources if r['type'] == 'TextualInversion'):
        pat = re.compile(
            r'(?<![^\s:(|\[\]])' + re.escape(emb['name']) + r'(?![^\s:)|\[\]\,])',
            re.MULTILINE | re.IGNORECASE
        )
        if pat.search(prompt) or pat.search(negative_prompt):
            result[f"embed:{emb['name']}"] = emb['hash'][:10]

    for net_type, net_name, _ in re.findall(r'<(lora):([a-zA-Z0-9_.\-\s]+):([0-9.]+)(?:[:].*)?>', prompt):
        match = next(
            (
                res for res in resources if res['type'] == 'LORA' and (
                    res['name'].lower() == net_name.lower()
                    or res['name'].lower().split('-')[0] == net_name.lower()
                )
            ),
            None
        )
        if match:
            result[f"{net_type}:{net_name}"] = match['hash'][:10]

    model_match = re.search(r'Model hash: ([0-9a-fA-F]{10})', generation_params)
    if model_match:
        sha256 = model_match.group(1)
        match = next(
            (r for r in resources if r['type'] == 'Checkpoint' and r['hash'].startswith(sha256)),
            None
        )
        if match:
            result['model'] = match['hash'][:10]

    return result
