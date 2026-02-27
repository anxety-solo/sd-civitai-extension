import json
import re
import os
from typing import Dict

from modules import sd_vae, shared, processing as P

import civitai.lib as civitai


# ~~ Infotext patching ~~

_orig_create_infotext = P.create_infotext

def _insert_infotext(*args, **kwargs) -> str:
    """Insert infotext with optional hash information"""
    infotext = _orig_create_infotext(*args, **kwargs)
    if not isinstance(infotext, str):
        return infotext
    if shared.opts.data.get('civitai_hashify_resources', True):
        extra = civitai_hashes(infotext)
        if extra:
            infotext = merge_infotext(infotext, extra)
    return infotext

P.create_infotext = _insert_infotext


# ~~ Hash helpers ~~

def merge_infotext(infotext: str, hashtext: Dict[str, str]) -> str:
    """Merge hash information into existing infotext"""
    match = re.search(r'Hashes:\s*(\{.*?\})', infotext)
    if match:
        try:
            existing = json.loads(match.group(1))
        except json.JSONDecodeError:
            existing = {}
        existing.update(hashtext)
        merged = f"Hashes: {json.dumps(existing)}"
        return re.sub(r'Hashes:\s*\{.*?\}', merged, infotext)
    return infotext + f", Hashes: {json.dumps(hashtext)}"

def _extract_prompt_parts(infotext: str) -> tuple[str, str, str]:
    """Extract prompt, negative prompt, and generation params from infotext"""
    parts = infotext.strip().split('\n', 1)
    prompt = parts[0].strip()
    rest = parts[1] if len(parts) > 1 else ''

    if rest.startswith('Negative prompt:'):
        neg_parts = rest.split('\n', 1)
        negative_prompt = neg_parts[0][len('Negative prompt:'):].strip()
        generation_params = neg_parts[1] if len(neg_parts) > 1 else ''
    else:
        negative_prompt = ''
        generation_params = rest

    return prompt, negative_prompt, generation_params

def _add_vae_hash(resources: list, resource_hashes: dict):
    """Add VAE hash if a VAE is currently loaded"""
    if sd_vae.loaded_vae_file is not None:
        vae_name = os.path.splitext(sd_vae.get_filename(sd_vae.loaded_vae_file))[0]
        match = next((r for r in resources if r['type'] == 'VAE' and r['name'] == vae_name), None)
        if match:
            resource_hashes['vae'] = match['hash'][:10]

def _add_embedding_hashes(resources: list, prompt: str, negative_prompt: str, resource_hashes: dict):
    """Add embedding hashes for any embeddings found in prompt or negative prompt"""
    for emb in [r for r in resources if r['type'] == 'TextualInversion']:
        pattern = re.compile(
            r'(?<![^\s:(|\[\]])' + re.escape(emb['name']) + r'(?![^\s:)|\[\]\,])',
            re.MULTILINE | re.IGNORECASE
        )
        if pattern.search(prompt) or pattern.search(negative_prompt):
            resource_hashes[f"embed:{emb['name']}"] = emb['hash'][:10]

def _add_lora_hashes(resources: list, prompt: str, resource_hashes: dict):
    """Add LoRA hashes for any LoRA networks referenced in the prompt"""
    pattern = r'<(lora):([a-zA-Z0-9_.\-\s]+):([0-9.]+)(?:[:].*)?>'
    for net_type, net_name, _ in re.findall(pattern, prompt):
        match = next((
            r for r in resources if r['type'] == 'LORA' and (
                r['name'].lower() == net_name.lower() or
                r['name'].lower().split('-')[0] == net_name.lower()
            )
        ), None)
        if match:
            resource_hashes[f"{net_type}:{net_name}"] = match['hash'][:10]

def _add_model_hash(resources: list, generation_params: str, resource_hashes: dict):
    """Add model (checkpoint) hash if found in generation params"""
    model_match = re.search(r'Model hash: ([0-9a-fA-F]{10})', generation_params)
    if model_match:
        h = model_match.group(1)
        match = next((r for r in resources if r['type'] == 'Checkpoint' and r['hash'].startswith(h)), None)
        if match:
            resource_hashes['model'] = match['hash'][:10]

def civitai_hashes(infotext: str) -> Dict[str, str]:
    """Extract and match resource hashes from infotext"""
    if not shared.opts.data.get('civitai_hashify_resources', True):
        return {}

    prompt, negative_prompt, generation_params = _extract_prompt_parts(infotext)
    resources = civitai.load_resource_list([])
    resource_hashes: Dict[str, str] = {}

    _add_vae_hash(resources, resource_hashes)
    _add_embedding_hashes(resources, prompt, negative_prompt, resource_hashes)
    _add_lora_hashes(resources, prompt, resource_hashes)
    _add_model_hash(resources, generation_params, resource_hashes)

    return resource_hashes