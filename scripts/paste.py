import json
import re
import os
from typing import Dict

# === WebUI imports ===
from modules import sd_vae, shared, processing as P

# === Extension imports ===
import civitai.lib as civitai


create_infotext = P.create_infotext


def insert_infotext(*args, **kwargs) -> str:
    """Insert infotext with optional hash information."""
    infotext = create_infotext(*args, **kwargs)
    if not isinstance(infotext, str):
        return infotext

    if shared.opts.data.get('civitai_hashify_resources', True):
        hashes = civitai_hashes(infotext)
        if hashes:
            infotext = merge_infotext(infotext, hashes)
    return infotext

P.create_infotext = insert_infotext

def merge_infotext(infotext: str, hashtext: Dict[str, str]) -> str:
    """Merge hash information into existing infotext."""
    hashes_match = re.search(r'Hashes:\s*(\{.*?\})', infotext)
    if hashes_match:
        try:
            existing_hashes = json.loads(hashes_match.group(1))
        except json.JSONDecodeError:
            existing_hashes = {}
        existing_hashes.update(hashtext)
        merged_hashes = f"Hashes: {json.dumps(existing_hashes)}"
        infotext = re.sub(r'Hashes:\s*\{.*?\}', merged_hashes, infotext)
    else:
        infotext += f", Hashes: {json.dumps(hashtext)}"
    return infotext

def _extract_prompt_parts(infotext: str) -> tuple:
    """Extract prompt, negative prompt, and generation params from infotext."""
    parts = infotext.strip().split("\n", 1)
    prompt = parts[0].strip()
    rest = parts[1] if len(parts) > 1 else ''

    if rest.startswith('Negative prompt:'):
        neg_parts = rest.split("\n", 1)
        negative_prompt = neg_parts[0][len('Negative prompt:'):].strip()
        generation_params = neg_parts[1] if len(neg_parts) > 1 else ''
    else:
        negative_prompt = ''
        generation_params = rest

    return prompt, negative_prompt, generation_params

def _add_vae_hash(resources: list, resource_hashes: dict) -> None:
    """Add VAE hash if loaded."""
    if sd_vae.loaded_vae_file is not None:
        vae_name = os.path.splitext(sd_vae.get_filename(sd_vae.loaded_vae_file))[0]
        vae_matches = [r for r in resources if r['type'] == 'VAE' and r['name'] == vae_name]
        if vae_matches:
            resource_hashes['vae'] = vae_matches[0]['hash'][:10]

def _add_embedding_hashes(resources: list, prompt: str, negative_prompt: str, resource_hashes: dict) -> None:
    """Add embedding hashes"""
    for embedding in [r for r in resources if r['type'] == 'TextualInversion']:
        pattern = re.compile(r'(?<![^\s:(|\[\]])' + re.escape(embedding['name']) + r'(?![^\s:)|\[\]\,])',
                           re.MULTILINE | re.IGNORECASE)
        if pattern.search(prompt) or pattern.search(negative_prompt):
            resource_hashes[f"embed:{embedding['name']}"] = embedding['hash'][:10]

def _add_lora_hashes(resources: list, prompt: str, resource_hashes: dict) -> None:
    """Add LoRA hashes"""
    additional_network_pattern = r'<(lora):([a-zA-Z0-9_\.\-\s]+):([0-9.]+)(?:[:].*)?>'

    for match in re.findall(additional_network_pattern, prompt):
        network_type, network_name, _ = match
        matched = [r for r in resources if r['type'] == 'LORA' and (
            r['name'].lower() == network_name.lower() or
            r['name'].lower().split('-')[0] == network_name.lower()
        )]
        if matched:
            resource_hashes[f"{network_type}:{network_name}"] = matched[0]['hash'][:10]

def _add_model_hash(resources: list, generation_params: str, resource_hashes: dict) -> None:
    """Add model (checkpoint) hash"""
    model_hash_pattern = r'Model hash: ([0-9a-fA-F]{10})'
    model_match = re.search(model_hash_pattern, generation_params)
    if model_match:
        model_hash = model_match.group(1)
        matched = [r for r in resources if r['type'] == 'Checkpoint' and r['hash'].startswith(model_hash)]
        if matched:
            resource_hashes['model'] = matched[0]['hash'][:10]

def civitai_hashes(infotext: str) -> Dict[str, str]:
    """Extract and match resource hashes from infotext."""
    if not shared.opts.data.get('civitai_hashify_resources', True):
        return {}

    prompt, negative_prompt, generation_params = _extract_prompt_parts(infotext)
    resources = civitai.load_resource_list([])
    resource_hashes = {}

    _add_vae_hash(resources, resource_hashes)
    _add_embedding_hashes(resources, prompt, negative_prompt, resource_hashes)
    _add_lora_hashes(resources, prompt, resource_hashes)
    _add_model_hash(resources, generation_params, resource_hashes)

    return resource_hashes