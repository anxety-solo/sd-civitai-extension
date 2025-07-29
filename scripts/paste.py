from typing import Dict
import json
import re
import os

from modules import sd_vae, shared, processing as P

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

def civitai_hashes(infotext: str) -> Dict[str, str]:
    """Extract and match resource hashes from infotext."""
    additional_network_type_map = {'lora': 'LORA'}
    additional_network_pattern = r'<(lora):([a-zA-Z0-9_\.\-\s]+):([0-9.]+)(?:[:].*)?>'
    model_hash_pattern = r'Model hash: ([0-9a-fA-F]{10})'

    hashify_resources = shared.opts.data.get('civitai_hashify_resources', True)
    if not hashify_resources:
        return {}

    parts = infotext.strip().split("\n", 1)

    prompt = parts[0].strip()
    rest = parts[1] if len(parts) > 1 else ''

    negative_prompt = ''
    generation_params = rest

    if rest.startswith('Negative prompt:'):
        neg_parts = rest.split("\n", 1)
        negative_prompt = neg_parts[0][len('Negative prompt:'):].strip()
        generation_params = neg_parts[1] if len(neg_parts) > 1 else ''

    resources = civitai.load_resource_list([])
    resource_hashes = {}

    # Add VAE hash if loaded
    if sd_vae.loaded_vae_file is not None:
        vae_name = os.path.splitext(sd_vae.get_filename(sd_vae.loaded_vae_file))[0]
        vae_matches = [r for r in resources if r['type'] == 'VAE' and r['name'] == vae_name]
        if vae_matches:
            resource_hashes['vae'] = vae_matches[0]['hash'][:10]

    # Add embedding hashes
    for embedding in [r for r in resources if r['type'] == 'TextualInversion']:
        pattern = re.compile(r'(?<![^\s:(|\[\]])' + re.escape(embedding['name']) + r'(?![^\s:)|\[\]\,])',
                           re.MULTILINE | re.IGNORECASE)
        if pattern.search(prompt) or pattern.search(negative_prompt):
            resource_hashes[f"embed:{embedding['name']}"] = embedding['hash'][:10]

    # Add additional network hashes (LoRA)
    for match in re.findall(additional_network_pattern, prompt):
        network_type, network_name, _ = match
        resource_type = additional_network_type_map[network_type]
        matched = [r for r in resources if r['type'] == resource_type and (
            r['name'].lower() == network_name.lower() or
            r['name'].lower().split('-')[0] == network_name.lower()
        )]
        if matched:
            resource_hashes[f"{network_type}:{network_name}"] = matched[0]['hash'][:10]

    # Add model hash
    model_match = re.search(model_hash_pattern, generation_params)
    if model_match:
        model_hash = model_match.group(1)
        matched = [r for r in resources if r['type'] == 'Checkpoint' and r['hash'].startswith(model_hash)]
        if matched:
            resource_hashes['model'] = matched[0]['hash'][:10]

    return resource_hashes