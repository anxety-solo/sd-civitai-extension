import requests
import json
from typing import Dict, Any, Optional

from modules.shared import opts

from .logger import log


BASE_URL   = 'https://civitai.com/api/v1'
USER_AGENT = 'CivitaiLink:Automatic1111'

_api_cache:   Dict[str, Optional[Dict[str, Any]]] = {}
_model_cache: Dict[int, Dict[str, Any]]           = {}


def _req(endpoint: str, method: str = 'GET', data: Any = None, params: Dict[str, str] = None) -> Dict[str, Any]:
    """Make an HTTP request to the Civitai API and return parsed JSON"""
    api_key = getattr(opts, 'ce_api_key', '')
    headers = {'User-Agent': USER_AGENT}

    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"

    if data is not None:
        headers['Content-Type'] = 'application/json'
        data = json.dumps(data)

    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    params = params or {}
    params.setdefault('nsfw', 'true')

    resp = requests.request(method, url, data=data, params=params, headers=headers)

    if resp.status_code != 200:
        raise Exception(f"Civitai API error {resp.status_code}: {resp.text}")
    return resp.json()


def get_by_hash(hashes: list) -> list:
    """Look up model versions by SHA256 hashes (batches of 100)"""
    results = []
    for i in range(0, len(hashes), 100):
        batch = hashes[i:i + 100]
        results.extend(_req('model-versions/by-hash', method='POST', data=batch))
    return results


def get_by_hash_cached(hashes: list) -> list:
    """Look up model versions by SHA256 with in-memory caching"""
    missing = [h for h in hashes if h not in _api_cache]
    if missing:
        log.success(f"Fetching info for {len(missing)} hashes")

        try:
            found = get_by_hash(missing)
        except Exception as e:
            log.error(f"API fetch failed: {e}")
            found = []

        seen = set()
        for ver in found:
            for file in ver.get('files', []):
                sha256 = file.get('hashes', {}).get('SHA256', '').lower()
                if sha256:
                    _api_cache[sha256] = ver
                    seen.add(sha256)

        for hash in set(missing) - seen:
            _api_cache[hash] = None

    return [v for h in hashes if (v := _api_cache.get(h))]


def get_model(model_id: int) -> Dict[str, Any]:
    """Fetch full model info by model ID (with cache)"""
    if model_id not in _model_cache:
        _model_cache[model_id] = _req(f'models/{model_id}')
    return _model_cache[model_id]
