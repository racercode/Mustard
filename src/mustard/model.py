"""Ollama transport with validated, content-addressed successful responses."""
import json
import time
import urllib.request
from pathlib import Path

from .storage import digest_json, write_json


def nonempty(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('model returned empty text')
    return text.strip()


class Ollama:
    def __init__(self, host, cache, attempts=3, timeout=600, transport=None, identities=None):
        self.host = host.rstrip('/')
        self.cache = Path(cache)
        self.attempts = attempts
        self.timeout = timeout
        self.transport = transport or self._http
        self.identities = identities or {}

    def _http(self, payload):
        request = urllib.request.Request(self.host + '/api/chat',
                                         data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def resolve_models(self, models):
        """Resolve mutable tags before using cached calls."""
        with urllib.request.urlopen(self.host + '/api/tags', timeout=self.timeout) as response:
            available = {item['name']: item['digest'] for item in json.load(response)['models']}
        for model in models:
            if model not in available:
                raise ValueError(f'model {model} is not available at {self.host}')
            self.identities[model] = available[model]
        return dict(self.identities)

    def chat(self, model, messages, options=None, validate=nonempty):
        if model not in self.identities:
            raise ValueError(f'no resolved model digest for {model}')
        payload = dict(model=model, messages=messages, options=options or {}, stream=False)
        key = digest_json(dict(request=payload, model_digest=self.identities[model]))
        path = self.cache / f'{key}.json'
        if path.exists():
            record = json.loads(path.read_text())
            text = nonempty(record['text'])
            validate(text)
            return text
        last_error = None
        for attempt in range(self.attempts):
            try:
                response = self.transport(payload)
                text = nonempty(response['message']['content'])
                validate(text)
                metadata = {k: response[k] for k in ('model', 'total_duration', 'load_duration',
                            'prompt_eval_count', 'eval_count') if k in response}
                write_json(path, dict(text=text, model_digest=self.identities[model], request_hash=key,
                                     options=payload['options'], metadata=metadata))
                return text
            except Exception as error:
                last_error = error
                if attempt + 1 < self.attempts:
                    time.sleep(min(2 ** attempt, 5))
        raise RuntimeError(f'model call failed after {self.attempts} attempts: {last_error}') from last_error
