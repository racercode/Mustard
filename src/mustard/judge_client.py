"""Chat Completions judge transport; credentials stay in the environment."""
import json
import os
import time
import urllib.request
from pathlib import Path

from .model import nonempty
from .storage import digest_json, write_json


class Judge:
    def __init__(self, cache, model='gpt-4o-2024-08-06', base_url='https://api.openai.com/v1',
                 attempts=3, timeout=180, transport=None):
        self.cache = Path(cache)
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.attempts = attempts
        self.timeout = timeout
        self.transport = transport or self._http

    def _http(self, payload):
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError('set OPENAI_API_KEY to run judge requests')
        request = urllib.request.Request(self.base_url + '/chat/completions',
                                         data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def chat(self, messages, validate):
        payload = dict(model=self.model, messages=messages)
        key = digest_json(dict(base_url=self.base_url, request=payload))
        path = self.cache / f'{key}.json'
        if path.is_file():
            record = json.loads(path.read_text())
            validate(record['text'])
            return record['text']
        last_error = None
        for attempt in range(self.attempts):
            try:
                result = self.transport(payload)
                text = nonempty(result['choices'][0]['message']['content'])
                validate(text)
                write_json(path, dict(text=text, model=result.get('model', self.model),
                                     system_fingerprint=result.get('system_fingerprint'),
                                     usage=result.get('usage'), request_hash=key))
                return text
            except Exception as error:
                last_error = error
                if attempt + 1 < self.attempts:
                    time.sleep(min(2 ** attempt, 5))
        raise RuntimeError(f'judge call failed after {self.attempts} attempts: {last_error}') from last_error
