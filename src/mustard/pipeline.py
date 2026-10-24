"""Keyframe captioning, normal-pattern induction, reasoning, and fusion."""
import base64
import json
import random
import re
import textwrap
from importlib.resources import files
from pathlib import Path

from . import prompts
from .keyframes import prepare, DEFAULTS as SELECTION_DEFAULTS
from .judging import explanation
from .model import nonempty
from .scoring import parse_score, strip_scores
from .storage import digest_file, digest_json, write_json

DEFAULT_SETTINGS = {
    'caption_model': 'qwen2.5vl:7b', 'reasoning_model': 'gemma3:12b',
    'n_reference': 30, 'seed': 2024,
    'mode': 'mustard', 'modalities': ['raw', 'trajectory', 'flow'],
    'options': {'caption': {'num_predict': 1024},
                'plot': {'temperature': 0.2, 'num_ctx': 8192},
                'pattern': {'num_ctx': 65536}, 'reasoning': {'num_ctx': 65536},
                'fusion': {'num_ctx': 8192, 'num_predict': 1024},
                'baseline': {'num_ctx': 65536}},
    'keyframes': dict(SELECTION_DEFAULTS),
}


def settings(overrides=None):
    result = json.loads(json.dumps(DEFAULT_SETTINGS))
    for key, value in (overrides or {}).items():
        if key not in result:
            raise ValueError(f'unknown pipeline setting: {key}')
        if key == 'options':
            for stage, options in value.items():
                if stage not in result['options']:
                    raise ValueError(f'unknown model options stage: {stage}')
                result['options'][stage].update(options)
        elif key == 'keyframes':
            result[key].update(value)
        else:
            result[key] = value
    if not isinstance(result['n_reference'], int) or result['n_reference'] < 1:
        raise ValueError('n_reference must be a positive integer')
    if result['keyframes'].get('frame_num', 8) != 8:
        raise ValueError('paper prompts require eight keyframes')
    modes = [('raw',), ('trajectory',), ('raw', 'flow'), ('trajectory', 'flow'), ('raw', 'trajectory', 'flow')]
    if not isinstance(result['modalities'], list) or len(set(result['modalities'])) != len(result['modalities']):
        raise ValueError('modalities must be a list without duplicates')
    if set(result['modalities']) - {'raw', 'trajectory', 'flow'}:
        raise ValueError('unknown modality')
    result['modalities'] = [name for name in ('raw', 'trajectory', 'flow') if name in result['modalities']]
    if tuple(result['modalities']) not in modes:
        raise ValueError('unsupported paper modality combination')
    if result['mode'] not in ('mustard', 'qwen-video'):
        raise ValueError('mode must be mustard or qwen-video')
    if result['mode'] == 'qwen-video' and result['modalities'] != ['raw']:
        raise ValueError('direct Qwen baseline accepts raw frames only')
    return result


def frame_pattern(number):
    def validate(text):
        nonempty(text)
        if not text.strip().startswith(f'<Frame {number}>:'):
            raise ValueError(f'pattern must start with <Frame {number}>:')
    return validate


def flow_pattern(text):
    nonempty(text)
    if not text.lstrip().startswith('max:'):
        raise ValueError('flow pattern must start with max:')


def fusion_response(text):
    parse_score(text)
    explanation(text, 'reasoning')
    explanation(text, 'final')


def plot_messages(path):
    system = files('mustard').joinpath('prompt_assets/plot-system-prompt.txt').read_text()
    user = files('mustard').joinpath('prompt_assets/plot-user-prompt.txt').read_text()
    image = base64.b64encode(Path(path).read_bytes()).decode()
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user, 'images': [image]}]


class Pipeline:
    def __init__(self, client, work, config=None):
        self.client = client
        self.work = Path(work)
        self.config = settings(config)
        self.fingerprints = {}

    def fingerprint(self, path):
        # The input files are assumed immutable for the duration of one command.
        path = str(path)
        if path not in self.fingerprints:
            self.fingerprints[path] = digest_file(path)
        return self.fingerprints[path]

    def chat(self, kind, messages, validator=nonempty):
        model = self.config['caption_model'] if kind in ('caption', 'plot', 'baseline') else self.config['reasoning_model']
        return self.client.chat(model, messages, self.config['options'][kind], validate=validator)

    def prepared_frames(self, sample):
        signature = {key: self.fingerprint(sample[key]) for key in ('raw_video', 'trajectory_video', 'tracks')}
        implementation = files('mustard').joinpath('_selection.py').read_bytes() + files('mustard').joinpath('keyframes.py').read_bytes()
        import hashlib
        key = digest_json(dict(inputs=signature, selection=self.config['keyframes'],
                               implementation=hashlib.sha256(implementation).hexdigest()))
        folder = self.work / 'frames' / key
        if folder.exists():
            manifest = json.loads((folder / 'frames.json').read_text())
            if any(not (folder / name).is_file() for kind in ('raw', 'trajectory') for name in manifest[kind]):
                raise ValueError(f'incomplete prepared frames: {folder}')
        else:
            manifest = prepare(sample['raw_video'], sample['trajectory_video'], sample['tracks'], folder,
                               self.config['keyframes'])
        return folder, manifest

    def captions(self, sample, raw=True, kinds=None):
        folder, manifest = self.prepared_frames(sample)
        result = {}
        kinds = kinds or (('raw', 'trajectory') if raw else ('trajectory',))
        for kind in kinds:
            prompt = prompts.static_caption_prompt(sample['object']) if kind == 'raw' else prompts.dynamic_caption_prompt(sample['object'])
            captions = []
            for name in manifest[kind]:
                image = base64.b64encode((folder / name).read_bytes()).decode()
                captions.append(self.chat('caption', [{'role': 'user', 'content': prompt, 'images': [image]}]))
            result[kind] = captions
        return result

    def plot_caption(self, sample):
        return self.chat('plot', plot_messages(sample['plot']))

    @staticmethod
    def join_captions(captions):
        return ''.join(f'<Frame {i + 1}>: {text.replace(chr(10), " ")}\n' for i, text in enumerate(captions))

    def train(self, records):
        if self.config['mode'] != 'mustard':
            raise ValueError('direct Qwen inference does not use trained patterns')
        groups = {}
        for sample in records:
            if sample['split'] == 'train':
                if sample['label'] != 0:
                    raise ValueError('normal pattern extraction accepts normal training samples only')
                groups.setdefault(sample['object'], []).append(sample)
        if not groups:
            raise ValueError('manifest has no training samples')
        modalities = self.config['modalities']
        single_raw = modalities == ['raw']
        bundle = dict(schema_version=1, config=self.config, model_digests=dict(self.client.identities), objects={})
        for obj, samples in sorted(groups.items()):
            samples = sorted(samples, key=lambda row: row['video'])
            n = self.config['n_reference']
            if len(samples) < n:
                raise ValueError(f'{obj}: requires {n} normal references, found {len(samples)}')
            # Per-object seed makes object subsets and ablations share references.
            rng = random.Random(digest_json(dict(seed=self.config['seed'], object=obj)))
            selected = sorted(rng.sample(samples, n), key=lambda row: row['video'])
            frame_sets, flows, references = [], [], []
            kind = 'raw' if single_raw else 'trajectory'
            for sample in selected:
                print(f'train {obj}/{sample["video"]}', flush=True)
                if single_raw or 'trajectory' in modalities:
                    frame_sets.append(self.captions(sample, kinds=(kind,))[kind])
                if 'flow' in modalities:
                    flows.append(self.plot_caption(sample))
                references.append(dict(video=sample['video'], input_hashes={
                    key: self.fingerprint(sample[key]) for key in ('raw_video', 'trajectory_video', 'tracks', 'plot')}))
            patterns = dict(references=references)
            if frame_sets:
                lines = []
                dtype = 'static' if single_raw else 'dynamic'
                prompt_fn = prompts._get_static_prompt if single_raw else prompts._get_dynamic_prompt
                for index in range(8):
                    system, task = prompt_fn(index + 1)
                    descriptions = '\n'.join(f'Description {i + 1}: {items[index]}' for i, items in enumerate(frame_sets))
                    user = task + '\n\n' + f'Here are all the Frame {index + 1} {dtype.upper()} descriptions:\n\n{descriptions}\n\n' + f'Output exactly one line starting with <Frame {index + 1}>:'
                    lines.append(self.chat('pattern', [{'role': 'system', 'content': system},
                                                       {'role': 'user', 'content': user}], frame_pattern(index + 1)))
                patterns[dtype] = '\n'.join(lines)
            if flows:
                system, task = prompts._get_optical_flow_prompt()
                content = '\n\n'.join(f'Data {i + 1}:\n{textwrap.indent(text, "  ")}' for i, text in enumerate(flows))
                user = task + '\n\n' + f'Here are the optical flow data summaries (sampled from {len(flows)} total):\n\n' + content + '\n\n' + "Remember: Output the 13-field pattern directly, starting with 'max:'"
                patterns['optical_flow'] = self.chat('pattern', [{'role': 'system', 'content': system},
                                                               {'role': 'user', 'content': user}], flow_pattern)
            bundle['objects'][obj] = patterns
        return bundle

    def reason(self, description, pattern, dtype):
        prompt_fn = {'static': prompts._get_static_anomaly_prompt,
                     'dynamic': prompts._get_dynamic_anomaly_prompt,
                     'optical_flow': prompts._get_optical_flow_anomaly_prompt}[dtype]
        system, task = prompt_fn()
        title = dtype.upper().replace('_', ' ')
        user = task + '\n\n' + f'Here is the {title} frame description set:\n\n' + textwrap.indent(description, '  ') + '\n\nAnd here is the pattern:\n\n' + textwrap.indent(pattern, '  ') + '\n\nRemember to provide the anomaly score in the specified format.'
        def validate(text):
            parse_score(text)
            nonempty(strip_scores(text))
        return self.chat('reasoning', [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], validate)

    def infer(self, records, bundle):
        results = []
        if self.config['mode'] != 'mustard':
            return self.qwen(records)
        if bundle.get('schema_version') != 1:
            raise ValueError('unsupported pattern artifact schema')
        for key in ('caption_model', 'reasoning_model', 'keyframes'):
            if bundle['config'][key] != self.config[key]:
                raise ValueError(f'pattern artifact and inference configuration differ: {key}')
        for stage in ('caption', 'plot'):
            if bundle['config']['options'][stage] != self.config['options'][stage]:
                raise ValueError(f'pattern artifact and inference options differ: {stage}')
        if bundle['model_digests'] != self.client.identities:
            raise ValueError('model digests differ from pattern extraction; use matching models')
        test = [sample for sample in records if sample['split'] == 'test']
        if not test:
            raise ValueError('manifest has no test samples')
        modalities = self.config['modalities']
        for sample in test:
            obj = sample['object']
            if obj not in bundle['objects']:
                raise ValueError(f'no normal patterns for {obj}')
            print(f'infer {obj}/{sample.get("anomaly")}/{sample["video"]}', flush=True)
            pattern = bundle['objects'][obj]
            required = {'static'} if modalities == ['raw'] else set()
            if 'trajectory' in modalities:
                required.add('dynamic')
            if 'flow' in modalities:
                required.add('optical_flow')
            if not required.issubset(pattern):
                raise ValueError(f'pattern artifact lacks modalities for {obj}: {sorted(required - set(pattern))}')
            kinds = tuple(kind for kind in ('raw', 'trajectory') if kind in modalities)
            captions = self.captions(sample, kinds=kinds) if kinds else {}
            raw = self.join_captions(captions['raw']) if 'raw' in captions else ''
            dynamic = self.reason(self.join_captions(captions['trajectory']), pattern['dynamic'], 'dynamic') if 'trajectory' in modalities else ''
            flow_description = self.plot_caption(sample) if 'flow' in modalities else ''
            flow = self.reason(flow_description, pattern['optical_flow'], 'optical_flow') if 'flow' in modalities else ''
            extra = {}
            if len(modalities) == 1:
                response = self.reason(raw, pattern['static'], 'static') if modalities == ['raw'] else dynamic
                text = strip_scores(response).removeprefix('Reasoning:').strip()
                nonempty(text)
                extra = dict(explanation=text, explanation_source='single-modality pattern reasoning')
            else:
                sections = []
                if dynamic:
                    sections.append(("Dynamic frames' reasoning (motion-annotated descriptions)", strip_scores(dynamic)))
                if flow:
                    sections.append(('Motion magnitude reasoning (optical flow plot)', strip_scores(flow)))
                if modalities == ['raw', 'trajectory', 'flow']:
                    # Preserve the recovered full-model prompt exactly.
                    user = prompts.build_user_prompt(mod_sections=sections, modality_list=['dynamic', 'oflow'], description_text=raw)
                else:
                    # Ablations explicitly identify the available inputs; omitted
                    # sources must not be described as present in the prompt.
                    blocks = [('Raw Frame Description', raw)] if raw else []
                    blocks.extend(sections)
                    user = 'Use only the following provided sources. An omitted source is unavailable.\n\n'
                    user += '\n\n'.join(f'{name}:\n{text}' for name, text in blocks)
                    user += '\n\nProvide Reasoning, one Anomaly Score, and Final Explanation in the required format.'
                response = self.chat('fusion', [{'role': 'system', 'content': prompts.build_system_prompt()},
                                                {'role': 'user', 'content': user}], fusion_response)
            results.append(dict(object=obj, video=sample['video'], anomaly=sample.get('anomaly'), label=sample['label'],
                                score=parse_score(response), response=response, dynamic_reasoning=dynamic,
                                optical_flow_reasoning=flow, raw_description=raw, flow_description=flow_description, **extra))
        return dict(schema_version=1, config=self.config, model_digests=dict(self.client.identities),
                    patterns_sha256=digest_json(bundle), predictions=results)

    def qwen(self, records):
        def validate(text):
            parse_score(text)
            if not re.search(r'^Reasoning:\s*\S', text, re.M | re.I):
                raise ValueError('direct Qwen output is missing its explanation')
        results = []
        for sample in (sample for sample in records if sample['split'] == 'test'):
            print(f'qwen {sample["object"]}/{sample.get("anomaly")}/{sample["video"]}', flush=True)
            folder, manifest = self.prepared_frames(sample)
            images = [base64.b64encode((folder / name).read_bytes()).decode() for name in manifest['raw']]
            response = self.chat('baseline', [{'role': 'user', 'content': prompts.qwen_question(8), 'images': images}], validate)
            text = re.split(r'^Reasoning:\s*', response, maxsplit=1, flags=re.M | re.I)[1].strip()
            results.append(dict(object=sample['object'], video=sample['video'], anomaly=sample.get('anomaly'),
                                label=sample['label'], score=parse_score(response), response=response,
                                explanation=text, explanation_source='direct Qwen Reasoning', raw_description=''))
        if not results:
            raise ValueError('manifest has no test samples')
        return dict(schema_version=1, config=self.config, model_digests=dict(self.client.identities), predictions=results)
