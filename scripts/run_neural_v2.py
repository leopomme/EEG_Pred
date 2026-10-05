#!/usr/bin/env python3
"""Fixed raw-EEG mechanism comparisons without reopening Cross confirmation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from eeg_comp.data import load_cache
from eeg_comp.neural import NeuralConfig, seed_everything
from eeg_comp.neural_v2 import NeuralV2Config, RawEEGV2
from eeg_comp.validation import summarize
from scripts.run_neural import make_folds, predict, protocol_weights, runtime_provenance, weighted_protocol_loss


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def fit(x, y, meta, train, valid, config, args, device, seed, epochs, history_path):
    """Same optimizer/batching/stopping policy as the original fixed baseline."""
    if set(y[train]) != {0, 1}:
        raise ValueError('Source fold must contain both classes')
    if valid is not None and (np.intersect1d(train, valid).size or not np.isin(y[valid], [0, 1]).all()):
        raise ValueError('Invalid inner split')
    seed_everything(seed)
    model = RawEEGV2(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    data = TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y[train]).float())
    loader = DataLoader(data, batch_size=64, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed), pin_memory=device.type == 'cuda')
    weights = protocol_weights(meta)
    best_loss, best_epoch, stale, best_state = float('inf'), epochs, 0, None
    with history_path.open('w') as history:
        for epoch in range(1, epochs + 1):
            model.train()
            total = 0.
            for batch, target in loader:
                batch, target = batch.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(batch), target)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                total += float(loss.detach()) * len(batch)
            row = {'epoch': epoch, 'optimizer_steps': epoch * len(loader), 'train_loss': total / len(train)}
            if valid is not None:
                _, p = predict(model, x, valid, device, 64)
                val_loss = weighted_protocol_loss(y[valid], p, meta.iloc[valid].run.to_numpy(), weights)
                row.update(valid_loss=val_loss, valid_accuracy=float(np.mean((p >= .5) == y[valid])))
                if val_loss < best_loss - 1e-4:
                    best_loss, best_epoch, stale = val_loss, epoch, 0
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                else:
                    stale += 1
            history.write(json.dumps(row) + '\n')
            history.flush()
            if epoch == 1 or epoch % 10 == 0:
                print(json.dumps({'fit': history_path.stem, **row}), flush=True)
            if valid is not None and stale >= args.patience:
                break
    if valid is not None:
        if best_state is None:
            raise RuntimeError('No valid source checkpoint')
        model.load_state_dict(best_state)
    return model, best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--competition', required=True, choices=['cross_subject', 'within_subject'])
    parser.add_argument('--mode', required=True, choices=['phase_stable', 'phase_baseline', 'cue_baseline'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--folds', help='Cross development 0..4; Within 0,1')
    parser.add_argument('--subjects', help='Within-only restricted smoke run')
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--max-epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.threads, args.max_epochs, args.patience) < 1:
        parser.error('Positive threads, epochs and patience required')
    if args.output.exists():
        raise FileExistsError(args.output)
    allowed = set(range(5 if args.competition == 'cross_subject' else 2))
    fold_ids = [int(fold) for fold in args.folds.split(',')] if args.folds else sorted(allowed)
    if len(set(fold_ids)) != len(fold_ids) or not set(fold_ids) <= allowed:
        parser.error('Only development fold IDs are allowed; confirmation cannot be reopened')
    subjects = args.subjects.split(',') if args.subjects else None
    if subjects and args.competition != 'within_subject':
        parser.error('--subjects is Within-only')
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Request a PBS GPU, or explicitly select CPU for a smoke test')
    device = torch.device(args.device)
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    arrays, meta = load_cache(ROOT / 'artifacts', args.competition)
    task_samples = 500 if args.mode == 'cue_baseline' else 1200
    cue = int(arrays['cue_index'])
    lo, hi = cue - 500, cue + task_samples
    if not (arrays['valid_mask'][:, lo:hi] & arrays['phase_valid_mask'][:, lo:hi]).all():
        raise ValueError('Unsupported baseline/task samples; no padding or filling')
    x = np.ascontiguousarray(arrays['X'][:, :, lo:hi], dtype=np.float32)
    y = arrays['y'].copy()
    del arrays
    config = NeuralV2Config(mode=args.mode, task_samples=task_samples)
    folds = make_folds(meta, y, args.competition, args.seed, fold_ids, subjects)
    args.output.mkdir(parents=True)
    cache_dir = ROOT / 'artifacts' / args.competition
    source_paths = ['eeg_comp/neural.py', 'eeg_comp/neural_v2.py', 'eeg_comp/data.py',
                    'eeg_comp/validation.py', 'scripts/run_neural.py', 'scripts/run_neural_v2.py']
    provenance = {'competition': args.competition, 'category': 'end-to-end DL',
                  'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  'model': asdict(config), 'raw_crop_samples': [lo, hi],
                  'source_sha256': {p: digest(ROOT / p) for p in source_paths},
                  'cache_sha256': {p: digest(cache_dir / p) for p in ['epochs.npz', 'metadata.csv', 'audit.json']},
                  'runtime': runtime_provenance(), 'torch_version': str(torch.__version__),
                  'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                  'threshold': .5, 'external_data': False,
                  'confirmation_policy': 'S019/S020 absent from development; previously used confirmation not reopened',
                  'information_budget': 'raw EEG only; per-trial normalization; source inner stopping; own-person fitting for Within',
                  'optimizer': {'name': 'AdamW', 'lr': .001, 'weight_decay': .01, 'batch_size': 64},
                  'epoch_transfer': 'Same epoch count on outer source refit; record optimizer steps as a diagnostic'}
    (args.output / 'config.json').write_text(json.dumps(provenance, indent=2) + '\n')
    frames, records = [], []
    started = time.monotonic()
    for fold in folds:
        directory = args.output / fold.name
        directory.mkdir()
        split = {part: getattr(fold, part).tolist() for part in ['train', 'inner_train', 'inner_valid', 'query']}
        split.update(subject=fold.subject, seed=fold.seed, confirmation=fold.confirmation)
        if fold.confirmation:
            raise ValueError('Confirmation is forbidden in this development runner')
        (directory / 'split.json').write_text(json.dumps(split, indent=2) + '\n')
        _, epochs = fit(x, y, meta, fold.inner_train, fold.inner_valid, config, args, device,
                        fold.seed, args.max_epochs, directory / 'inner_history.jsonl')
        model, _ = fit(x, y, meta, fold.train, None, config, args, device,
                       fold.seed, epochs, directory / 'refit_history.jsonl')
        logits, p = predict(model, x, fold.query, device, 64)
        state_path = directory / 'model.pt'
        torch.save({'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'model_config': asdict(config), 'competition': args.competition,
                    'train_epoch_indices': fold.train.tolist(), 'epochs': epochs, 'seed': fold.seed,
                    'provenance': provenance}, state_path)
        saved = torch.load(state_path, map_location='cpu', weights_only=True)
        restored_config = dict(saved['model_config'])
        restored_config['phase_config'] = NeuralConfig(**restored_config['phase_config'])
        replay = RawEEGV2(NeuralV2Config(**restored_config)).to(device)
        replay.load_state_dict(saved['model_state_dict'], strict=True)
        replay_logits, replay_p = predict(replay, x, fold.query, device, 64)
        np.testing.assert_allclose(p, replay_p, atol=1e-6, rtol=0)
        np.testing.assert_allclose(logits, replay_logits, atol=1e-5, rtol=0)
        np.testing.assert_array_equal(p >= .5, replay_p >= .5)
        frame = meta.iloc[fold.query].copy()
        frame['p_move'], frame['logit_move'], frame['fold'] = p, logits, fold.name
        frame['prediction'] = np.where(p >= .5, 'move', 'rest')
        frame.to_csv(directory / 'oof.csv', index=False)
        frames.append(frame)
        records.append({'fold': fold.name, 'subject': fold.subject, 'selected_epochs': epochs,
                        'inner_optimizer_steps': epochs * ((len(fold.inner_train) + 63) // 64),
                        'refit_optimizer_steps': epochs * ((len(fold.train) + 63) // 64),
                        'checkpoint_sha256': digest(state_path), 'same_device_replay_exact': bool(np.array_equal(p, replay_p))})
        combined = pd.concat(frames).sort_values('epoch_index')
        if combined.epoch_index.duplicated().any():
            raise ValueError('Repeated outer query')
        combined.to_csv(args.output / 'oof.csv', index=False)
        report = {'metrics': summarize(combined, meta), 'folds': records,
                  'elapsed_seconds': time.monotonic() - started,
                  'complete_requested_validation': len(records) == len(folds)}
        (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'completed': fold.name, 'epochs': epochs,
                          'accuracy': float(np.mean((p >= .5) == y[fold.query]))}), flush=True)


if __name__ == '__main__':
    main()
