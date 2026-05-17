#!/usr/bin/env python3
"""Rewrite ONNX policies in place so the batch axis is statically 1.

Isaac Lab's ``play.py --export_policy`` exports policies with
``dynamic_axes={'obs': {0: 'batch'}, 'actions': {0: 'batch'}}``. The deploy-side
ORT wrapper in ``deploy/include/isaaclab/algorithms/algorithms.h`` reads the
declared input shape directly and passes it to ``Ort::Value::CreateTensor``,
which throws ``"tried creating tensor with negative value in shape"`` when it
encounters the resulting ``-1``. Until every deployed controller binary picks
up the defensive clamp added to that header, the rule on disk is: deploy ONNX
files must declare a static batch dimension of 1.

This script enforces that rule. Idempotent: already-static models are left
untouched (no backup, no rewrite).

Usage:
    python scripts/freeze_onnx_batch.py PATH [PATH ...]

PATH may be either an ``exported/policy.onnx`` file or a directory; in the
latter case all ``*.onnx`` files anywhere under it are processed. A ``.bak``
copy is saved next to each rewritten file the first time it is processed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import onnx


def _shape_has_dynamic_dim(tensor):
    for d in tensor.type.tensor_type.shape.dim:
        if d.HasField('dim_param') or d.dim_value <= 0:
            return True
    return False


def _freeze(path):
    """Rewrite ``path`` so all I/O tensors have positive static dims.

    Returns ``True`` if the file was modified, ``False`` if it was already
    static (or no I/O tensor needed updating).
    """
    model = onnx.load(str(path))
    needs_rewrite = any(
        _shape_has_dynamic_dim(t)
        for t in list(model.graph.input) + list(model.graph.output)
    )
    if not needs_rewrite:
        print(f'  [skip] already static: {path}')
        return False

    bak = path.with_suffix(path.suffix + '.bak')
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f'  backed up -> {bak.name}')

    for tensor in list(model.graph.input) + list(model.graph.output):
        for idx, dim in enumerate(tensor.type.tensor_type.shape.dim):
            if dim.HasField('dim_param') or dim.dim_value <= 0:
                old = dim.dim_param if dim.HasField('dim_param') else dim.dim_value
                dim.Clear()
                dim.dim_value = 1
                print(f'  {tensor.name}: dim[{idx}] {old!r} -> 1')

    onnx.save(model, str(path))
    onnx.checker.check_model(onnx.load(str(path)))
    print(f'  [ok] rewrote {path}')
    return True


def _iter_targets(paths):
    out = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.rglob('*.onnx')))
        elif p.suffix == '.onnx' and p.is_file():
            out.append(p)
        else:
            print(f'  [warn] skipping (not an .onnx file or directory): {p}', file=sys.stderr)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'paths',
        nargs='+',
        type=Path,
        help='ONNX file(s) or directory(ies) to walk recursively for *.onnx files.',
    )
    args = parser.parse_args()

    targets = _iter_targets(args.paths)
    if not targets:
        print('[freeze_onnx_batch] no .onnx files found.', file=sys.stderr)
        return 1

    modified = 0
    for target in targets:
        print(f'== {target} ==')
        if _freeze(target):
            modified += 1

    print(f'\n[freeze_onnx_batch] processed {len(targets)} file(s); rewrote {modified}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
