#!/usr/bin/env python3
"""Normalize Isaac Lab-exported ``params/deploy.yaml`` files for yaml-cpp.

Some Isaac Lab export paths emit the Python repr ``None`` for fields that are
``Python None`` (notably ``actions.JointPositionAction.joint_ids``). yaml-cpp
parses ``None`` as a string scalar, not as a YAML null, so the deploy-side
parser then throws ``YAML::TypedBadConversion`` when it tries to coerce that
string into ``std::vector<int>``:

    terminate called after throwing an instance of
    'YAML::TypedBadConversion<std::vector<int, std::allocator<int> > >'
      what():  yaml-cpp: error at line 31, column 16: bad conversion

This helper rewrites the value to the lowercase YAML null literal ``null``,
which both PyYAML and yaml-cpp parse as null. Idempotent: already-correct files
are reported as ``[ok]`` and left untouched.

Usage:
    python scripts/sanitize_deploy_yaml.py PATH [PATH ...]

PATH may be a ``deploy.yaml`` file or a directory; in the latter case all
``deploy.yaml`` files anywhere under it are processed. A ``.bak.None_to_null``
copy is saved next to each rewritten file the first time it is processed.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

PATTERN = re.compile(r'^(\s*joint_ids:\s+)None\s*$', re.MULTILINE)


def _sanitize(path):
    """Rewrite ``path`` if it contains ``joint_ids: None``.

    Returns ``True`` if the file was modified, ``False`` if it was already
    correct.
    """
    text = path.read_text()
    if not PATTERN.search(text):
        return False

    bak = path.with_suffix(path.suffix + '.bak.None_to_null')
    if not bak.exists():
        shutil.copy2(path, bak)

    path.write_text(PATTERN.sub(r'\1null', text))
    return True


def _iter_targets(paths):
    out = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.rglob('deploy.yaml')))
        elif p.name == 'deploy.yaml' and p.is_file():
            out.append(p)
        else:
            print(f'  [warn] skipping (not a deploy.yaml or directory): {p}', file=sys.stderr)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'paths',
        nargs='+',
        type=Path,
        help='deploy.yaml file(s) or directory(ies) to walk recursively.',
    )
    args = parser.parse_args()

    targets = _iter_targets(args.paths)
    if not targets:
        print('[sanitize_deploy_yaml] no deploy.yaml files found.', file=sys.stderr)
        return 1

    modified = 0
    for target in targets:
        if _sanitize(target):
            print(f'  [fixed] {target}')
            modified += 1
        else:
            print(f'  [ok]    {target}')

    print(f'\n[sanitize_deploy_yaml] processed {len(targets)} file(s); rewrote {modified}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
