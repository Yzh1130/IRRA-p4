"""Execute a hash-verified script with single-class guards for three unused heads.

The frozen source is neither changed nor copied. Every other fit goes through
scikit-learn unchanged, including single-class fits outside the allowlist.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import linecache
import re
import runpy
import sys
import numpy as np
import sklearn.linear_model as linear_model

ORIGINAL_LR = linear_model.LogisticRegression
GUARDED_HEADS = ('anyh', 'srci', 'gateC')
FORBIDDEN_METHODS = frozenset(('A-ANY-post', 'A-SRCind-post', 'B-sup',
                             'B-sup-crossonly', 'B-joint', 'B-joint-nopen'))
ASSIGNMENT = re.compile(r'^(anyh|srci|gateC)\s*=\s*LogisticRegression\s*\(')


def guarded_class(script, events):
    """Bind a GuardedLR subclass to exactly one caller file and event list."""
    target = Path(script).resolve()

    class GuardedLR(ORIGINAL_LR):
        def fit(self, X, y, *args, **kwargs):
            labels = np.unique(y)
            if len(labels) != 1:
                return super().fit(X, y, *args, **kwargs)
            frame = sys._getframe(1)
            filename, lineno = frame.f_code.co_filename, frame.f_lineno
            del frame
            match = ASSIGNMENT.match(linecache.getline(filename, lineno).lstrip())
            if Path(filename).resolve() == target and match:
                label = bool(labels[0])
                self.classes_ = np.array([False, True])
                self.coef_ = np.zeros((1, X.shape[1]), dtype=np.float64)
                self.intercept_ = np.array([20.0 if label else -20.0])
                self.n_features_in_ = X.shape[1]
                self._r18_constant_label = label
                events.append({'head': match.group(1), 'file': filename,
                               'line': lineno, 'samples': len(y),
                               'label': labels[0].item()})
                return self
            print('[run_frozen] unguarded single-class fit: '
                  f'{filename}:{lineno}, samples={len(y)}, label={labels[0]!r}',
                  file=sys.stderr, flush=True)
            return super().fit(X, y, *args, **kwargs)

        def predict_proba(self, X):
            if hasattr(self, '_r18_constant_label'):
                p = float(self._r18_constant_label)
                return np.tile([1.0 - p, p], (X.shape[0], 1))
            return super().predict_proba(X)

        def predict(self, X):
            if hasattr(self, '_r18_constant_label'):
                return np.full(X.shape[0], self._r18_constant_label, dtype=bool)
            return super().predict(X)

    return GuardedLR


def method_names(arguments):
    names = []
    for index, value in enumerate(arguments):
        if value == '--methods':
            for method in arguments[index + 1:]:
                if method.startswith('--'):
                    break
                names.append(method)
        elif value.startswith('--methods='):
            names.append(value.split('=', 1)[1])
    return names


def run(script, arguments, expected_sha, log):
    events = []
    digest = None
    try:
        digest = hashlib.sha256(Path(script).read_bytes()).hexdigest()
        if not re.fullmatch(r'[0-9a-fA-F]{64}', expected_sha) or digest != expected_sha.lower():
            print('[run_frozen] script SHA256 mismatch', file=sys.stderr)
            raise SystemExit(2)
        forbidden = FORBIDDEN_METHODS.intersection(method_names(arguments))
        if forbidden:
            print('[run_frozen] forbidden methods: ' + ', '.join(sorted(forbidden)), file=sys.stderr)
            raise SystemExit(2)
        original_class, original_argv = linear_model.LogisticRegression, sys.argv
        linecache.checkcache(str(script))
        try:
            linear_model.LogisticRegression = guarded_class(script, events)
            sys.argv = [str(script), *arguments]
            runpy.run_path(str(script), run_name='__main__')
        finally:
            linear_model.LogisticRegression, sys.argv = original_class, original_argv
    finally:
        path = Path(log)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'script': str(script), 'script_sha256': digest,
                                    'guarded_heads': list(GUARDED_HEADS), 'events': events},
                                   ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8', newline='\n')
        counts = Counter(e['head'] for e in events)
        detail = ', '.join(f'{h}: {n}' for h, n in sorted(counts.items())) or 'none'
        print(f'[run_frozen] degenerate fits: {len(events)} ({detail})', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expect_sha', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    cli = parser.parse_args(argv)
    command = cli.command[1:] if cli.command and cli.command[0] == '--' else cli.command
    if not command:
        parser.error('a script is required after --')
    return run(command[0], command[1:], cli.expect_sha, cli.log)


if __name__ == '__main__':
    raise SystemExit(main())
