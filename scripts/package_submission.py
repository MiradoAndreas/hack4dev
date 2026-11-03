"""Build a competition handoff archive from an explicit set of project files."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def package(include_models=False, output=None):
    target = Path(output) if output else ROOT / 'dist/competition_submission.zip'
    target.parent.mkdir(parents=True, exist_ok=True)
    files = [ROOT / name for name in ['README.md', 'pyproject.toml', 'uv.lock', '.python-version']]
    for folder in ['docs', 'notebooks', 'scripts']:
        files.extend(p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.md', '.csv', '.png', '.ipynb', '.py'})
    model_manifest = {}
    if include_models:
        for name in ['madrid_source_model.joblib', 'amsterdam_fold0_n25_rep0.joblib', 'experiment.json']:
            p = ROOT / 'data/baseline' / name
            if not p.is_file():
                raise FileNotFoundError(f'Required baseline artifact is missing: {name}')
            files.append(p)
            with p.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            model_manifest[str(p.relative_to(ROOT))] = {'sha256': digest, 'bytes': p.stat().st_size}
    with ZipFile(target, 'w', compression=ZIP_DEFLATED) as archive:
        for p in sorted(set(files)):
            archive.write(p, p.relative_to(ROOT).as_posix())
        if model_manifest:
            archive.writestr('MODEL_MANIFEST.json', json.dumps(model_manifest, indent=2) + '\n')
    print(f'Created {target} ({len(set(files))} files; models={include_models})')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-models', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    package(args.include_models, args.output)
