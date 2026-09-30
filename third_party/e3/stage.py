"""CPU-only, copy-only staging. No image build, loader rewrite or remote action."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

PACKAGE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_native(root):
    provenance = json.loads((PACKAGE / 'provenance.json').read_text())
    version = hashlib.sha256()
    for name, expected in provenance['native_loader_sha256'].items():
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Native packing/loader source changed: {name}')
        version.update(name.encode())
        version.update(data)
    saved = (root / 'load_accel/loader-version.txt').read_text().strip()
    if version.hexdigest() != saved or saved != provenance['loader_version']:
        raise ValueError('Native loader-version does not match actual hashed source')
    return saved


def stage(root, output):
    root, output = Path(root), Path(output)
    version = verify_native(root)
    provenance = json.loads((PACKAGE / 'provenance.json').read_text())
    for name, expected in provenance['copied_sha256'].items():
        if digest(PACKAGE / name) != expected:
            raise ValueError(f'E3 source/artifact mismatch: {name}')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite stage: {output}')
    output.mkdir(parents=True)
    shutil.copytree(PACKAGE, output / 'e3', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(PACKAGE / 'sitecustomize.py', output / 'sitecustomize.py')
    files = sorted(p for p in output.rglob('*') if p.is_file())
    manifest = dict(loader_version=version,
                    files={str(p.relative_to(output)): digest(p) for p in files})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def verify_stage(output):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(output / name) != expected:
            raise ValueError(f'Staged file mismatch: {name}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PACKAGE.parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    result = verify_stage(args.output) if args.verify else stage(args.root, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
