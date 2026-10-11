"""Assemble a new local release; never upload or overwrite an existing directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new release directory.')
    names = ['model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json']
    for name in names:
        if not (args.checkpoint/name).is_file():
            parser.error(f'Missing required file: {name}')
    expected = 'af9c2c25534e5e63980eb13a1e2f09811a71264a96c381152000bdfcd71088a5'
    if hashlib.sha256((args.checkpoint/'model.safetensors').read_bytes()).hexdigest() != expected:
        parser.error('These model-card results apply only to the evaluated Typed specialist; weight hash mismatch.')
    manifest = json.loads(args.run_manifest.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='arbiter-release-', dir=args.output.parent))
    try:
        for name in names + ['chat_template.jinja']:
            if (args.checkpoint/name).is_file():
                shutil.copy2(args.checkpoint/name, staging/name)
        shutil.copy2(ROOT/'docs/model_card_typed.md', staging/'README.md')
        record = {'training_manifest': manifest, 'files': {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in staging.iterdir()}}
        (staging/'reproduction.json').write_text(json.dumps(record, indent=2)+'\n')
        # Load the assembled package independently, with no remote fallback.
        from arbiter import ArbiterModel
        model = ArbiterModel.from_pretrained(staging, device='cpu', local_files_only=True)
        model.decide(context='A duplicate payment was reported.', instruction='Select the relevant team.',
                     criteria={'payments':'Payment issues', 'access':'Login issues'})
        if args.output.exists():
            raise FileExistsError(args.output)
        staging.rename(args.output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(args.output)


if __name__ == '__main__':
    main()
