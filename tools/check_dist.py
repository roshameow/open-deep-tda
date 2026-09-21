"""Inspect archives without extracting/executing them; fail on obvious leakage."""
import re
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

PRIVATE_NAMES={'RELEASE_WORKPLAN.md','IMPLEMENTATION_CONTRACT.md','PUBLICATION_AUDIT.md'}
BAD_PARTS={'data','outputs','build','docs','__pycache__','.pytest_cache','.git'}
BAD_SUFFIXES={'.pt','.pth','.pkl','.npy','.npz','.pyc','.pyo','.o','.a'}
TEXT_SUFFIXES={'.py','.md','.json','.txt','.toml','.yml','.yaml','.cpp','.hpp','.h','.cfg','.in'}
PATTERNS=[re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,})'),
          re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
          re.compile(rb'/Users/[A-Za-z0-9][A-Za-z0-9._-]*/')]


def check(path):
    wheel=path.suffix=='.whl'
    if wheel:
        archive=zipfile.ZipFile(path)
        items=[(x.filename,lambda x=x:archive.read(x)) for x in archive.infolist() if not x.is_dir()]
    else:
        archive=tarfile.open(path,'r:gz')
        items=[]
        for entry in archive.getmembers():
            if entry.issym() or entry.islnk():raise ValueError(f'{path}: links are not expected: {entry.name}')
            if entry.isfile():items.append((entry.name,lambda entry=entry:archive.extractfile(entry).read()))
    names=[PurePosixPath(name) for name,_ in items]
    for required in ['LICENSE','THIRD_PARTY_NOTICES.md']:
        if not any(n.name==required for n in names):raise ValueError(f'{path}: missing {required}')
    for (name,read),p in zip(items,names):
        if p.is_absolute() or '..' in p.parts or set(p.parts)&BAD_PARTS or p.name in PRIVATE_NAMES or p.suffix in BAD_SUFFIXES:
            raise ValueError(f'{path}: excluded payload {name}')
        if p.suffix in {'.so','.pyd','.dylib','.dll','.exe'} and not (wheel and p.name.startswith('_core.') and p.suffix in {'.so','.pyd'}):
            raise ValueError(f'{path}: unexpected binary {name}')
        if p.suffix=='.json' and p.parent.name=='research':raise ValueError(f'{path}: raw research JSON {name}')
        if p.suffix in TEXT_SUFFIXES or p.name in {'METADATA','PKG-INFO','LICENSE'}:
            text=read()
            if any(pattern.search(text) for pattern in PATTERNS):raise ValueError(f'{path}: possible identifying path/secret in {name} (content suppressed)')
    archive.close()
    print(f'{path.name}: {len(items)} files; required notices and payload-boundary scan passed')


def main():
    paths=[]
    for value in sys.argv[1:]:
        p=Path(value)
        paths.extend(sorted(p.glob('*.tar.gz'))+sorted(p.glob('*.whl')) if p.is_dir() else [p])
    if not paths:raise SystemExit('usage: python tools/check_dist.py dist/ OR archive...')
    for path in paths:check(path)


if __name__=='__main__':main()
