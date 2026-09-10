"""Read-only GitHub preflight; reports locations, never matching secret values."""
import argparse
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIRS = {'data','models','runs','outputs','tools','.venv','venv','.agents','.codex','build','dist'}
LOCAL_NAMES = {'llm_api.json','github_token.json','.env'}
MEDIA = {'.pt','.pth','.onnx','.safetensors','.engine','.plan','.mp4','.mov','.avi','.mkv','.webm','.pem','.key'}
LIMIT = 50 * 1024 * 1024
SECRETS = {
    'GitHub token': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})'),
    'API token': re.compile(rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}'),
    'private key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'credential value': re.compile(rb'''(?i)["'](?:api_key|access_token|github_token)["']\s*:\s*["'][A-Za-z0-9_/-]{24,}["']'''),
}


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def credential_issues(label, data):
    return [f'{label}: possible {kind}' for kind,pattern in SECRETS.items() if pattern.search(data)]


def link_issues(path, data, published):
    text=data.decode('utf-8')
    targets=re.findall(r'\[[^\]\n]*\]\(([^)\s]+)\)',text)
    targets += re.findall(r'<img\b[^>]*\bsrc="([^"]+)"',text)
    issues=[]
    for target in targets:
        url=urlsplit(target.strip('<>'))
        if url.scheme or url.netloc or not url.path:
            continue
        dest=(path.parent/unquote(url.path)).resolve()
        if not dest.is_relative_to(ROOT) or not dest.exists():
            issues.append(f'{path.relative_to(ROOT)}: broken local link {target}')
        elif dest.is_file() and dest.relative_to(ROOT).as_posix() not in published:
            issues.append(f'{path.relative_to(ROOT)}: link targets unpublished file {target}')
    return issues


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history',action='store_true')
    args=parser.parse_args()
    names=set(git('ls-files','--cached','--others','--exclude-standard','-z').decode().strip('\0').split('\0'))
    published={n for n in names if n and (ROOT/n).is_file()}
    issues=[]
    largest=0
    for name in sorted(published):
        path=ROOT/name
        if (Path(name).parts[0] in LOCAL_DIRS or path.name in LOCAL_NAMES or
                (path.name.startswith('.env.') and path.name!='.env.example') or path.suffix.lower() in MEDIA):
            issues.append(f'{name}: local-only file is visible to Git')
            continue
        if path.is_symlink():
            issues.append(f'{name}: review symlink before publishing')
            continue
        size=path.stat().st_size
        largest=max(largest,size)
        if size>LIMIT:
            issues.append(f'{name}: file exceeds 50 MiB')
            continue
        data=path.read_bytes()
        issues += credential_issues(name,data)
        if path.suffix=='.md':
            issues += link_issues(path,data,published)
    history_count=0
    if args.history:
        for row in git('rev-list','--objects','--all').decode().splitlines():
            oid,_,name=row.partition(' ')
            if git('cat-file','-t',oid).strip()!=b'blob':
                continue
            history_count+=1
            label=f'history {oid[:12]} ({name})'
            size=int(git('cat-file','-s',oid))
            if size>LIMIT:
                issues.append(f'{label}: file exceeds 50 MiB')
                continue
            if Path(name).name in LOCAL_NAMES:
                issues.append(f'{label}: historical credential filename')
            issues += credential_issues(label,git('cat-file','blob',oid))
    print(f'Checked {len(published)} current files; largest {largest/1024/1024:.2f} MiB; {history_count} historical blobs.')
    for issue in issues:print('FAIL:',issue)
    if not (ROOT/'LICENSE').exists():
        print('NOTE: First-party project license has not been selected; see THIRD_PARTY.md.')
    print('Heuristic secret scan only; review the staged diff before publishing.')
    if issues:raise SystemExit(1)
    print('Preflight passed. No files staged, committed or pushed.')


if __name__=='__main__':main()
