"""Installed command; imports heavy inference dependencies only when requested."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog='volleymole', description='单场比赛 → 五佳球／十佳球')
    parser.add_argument('command', choices=['run','infer','models','verify'])
    if len(sys.argv)==1 or sys.argv[1] in ('-h','--help'):
        parser.print_help()
        return
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    if args.command == 'models':
        from .models import main as command
    elif args.command == 'infer':
        from .inference import main as command
    elif args.command == 'verify':
        from .run_match import verify_cli as command
    else:
        from .run_match import main as command
    try:
        command(rest)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f'VolleyMole: {exc}', file=sys.stderr)
        raise SystemExit(1)
