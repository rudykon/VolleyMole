"""Installed command; imports heavy inference dependencies only when requested."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog='volleymole', description='完整回合五佳球／十佳球；可选事件理解双榜')
    parser.add_argument('command', choices=['run','match','infer','models','assets','verify'])
    if len(sys.argv)==1 or sys.argv[1] in ('-h','--help'):
        parser.print_help()
        return
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    if args.command == 'assets':
        from .assets import main as command
    elif args.command == 'match':
        from .match_collection import main as command
    elif args.command == 'models':
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
