"""Installed command; imports heavy inference dependencies only when requested."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog='volleymole', description='完整回合五佳球／十佳球；可选事件理解双榜')
    parser.add_argument('command', choices=['web','run','match','templates','infer','models','assets','verify','review-replays','meme-audio','meme-director'])
    if len(sys.argv)==1 or sys.argv[1] in ('-h','--help'):
        parser.print_help()
        return
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    if args.command == 'web':
        from .web.server import main as command
    elif args.command == 'assets':
        from .assets import main as command
    elif args.command == 'templates':
        from .templates import main as command
    elif args.command == 'match':
        from .match_collection import main as command
    elif args.command == 'models':
        from .models import main as command
    elif args.command == 'infer':
        from .inference import main as command
    elif args.command == 'verify':
        from .run_match import verify_cli as command
    elif args.command == 'review-replays':
        from .replay_stage import main as command
    elif args.command == 'meme-audio':
        from .meme_audio import main as command
    elif args.command == 'meme-director':
        from .meme_stage import main as command
    else:
        from .run_match import main as command
    try:
        command(rest)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f'VolleyMole: {exc}', file=sys.stderr)
        raise SystemExit(1)
