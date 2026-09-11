"""Development-only: extract SC faces, preserving all glyphs and font metadata."""
import argparse
from pathlib import Path


def main():
    from fontTools.ttLib import TTCollection
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--family',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('输出已存在，不覆盖字体')
    collection=TTCollection(args.source)
    font=next(f for f in collection.fonts if f['name'].getDebugName(1)==args.family)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    font.save(args.output)
    print(args.output)


if __name__=='__main__':main()
