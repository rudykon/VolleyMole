"""Rasterize the user-owned SVG without changing its colors, paths or aspect ratio.

Run with the system Python (python3-gi, gir1.2-rsvg-2.0).
"""
import argparse
import gi

gi.require_version('Rsvg', '2.0')
from gi.repository import Rsvg


def render(source, output):
    pixels = Rsvg.Handle.new_from_file(source).get_pixbuf()
    if pixels is None:
        raise ValueError('SVG 标志无法栅格化')
    pixels.savev(output, 'png', [], [])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('output')
    args = parser.parse_args()
    render(args.source, args.output)
