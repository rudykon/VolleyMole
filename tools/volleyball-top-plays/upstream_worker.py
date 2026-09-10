"""Deprecated worker spelling; no upstream imports or external environments."""
import sys
from _compat import forward

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1] in ('analytics','tracking','player','shared'):
        sys.argv[1:2] = ['--kind',sys.argv[1]]
    forward('inference',__name__)
else:
    forward('jersey',__name__)
