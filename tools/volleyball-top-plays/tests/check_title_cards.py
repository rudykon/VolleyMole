"""Historical check entry; implementation is in the installed package."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _compat import forward

forward('check_title_cards',__name__)
