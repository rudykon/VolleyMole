"""Deprecated source-tree entry points; implementation lives only in volleymole."""
import importlib
import importlib.util
import os
from pathlib import Path
import runpy
import sys


def forward(module, caller):
    root = Path(__file__).resolve().parents[2]
    python = root/'.venv/bin/python'
    if caller=='__main__' and module=='rasterize_logo':
        # librsvg introspection is a system dependency, not part of the ML venv.
        os.execv('/usr/bin/python3',['/usr/bin/python3',str(root/'src/volleymole/rasterize_logo.py'),*sys.argv[1:]])
    if caller=='__main__' and python.is_file() and Path(sys.prefix).resolve()!=python.parent.parent.resolve():
        os.execv(str(python),[str(python),'-m','volleymole.'+module,*sys.argv[1:]])
    if importlib.util.find_spec('volleymole') is None:
        # Source-tree compatibility only. Installed wheels never import this file.
        sys.path.insert(0,str(root/'src'))
    if caller=='__main__':
        runpy.run_module('volleymole.'+module,run_name='__main__')
    else:
        # Alias the actual module, so historical unittest.mock targets still work.
        sys.modules[caller] = importlib.import_module('volleymole.'+module)
