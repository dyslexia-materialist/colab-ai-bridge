"""Build the self-contained Colab notebook from editable source files.

Usage: python tools/build_one_cell.py
No third-party Python packages required.
"""
from pathlib import Path
import json
import ast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dist'
OUT.mkdir(exist_ok=True)
html = (ROOT / 'web' / 'index.html').read_text(encoding='utf-8')
worker = (ROOT / 'src' / 'worker.template.py').read_text(encoding='utf-8')
launcher = (ROOT / 'src' / 'launcher.template.py').read_text(encoding='utf-8')

if "'''" in html:
    raise ValueError("HTML contains triple single quotes; switch the embedding delimiter")
if 'HTML_PAGE = __INLINE_HTML__' not in worker:
    raise ValueError('Worker is missing the HTML injection marker')
worker = worker.replace('HTML_PAGE = __INLINE_HTML__', "HTML_PAGE = r'''" + html + "'''")
if '"""' in worker:
    raise ValueError('Worker contains triple double quotes; switch the embedding delimiter')
if 'WORKER_CODE = __INLINE_WORKER__' not in launcher:
    raise ValueError('Launcher is missing the worker injection marker')
source = launcher.replace('WORKER_CODE = __INLINE_WORKER__', 'WORKER_CODE = r"""' + worker + '"""')
compile(source, '<Colab_AI_Bridge_v09_one_cell.py>', 'exec')
compile(worker, '<worker.py>', 'exec')

py_name = 'Colab_AI_Bridge_v09_one_cell.py'
ipynb_name = 'Colab_AI_Bridge_v09_one_cell.ipynb'
(OUT / py_name).write_text(source, encoding='utf-8')
notebook = {
  'cells': [
    {
      'cell_type': 'markdown',
      'metadata': {},
      'source': [
        '# Colab AI Bridge v0.9 — English Workspace\n',
        'Run the single code cell below with a **GPU** runtime. Set `NGROK_AUTHTOKEN`, `BRIDGE_PASSWORD` (12+ characters), and optionally `HF_TOKEN` in Colab Secrets.\n',
        'This notebook is built from `src/` and `web/` with `python tools/build_one_cell.py`.\n'
      ]
    },
    {'cell_type': 'code','execution_count': None,'metadata':{},'outputs':[],'source':source.splitlines(keepends=True)}
  ],
  'metadata': {'kernelspec': {'display_name': 'Python 3','language':'python','name':'python3'}, 'language_info': {'name':'python'}},
  'nbformat':4,'nbformat_minor':5
}
(OUT/ipynb_name).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print('Built:',OUT/py_name)
print('Built:',OUT/ipynb_name)
