# COLAB AI BRIDGE v0.9 - TEK HUCRE KURULUM + WEB + REST API
# Google Colab (Python 3.13 / GPU). Tekrar calistirildiginda ayni ortamda tekrardan kurulum yapmayi minimize eder.
# Colab Secrets: NGROK_AUTHTOKEN, BRIDGE_PASSWORD (en az 12 karakter), HF_TOKEN (opsiyonel)
import os, sys, subprocess, pathlib, shutil, time, json, signal, importlib, textwrap
from getpass import getpass

print('Python:', sys.version.split()[0])

# 1) Colab'in torch/CUDA paketlerine dokunmadan gerekli kutuphaneleri kur.
REQ = [
    'fastapi==0.135.1',
    'uvicorn>=0.30,<1',
    'pyngrok>=7.2,<8',
    'transformers==5.22.2',
    'diffusers==0.42.1',
    'accelerate==1.15.0',
    'huggingface-hub>=1.32,<2',
    'safetensors>=0.5',
    'sentencepiece>=0.2.1',
    'requests>=2.31,<3',
    'python-multipart>=0.0.20,<1',
]

def pip_or_error(arguments, title):
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', *arguments],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode:
        print('\nPIP ERROR OUTPUT:\n'+r.stdout[-12000:])
        raise RuntimeError(title + ' failed')

print('1/4 - Checking dependencies…')
pip_or_error(['--prefer-binary', *REQ], 'Dependency installation')

# 2) Pillow: Colab sistem paketleriyle karismasin diye AYRI klasore kurulur.
# Python prosesi yeniden baslatilacagi icin cached PIL modul karismasi yasamayiz.
ROOT = pathlib.Path('/content/colab_ai_bridge_v09')
ROOT.mkdir(parents=True, exist_ok=True)
PIL_DIR = ROOT / 'isolated_pillow'
WORKER = ROOT / 'worker.py'
LOG = ROOT / 'worker.log'
URL_FILE = ROOT / 'public_url.txt'
PID_FILE = ROOT / 'worker.pid'

def pillow_probe():
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PIL_DIR) + os.pathsep + env.get('PYTHONPATH','')
    sample = ("import pathlib, PIL\n"
              "from PIL import Image, ImageText, ImageFont, ImageOps, ImageDraw, ImageFilter\n"
              "from PIL._typing import _Ink\n"
              "assert pathlib.Path(PIL.__file__).resolve().is_relative_to(pathlib.Path(%r).resolve())\n"
              "print('Pillow:',PIL.__version__,'|', PIL.__file__)") % str(PIL_DIR)
    return subprocess.run([sys.executable, '-c', sample], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

print('2/4 - Verifying isolated Pillow…')
probe = pillow_probe()
if probe.returncode:
    if PIL_DIR.exists():
        shutil.rmtree(PIL_DIR)
    PIL_DIR.mkdir(parents=True, exist_ok=True)
    pip_or_error(['--no-cache-dir', '--no-deps', '--only-binary=:all:', '--target', str(PIL_DIR), 'Pillow==12.3.0'],
                 'Isolated Pillow installation')
    probe = pillow_probe()
if probe.returncode:
    print(probe.stdout[-3000:])
    raise RuntimeError('Pillow installation failed; share the log above.')
print(probe.stdout.strip())

# 3) Colab Secrets - sifreleri kod icine veya loglara yazma.
try:
    from google.colab import userdata
    def secret(name):
        try: return (userdata.get(name) or '').strip()
        except Exception: return ''
except ImportError:
    def secret(name): return ''

NGROK_TOKEN = secret('NGROK_AUTHTOKEN') or getpass('ngrok authtoken: ').strip()
API_PASSWORD = secret('BRIDGE_PASSWORD') or getpass('Web/API password (12+ characters): ').strip()
HF_TOKEN = secret('HF_TOKEN')
if len(API_PASSWORD) < 12 or not NGROK_TOKEN:
    raise ValueError('Missing NGROK_AUTHTOKEN or BRIDGE_PASSWORD (12+ characters).')

# 4) v0.5.x veya v0.6 ayni Colab VM uzerinde hala aciksa eski worker'i
# guvenli sekilde durdur; yoksa ayni ngrok adresi ERR_NGROK_334 hatasi verir.
for previous_root in [pathlib.Path('/content/colab_ai_bridge_v05'), pathlib.Path('/content/colab_ai_bridge_v06'), pathlib.Path('/content/colab_ai_bridge_v07'), pathlib.Path('/content/colab_ai_bridge_v08'), ROOT]:
    previous_worker = previous_root / 'worker.py'
    previous_pid_file = previous_root / 'worker.pid'
    if not previous_pid_file.exists():
        continue
    try:
        old_pid = int(previous_pid_file.read_text().strip())
        cmdline = pathlib.Path(f'/proc/{old_pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='ignore')
        if str(previous_worker) in cmdline:
            print('Stopping previous Bridge worker:', old_pid)
            os.killpg(old_pid, signal.SIGTERM)
            time.sleep(3)
    except (ProcessLookupError, FileNotFoundError, ValueError, PermissionError):
        pass
URL_FILE.unlink(missing_ok=True)

# Bu kaynak kod notebook'a gomuludur; baska dosya indirmen gerekmez.
WORKER_CODE = __INLINE_WORKER__
WORKER.write_text(WORKER_CODE, encoding='utf-8')

env = os.environ.copy()
env['PYTHONPATH'] = str(PIL_DIR) + os.pathsep + env.get('PYTHONPATH','')
env['BRIDGE_NGROK_TOKEN'] = NGROK_TOKEN
env['BRIDGE_API_PASSWORD'] = API_PASSWORD
env['BRIDGE_HF_TOKEN'] = HF_TOKEN or ''
env['BRIDGE_URL_FILE'] = str(URL_FILE)
print('3/4 - Starting FastAPI and ngrok…')
with LOG.open('w', encoding='utf-8') as logfile:
    process = subprocess.Popen([sys.executable, '-u', str(WORKER)], cwd=str(ROOT),
                               stdout=logfile, stderr=subprocess.STDOUT,
                               env=env, start_new_session=True)
PID_FILE.write_text(str(process.pid))

for i in range(240):
    if URL_FILE.exists():
        break
    if process.poll() is not None:
        log_text = LOG.read_text(encoding='utf-8')
        print(log_text[-8000:])
        if 'ERR_NGROK_334' in log_text:
            raise RuntimeError('NGROK DOMAIN IS BUSY. Stop the old Colab/ngrok session and rerun this cell. (ERR_NGROK_334)')
        raise RuntimeError('Bridge worker exited unexpectedly')
    time.sleep(0.5)
else:
    print(LOG.read_text(encoding='utf-8')[-7000:])
    raise RuntimeError('Bridge startup failed; share the log details')

url = URL_FILE.read_text(encoding='utf-8').strip()
print('4/4 - BRIDGE READY!')
print('WEB:  ',url)
print('API:  ',url + '/api/health')
print('DOCS: ',url + '/docs')
print('Sign in: admin / BRIDGE_PASSWORD from Colab Secrets')
print('Open Model configuration in the web UI. Chat test: Qwen/Qwen2.5-0.5B-Instruct')
print('Image example: stabilityai/sdxl-turbo (text-to-image and image-to-image)')
print('Image-to-Image API: POST /api/image-to-image (multipart/form-data; file, prompt, strength, steps)')
print('Listing cover API: POST /api/estate-cover (multipart/form-data; files 3-6, prompt, format, style)')
print('GGUF example: HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive (chat / llama.cpp)')
print('llama.cpp compiles CUDA support on the first GGUF load; monitor the load stage in the UI.')
print('Note: The public API stops when the Colab runtime stops.')
