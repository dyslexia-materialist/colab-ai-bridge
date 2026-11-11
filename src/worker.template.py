# COLAB AI BRIDGE v0.9 WORKER (tek hucre tarafindan otomatik yazilir)
# Pillow izole klasorden yuklenir; buradan sonra model kutuphaneleri ice aktarilir.
from PIL import Image, ImageText, ImageFont, ImageOps, ImageDraw, ImageFilter
from PIL._typing import _Ink
import PIL
print('Pillow verified:', PIL.__version__, PIL.__file__, flush=True)

import gc, io, json, time, secrets, threading, traceback, subprocess, shutil, os, re, pathlib, requests, math
from typing import Literal
from getpass import getpass
from fastapi import FastAPI, HTTPException, Depends, Request, Header, File, Form, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from huggingface_hub.utils import validate_repo_id
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from pyngrok import ngrok
from transformers import CLIPTokenizer
from huggingface_hub import hf_hub_download, HfApi, try_to_load_from_cache

# SDXL Turbo tokenizer dosyalari eski BPE formatindadir (vocab.json + merges.txt).
# Transformers 5'te bazi eski CLIP depolari tokenizer.json bulunmadiginda
# genel bir sentencepiece/tiktoken hatasi verebilir. Burada modelin KENDI
# BPE vocab/merges dosyalarindan tokenizera dogrudan basliyoruz.
def make_legacy_clip_tokenizer(repo_id, subfolder):
    vocab_path = hf_hub_download(repo_id=repo_id, filename='vocab.json',
                                 subfolder=subfolder, token=HF_TOKEN)
    merges_path = hf_hub_download(repo_id=repo_id, filename='merges.txt',
                                  subfolder=subfolder, token=HF_TOKEN)
    with open(vocab_path, encoding='utf-8') as f:
        vocab = json.load(f)
    with open(merges_path, encoding='utf-8') as f:
        merges = [tuple(line.strip().split()) for line in f
                  if line.strip() and not line.startswith('#')]
    # SDXL'in iki CLIP encoder'i 77-token context kullanir. Bu sinir
    # verilmezse elle olusturulan tokenizer.model_max_length cok buyuk
    # sentinel degeri alabilir; padding='max_length' OverflowError verir.
    tokenizer = CLIPTokenizer(vocab=vocab, merges=merges, model_max_length=77)
    tokenizer.model_max_length = 77
    test = tokenizer('test', padding='max_length', truncation=True,
                     max_length=77)
    assert len(test.input_ids) == 77, 'CLIP tokenizer uzunluk testi basarisiz'
    print('CLIP tokenizer ready:', subfolder,
          '| max_length:', tokenizer.model_max_length, flush=True)
    return tokenizer

from PIL import Image
import torch
import uvicorn
from starlette.concurrency import run_in_threadpool

if not torch.cuda.is_available():
    raise RuntimeError('GPU is unavailable. Select a GPU runtime and rerun the cell.')

# Secret bilgileri Colab hucresi tarafindan ortam degiskenlerine iletilir.
import os
NGROK_TOKEN = os.environ.get('BRIDGE_NGROK_TOKEN', '')
API_PASSWORD = os.environ.get('BRIDGE_API_PASSWORD', '')
HF_TOKEN = os.environ.get('BRIDGE_HF_TOKEN') or None
if not NGROK_TOKEN or len(API_PASSWORD) < 12:
    raise RuntimeError('Missing or invalid NGROK_AUTHTOKEN / BRIDGE_PASSWORD.')

PORT = 8905
DTYPE = torch.float16
state_lock = threading.Lock()
gpu_lock = threading.Lock()
model_state = {'status': 'idle', 'model_id': None, 'task': None, 'requested': None, 'error': None, 'stage': '', 'engine': None, 'gguf_file': None, 'gguf_quant': None, 'context_size': None, 'img2img_supported': False}
models = {'chat': None, 'tokenizer': None, 'image': None, 'img2img': None}
llama_proc = None
LLAMA_PORT = 8916
LLAMA_API_KEY = secrets.token_urlsafe(24)
LLAMA_BASE = f'http://127.0.0.1:{LLAMA_PORT}'
ROOT = pathlib.Path('/content/colab_ai_bridge_v09')
LLAMA_REPO = ROOT / 'llama.cpp'
LLAMA_BIN = LLAMA_REPO / 'build' / 'bin' / 'llama-server'
LLAMA_LOG = ROOT / 'llama-server.log'
BUILD_LOG = ROOT / 'llama-build.log'

class LoadRequest(BaseModel):
    model_id: str = Field(min_length=3, max_length=200)
    task: Literal['chat', 'image', 'image_to_image']
    gguf_file: str | None = Field(default=None, max_length=255)
    engine: Literal['auto', 'transformers', 'llama.cpp', 'diffusers'] = 'auto'
    context_size: Literal[2048, 4096, 8192, 16384] = 4096

class ChatItem(BaseModel):
    role: Literal['user','assistant']
    content: str = Field(min_length=1, max_length=4000)

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatItem] = Field(default_factory=list, max_length=20)
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0, le=2)

class ImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str = Field(default='', max_length=1000)
    width: int = Field(default=512, ge=256, le=1024, multiple_of=8)
    height: int = Field(default=512, ge=256, le=1024, multiple_of=8)
    steps: int = Field(default=4, ge=1, le=50)
    guidance_scale: float = Field(default=0, ge=0, le=20)
    seed: int | None = Field(default=None, ge=0, le=4294967295)

# Image-to-Image istegi multipart/form-data kabul eder. Maksimum 10 MB.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_PIXELS = 24_000_000
Image.MAX_IMAGE_PIXELS = MAX_UPLOAD_PIXELS

basic = HTTPBasic(auto_error=False)

def authorize(credentials: HTTPBasicCredentials | None = Depends(basic),
              x_bridge_password: str | None = Header(default=None)):
    valid_header = bool(x_bridge_password) and secrets.compare_digest(x_bridge_password, API_PASSWORD)
    valid_basic = (credentials is not None and
                   secrets.compare_digest(credentials.username, 'admin') and
                   secrets.compare_digest(credentials.password, API_PASSWORD))
    if not (valid_header or valid_basic):
        raise HTTPException(status_code=401, detail='Authentication required',
                            headers={'WWW-Authenticate': 'Basic realm="Colab AI Bridge"'})

def get_status():
    with state_lock:
        return dict(model_state)

def set_status(**kw):
    with state_lock:
        model_state.update(kw)

# GGUF dosyalarini Hugging Face deposundan algila; mmproj ve cok parcali
# GGUF dosyalarini ilk surumde metin modeli olarak kabul etmiyoruz.
def repo_files(repo_id):
    info = HfApi(token=HF_TOKEN).model_info(repo_id, files_metadata=True)
    siblings = info.siblings or []
    ggufs = []
    for f in siblings:
        name = f.rfilename
        basename = name.rsplit('/', 1)[-1]
        if (not name.lower().endswith('.gguf') or
            basename.lower().startswith('mmproj') or
            re.search(r'-\d{5}-of-\d{5}\.gguf$', basename, re.IGNORECASE)):
            continue
        quant_match = re.search(r'(IQ[1-9]_[A-Z0-9_]+|Q[2-9]_[A-Z0-9_]+|BF16|F16|F32)\.gguf$', basename, re.I)
        quant = quant_match.group(1).upper() if quant_match else 'GGUF'
        ggufs.append({'name': name, 'quant': quant, 'bytes': f.size, 'gb': round(f.size/1e9,2) if f.size else None})
    ggufs.sort(key=lambda x: x['bytes'] if x['bytes'] is not None else 10**20)
    has_st = any(x.rfilename.lower().endswith('.safetensors') for x in siblings)
    has_shards = any(re.search(r'-\d{5}-of-\d{5}\.gguf$',x.rfilename,re.I) for x in siblings)
    return {'model_id': repo_id, 'gguf_files': ggufs, 'has_safetensors': has_st,
            'has_sharded_gguf': has_shards}

def auto_gguf_file(files):
    if not files: raise ValueError('No supported single-file GGUF found in this repository.')
    free_vram, _ = torch.cuda.mem_get_info()
    # VRAM'de KV cache ve diger GPU isleri icin rezerv birak.
    budget = max(0, free_vram - 4 * 1024**3)
    fits = [f for f in files if f['bytes'] and f['bytes'] <= budget]
    if fits:
        return max(fits, key=lambda f: f['bytes'])['name']
    if len(files) == 1 and files[0]['bytes'] is None:
        return files[0]['name']
    raise ValueError('No GGUF file fits the automatic GPU memory budget (or sizes are unknown). Select a smaller quantization or consider partial CPU offload.')

def run_build(args, label, timeout):
    ROOT.mkdir(parents=True, exist_ok=True)
    with BUILD_LOG.open('a', encoding='utf-8') as lg:
        lg.write('\n' + label + ': ' + ' '.join(map(str,args)) + '\n')
        lg.flush()
        p = subprocess.run(args, stdout=lg, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if p.returncode:
        details = BUILD_LOG.read_text(errors='replace')[-1800:]
        raise RuntimeError(label + ' failed. Details: ' + str(BUILD_LOG) + '\n' + details)

def ensure_llama_binary():
    if LLAMA_BIN.is_file(): return
    set_status(stage='Installing llama.cpp and compiling CUDA support (first GGUF load may be slow)')
    if not shutil.which('git'):
        raise RuntimeError('git is unavailable in the Colab runtime.')
    if not shutil.which('nvcc') and not pathlib.Path('/usr/local/cuda/bin/nvcc').is_file():
        raise RuntimeError('CUDA compiler nvcc not found. CUDA Toolkit is required for GGUF.')
    if not shutil.which('cmake') or not shutil.which('ninja'):
        run_build([os.sys.executable,'-m','pip','install','-q','cmake>=3.28','ninja>=1.11'], 'Installing CMake/Ninja', 500)
    if not (LLAMA_REPO/'CMakeLists.txt').exists():
        if LLAMA_REPO.exists(): shutil.rmtree(LLAMA_REPO)
        run_build(['git','clone','--depth','1','https://github.com/ggml-org/llama.cpp.git',str(LLAMA_REPO)], 'Cloning llama.cpp', 400)
    cc = ''.join(map(str, torch.cuda.get_device_capability(0)))
    run_build(['cmake','-S',str(LLAMA_REPO),'-B',str(LLAMA_REPO/'build'),'-G','Ninja',
               '-DCMAKE_BUILD_TYPE=Release','-DGGML_CUDA=ON','-DLLAMA_CURL=OFF',
               f'-DCMAKE_CUDA_ARCHITECTURES={cc}'], 'llama.cpp CMake', 600)
    run_build(['cmake','--build',str(LLAMA_REPO/'build'),'--target','llama-server','--parallel','2'],
              'Compiling llama.cpp with CUDA', 2400)
    if not LLAMA_BIN.is_file(): raise RuntimeError('llama-server was built but the executable was not found: '+str(LLAMA_BIN))

def stop_llama():
    global llama_proc
    if llama_proc is not None:
        if llama_proc.poll() is None:
            llama_proc.terminate()
            try: llama_proc.wait(timeout=12)
            except subprocess.TimeoutExpired:
                llama_proc.kill(); llama_proc.wait(timeout=8)
        llama_proc = None

def start_llama(path, context_size=4096):
    global llama_proc
    set_status(stage='Loading GGUF into GPU / CPU memory')
    env = os.environ.copy()
    env['LLAMA_API_KEY'] = LLAMA_API_KEY
    args = [str(LLAMA_BIN), '--model', str(path), '--host','127.0.0.1',
            '--port',str(LLAMA_PORT), '--ctx-size',str(context_size),
            '--n-gpu-layers','auto', '--jinja']
    with LLAMA_LOG.open('w', encoding='utf-8') as logfile:
        llama_proc = subprocess.Popen(args, stdout=logfile, stderr=subprocess.STDOUT, env=env)
    for i in range(900):
        if llama_proc.poll() is not None:
            raise RuntimeError('llama-server failed to start: ' + LLAMA_LOG.read_text(errors='replace')[-1600:])
        try:
            health = requests.get(LLAMA_BASE+'/health', timeout=2)
            if health.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError('llama-server did not become ready; log: ' + str(LLAMA_LOG))

def unload():
    stop_llama()
    models['chat'] = models['tokenizer'] = models['image'] = models['img2img'] = None
    gc.collect()
    torch.cuda.empty_cache()

def loader(model_id, task, gguf_file=None, requested_engine="auto", context_size=4096):
    try:
        with gpu_lock:
            set_status(stage='Unloading previous model from GPU memory')
            unload()
            if task == 'chat':
                set_status(stage='Inspecting model repository files')
                info = repo_files(model_id)
                if requested_engine == 'diffusers':
                    raise ValueError('Diffusers cannot run chat models')
                use_gguf = requested_engine == 'llama.cpp' or (requested_engine == 'auto' and bool(info['gguf_files']) and (bool(gguf_file) or not info['has_safetensors']))
                if requested_engine == 'transformers' and gguf_file:
                    raise ValueError('GGUF files cannot run with Transformers here')
                if use_gguf and not info['gguf_files']:
                    raise ValueError('No supported GGUF file in this repository')
                if not use_gguf and not info['has_safetensors']:
                    raise ValueError('No supported GGUF or safetensors files found')
                if use_gguf:
                    available = {f['name']: f for f in info['gguf_files']}
                    choice = gguf_file or auto_gguf_file(info['gguf_files'])
                    if choice not in available:
                        raise ValueError('Selected GGUF file is not available in this repository')
                    set_status(engine='llama.cpp', gguf_file=choice, gguf_quant=available[choice].get('quant'),
                               context_size=context_size, stage='Preparing llama.cpp GGUF engine')
                    ensure_llama_binary()
                    size = available[choice]['bytes']
                    cached = try_to_load_from_cache(repo_id=model_id, filename=choice)
                    if not isinstance(cached, str) and size:
                        disk_free = shutil.disk_usage('/content').free
                        if disk_free < size + (2 * 1024**3):
                            raise RuntimeError(f'Insufficient disk space: file ~{size/1e9:.1f} GB; free space {disk_free/1e9:.1f} GB. Choose a smaller GGUF.')
                    set_status(stage='Downloading GGUF / using local cache')
                    path = hf_hub_download(repo_id=model_id, filename=choice, token=HF_TOKEN)
                    start_llama(path, context_size=context_size)
                    engine = 'llama.cpp'
                else:
                    set_status(engine='transformers', gguf_file=None, gguf_quant=None, context_size=None, stage='Loading Transformers model')
                    tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=False)
                    mdl = AutoModelForCausalLM.from_pretrained(
                        model_id, token=HF_TOKEN, dtype=DTYPE,
                        device_map='auto', trust_remote_code=False, use_safetensors=True
                    )
                    mdl.eval()
                    models['tokenizer'], models['chat'] = tok, mdl
                    engine = 'transformers'
            elif task in ('image', 'image_to_image'):
                if requested_engine not in ('auto', 'diffusers'):
                    raise ValueError('Image tasks require the Diffusers or Auto engine')
                if gguf_file:
                    raise ValueError('GGUF chat models cannot generate images')
                set_status(engine='diffusers', gguf_file=None, gguf_quant=None, context_size=None, stage='Loading Diffusers model')
                image_kwargs = dict(token=HF_TOKEN, dtype=DTYPE, use_safetensors=True)
                if model_id == 'stabilityai/sdxl-turbo':
                    image_kwargs['tokenizer'] = make_legacy_clip_tokenizer(model_id, 'tokenizer')
                    image_kwargs['tokenizer_2'] = make_legacy_clip_tokenizer(model_id, 'tokenizer_2')
                if task == 'image_to_image':
                    # Yalnizca img2img destegi bulunan modeller icin.
                    img_pipe = AutoPipelineForImage2Image.from_pretrained(model_id, **image_kwargs)
                    models['img2img'] = img_pipe.to('cuda')
                    models['image'] = None
                else:
                    pipe = AutoPipelineForText2Image.from_pretrained(model_id, **image_kwargs)
                    pipe = pipe.to('cuda')
                    models['image'] = pipe
                    # from_pipe agirliklari paylasir; ikinci model kopyasi yuklemez.
                    try:
                        models['img2img'] = AutoPipelineForImage2Image.from_pipe(pipe)
                        print('Image-to-Image pipeline is available.', flush=True)
                    except (ValueError, TypeError, ImportError, NotImplementedError) as exc:
                        models['img2img'] = None
                        print('Image-to-Image is not supported for this model:', str(exc)[:200], flush=True)
                engine = 'diffusers'
            else:
                raise ValueError('Unknown model task: ' + str(task))
            set_status(status='ready', model_id=model_id, task=task,
                       requested=None, error=None, stage='Model ready', engine=engine,
                       img2img_supported=models['img2img'] is not None)
            print('Model ready:', model_id, task, engine, flush=True)
    except Exception as e:
        traceback.print_exc(limit=8)
        # exception with GPU lock unwound; cleanup here
        with gpu_lock:
            unload()
        set_status(status='error', model_id=None, task=None, requested=model_id,
                   engine=None, gguf_file=None, gguf_quant=None, context_size=None, stage='Error',
                   error=f'{type(e).__name__}: {str(e)[:1100]}', img2img_supported=False)

app = FastAPI(title='Colab AI Bridge', version='0.9.0')

@app.get('/', response_class=HTMLResponse, dependencies=[Depends(authorize)])
def home():
    return HTMLResponse(content=HTML_PAGE, headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})

@app.get('/api/health')
def health():
    return {'online': True}

@app.get('/api/models/status', dependencies=[Depends(authorize)])
def status():
    s = get_status()
    free, total = torch.cuda.mem_get_info()
    return {**s, 'gpu': torch.cuda.get_device_name(0),
            'vram_free_gb': round(free / 1024**3, 2),
            'vram_total_gb': round(total / 1024**3, 2)}

@app.get('/api/models/inspect', dependencies=[Depends(authorize)])
def inspect_model(model_id: str):
    try:
        validate_repo_id(model_id)
        info = repo_files(model_id)
        try: info['auto_gguf_file'] = auto_gguf_file(info['gguf_files']) if info['gguf_files'] else None
        except ValueError: info['auto_gguf_file'] = None
        return info
    except Exception as exc:
        raise HTTPException(400, 'Could not inspect model repository: ' + str(exc)[:250])

@app.post('/api/models/load', dependencies=[Depends(authorize)])
def load_model(data: LoadRequest):
    name = data.model_id.strip()
    try:
        validate_repo_id(name)
    except Exception:
        raise HTTPException(400, 'Invalid Hugging Face model ID')
    with state_lock:
        if model_state['status'] == 'loading':
            raise HTTPException(409, 'Another model is loading')
        if model_state['status'] == 'ready' and model_state['model_id'] == name and model_state['task'] == data.task and (not data.gguf_file or model_state.get('gguf_file') == data.gguf_file) and (data.engine == 'auto' or model_state.get('engine') == data.engine) and (model_state.get('engine') != 'llama.cpp' or model_state.get('context_size') == data.context_size):
            return {'status': 'ready', 'model_id': name, 'task': data.task}
        model_state.update(status='loading', requested=name, model_id=None, task=data.task, error=None, stage='Starting', engine=None, gguf_file=data.gguf_file, gguf_quant=None, context_size=data.context_size, img2img_supported=False)
    threading.Thread(target=loader, args=(name, data.task, data.gguf_file, data.engine, data.context_size), daemon=True).start()
    return {'status': 'loading', 'model_id': name, 'task': data.task}

@app.post('/api/chat', dependencies=[Depends(authorize)])
def chat(data: ChatRequest):
    if get_status()['status'] != 'ready' or get_status()['task'] != 'chat':
        raise HTTPException(503, 'Chat model is not ready')
    with gpu_lock:
        if get_status()['status'] != 'ready' or (get_status().get('engine') != 'llama.cpp' and models['chat'] is None):
            raise HTTPException(503, 'Model is being changed')
        msgs = [{'role': x.role, 'content': x.content} for x in data.history]
        msgs.append({'role': 'user', 'content': data.message})
        if get_status().get('engine') == 'llama.cpp':
            try:
                payload = {'model': 'current', 'messages': msgs,
                           'max_tokens': data.max_new_tokens,
                           'temperature': data.temperature, 'stream': False}
                res = requests.post(LLAMA_BASE+'/v1/chat/completions',
                    headers={'Authorization':'Bearer '+LLAMA_API_KEY},
                    json=payload, timeout=300)
                if res.status_code != 200:
                    raise RuntimeError('llama-server HTTP '+str(res.status_code)+': '+res.text[:200])
                output = res.json()['choices'][0]['message']['content']
                if isinstance(output, list):
                    output = '\n'.join(str(x.get('text',x)) if isinstance(x,dict) else str(x) for x in output)
                return {'answer': str(output or '').strip(), 'model': get_status()['model_id'], 'engine': 'llama.cpp'}
            except (requests.RequestException, ValueError, KeyError, IndexError, RuntimeError) as exc:
                traceback.print_exc(limit=6)
                raise HTTPException(500, 'GGUF chat error: '+str(exc)[:240])
        mdl, tok = models['chat'], models['tokenizer']
        if getattr(tok, 'chat_template', None):
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors='pt')
        else:
            prompt = '\n'.join(f"{x['role']}: {x['content']}" for x in msgs) + '\nassistant:'
            enc = tok(prompt, return_tensors='pt')
        enc = {k: v.to(mdl.device) for k,v in enc.items()}
        params = {'max_new_tokens': data.max_new_tokens,
                  'do_sample': data.temperature > 0,
                  'pad_token_id': tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id}
        if data.temperature > 0:
            params['temperature'] = data.temperature
            params['top_p'] = 0.95
        try:
            with torch.inference_mode():
                out = mdl.generate(**enc, **params)
            answer = tok.decode(out[0][enc['input_ids'].shape[-1]:], skip_special_tokens=True)
            return {'answer': answer.strip(), 'model': get_status()['model_id']}
        except Exception as e:
            traceback.print_exc(limit=6)
            raise HTTPException(500, f'Model generation error: {type(e).__name__}')

@app.post('/api/image', dependencies=[Depends(authorize)])
def image(data: ImageRequest):
    if get_status()['status'] != 'ready' or get_status()['task'] != 'image':
        raise HTTPException(503, 'Image model is not ready')
    with gpu_lock:
        if get_status()['status'] != 'ready' or models['image'] is None:
            raise HTTPException(503, 'Model is being changed')
        seed_generator = torch.Generator(device='cuda').manual_seed(data.seed) if data.seed is not None else None
        try:
            with torch.inference_mode():
                result = models['image'](
                    prompt=data.prompt,
                    negative_prompt=data.negative_prompt or None,
                    width=data.width, height=data.height,
                    num_inference_steps=data.steps,
                    guidance_scale=data.guidance_scale,
                    generator=seed_generator
                )
            buf = io.BytesIO()
            result.images[0].save(buf, format='PNG')
            return Response(content=buf.getvalue(), media_type='image/png',
                            headers={'Cache-Control':'no-store'})
        except Exception as e:
            traceback.print_exc(limit=6)
            raise HTTPException(500, f'Image generation error: {type(e).__name__}')

# Yuklenen referans resmi bellekte isler; diskte kalici olarak kaydetmez.
def _generate_image_to_image(source: Image.Image, prompt: str, negative_prompt: str,
                             width: int, height: int, steps: int,
                             guidance_scale: float, strength: float, seed: int | None):
    with gpu_lock:
        status = get_status()
        pipe = models['img2img']
        if status['status'] != 'ready' or pipe is None:
            raise HTTPException(503, 'Image-to-Image model is not ready or is unsupported')
        # Goruntunun merkezinden kirparak hedef boyuta uyarla.
        source = ImageOps.fit(source, (width, height), method=Image.Resampling.LANCZOS)
        generator = torch.Generator(device='cuda').manual_seed(seed) if seed is not None else None
        try:
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    image=source,
                    negative_prompt=negative_prompt or None,
                    strength=strength,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                )
            out = io.BytesIO()
            result.images[0].save(out, format='PNG')
            return out.getvalue()
        except Exception as exc:
            traceback.print_exc(limit=8)
            raise HTTPException(500, 'Image-to-Image generation error: '+type(exc).__name__+': '+str(exc)[:240])

@app.post('/api/image-to-image', dependencies=[Depends(authorize)],
          responses={200: {'content': {'image/png': {}}, 'description': 'PNG sonucu'}})
async def image_to_image(
    file: UploadFile = File(..., description='PNG, JPEG veya WEBP; en fazla 10 MB'),
    prompt: str = Form(..., min_length=1, max_length=2000),
    negative_prompt: str = Form('', max_length=1000),
    width: int = Form(512, ge=256, le=1024, multiple_of=8),
    height: int = Form(512, ge=256, le=1024, multiple_of=8),
    steps: int = Form(4, ge=1, le=50),
    guidance_scale: float = Form(0.0, ge=0, le=20),
    strength: float = Form(0.7, gt=0, le=1),
    seed: int | None = Form(None, ge=0, le=4294967295)
):
    if get_status()['status'] != 'ready' or get_status()['task'] not in ('image', 'image_to_image'):
        raise HTTPException(503, 'Load an image or image-to-image model first')
    if models['img2img'] is None:
        raise HTTPException(400, 'Selected model does not support image-to-image')
    if file.content_type not in ('image/png', 'image/jpeg', 'image/webp'):
        raise HTTPException(415, 'Only PNG, JPEG, or WEBP files are supported')
    if steps * strength < 1:
        raise HTTPException(422, 'Steps × strength must be at least 1')
    try:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, 'File exceeds 10 MB')
        try:
            with Image.open(io.BytesIO(data)) as raw:
                if raw.format not in ('PNG', 'JPEG', 'WEBP'):
                    raise ValueError('Unsupported image format')
                if raw.width * raw.height > MAX_UPLOAD_PIXELS:
                    raise ValueError('Image exceeds the pixel limit')
                source = ImageOps.exif_transpose(raw).convert('RGB')
                source.load()
        except (ValueError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(400, 'Could not open image: '+str(exc)[:160])
        output_bytes = await run_in_threadpool(
            _generate_image_to_image, source, prompt, negative_prompt,
            width, height, steps, guidance_scale, strength, seed
        )
        return Response(content=output_bytes, media_type='image/png',
                        headers={'Cache-Control':'no-store'})
    finally:
        await file.close()

# ----------------------------------------------------------
# EMLAK ILAN KAPAGI: 3-6 gercek fotografi tasarimda kullanir.
# Metin alanlari kullanicinin promptundan okunur; emlak ozelligi uydurulmaz.
# Bu tasarim GPU veya yuklu diffusion modeline bagimli degildir.
# ----------------------------------------------------------
MAX_COVER_FILES = 6
MIN_COVER_FILES = 3
COVER_MAX_PER_FILE = 12 * 1024 * 1024
COVER_MAX_TOTAL = 48 * 1024 * 1024
COVER_MAX_PIXELS = 30_000_000
COVER_SIZES = {'story': (1080,1920), 'post': (1080,1350),
               'square': (1080,1080), 'landscape': (1200,675)}
COVER_THEMES = {
  'navy': {'bg':'#0d1b2a','panel':'#142c43','accent':'#eab56b','text':'#ffffff','muted':'#b9cadd'},
  'black': {'bg':'#111419','panel':'#25282e','accent':'#dfb777','text':'#ffffff','muted':'#d6d8dc'},
  'blue': {'bg':'#132849','panel':'#1e416f','accent':'#6fcbe8','text':'#ffffff','muted':'#d2e5f4'},
  'light': {'bg':'#f5f1e9','panel':'#ffffff','accent':'#b27538','text':'#172334','muted':'#506072'},
}
def _pick_cover_theme(prompt, style):
    if style != 'auto': return style
    p = prompt.casefold()
    if any(x in p for x in ('minimal', 'beyaz', 'açık renk','acik renk','sade')): return 'light'
    if any(x in p for x in ('siyah', 'black', 'koyu', 'premium')): return 'black'
    if any(x in p for x in ('mavi', 'lacivert', 'blue')): return 'blue'
    return 'navy'

def _listing_fields(prompt):
    # Girilen ilandaki bilgileri oldugu gibi aktar; fiyat/m2/oda bilgisi uydurma.
    labels = {
        'title': r'(?:başlık|baslik|title|ilan başlığı|ilan basligi)',
        'price': r'(?:fiyat|price|bedel)',
        'location': r'(?:konum|lokasyon|location|adres)',
        'features': r'(?:özellikler|ozellikler|detaylar|features|highlights)',
        'type': r'(?:ilan tipi|ilan türü|ilan turu|type|listing type)'
    }
    data = {}
    for name,keys in labels.items():
        m = re.search(r'(?im)^\s*(?:'+keys+r')\s*[:：=-]\s*(.+?)\s*$',prompt)
        if m: data[name] = m.group(1).strip()
    low=prompt.casefold()
    inferred_type = 'FOR RENT' if any(x in low for x in ('kiralık','kiralik','for rent','rental')) else ('FOR SALE' if any(x in low for x in ('satılık','satilik','for sale','selling')) else 'PROPERTY LISTING')
    kind=data.get('type', inferred_type).upper()
    if len(kind)>36: kind=kind[:36]
    title=data.get('title', 'PROPERTY LISTING')
    price=data.get('price','')
    if not price:
        m=re.search(r'(?i)\b(?:\d{1,3}(?:[.,]\d{3})+|\d{4,})\s*(?:TL|₺|TRY)\b',prompt)
        if m: price=m.group(0)
    location=data.get('location','')
    features=data.get('features','')
    if not features:
        bits=[]
        for pat in (r'\b\d+\s*\+\s*\d+\b',r'\b\d+(?:[.,]\d+)?\s*(?:m²|m2)\b'):
            m=re.search(pat,prompt,flags=re.I)
            if m: bits.append(m.group(0))
        features=' • '.join(bits)
    return {'type':kind,'title':title[:150],'price':price[:70],
            'location':location[:120],'features':features[:220]}

def _font(size, bold=False):
    paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'
    ]
    for path in paths:
        if os.path.exists(path): return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def _fit_line(draw, text, width, size, min_size=22, bold=False):
    text=str(text or '').strip()
    while size>=min_size and draw.textbbox((0,0),text,font=_font(size,bold))[2]>width:
        size-=2
    font=_font(size,bold)
    if draw.textbbox((0,0),text,font=font)[2]>width:
        while text and draw.textbbox((0,0),text+'…',font=font)[2]>width:
            text=text[:-1]
        text=text+'…' if text else ''
    return text,font

def _wrap_lines(draw, text, width, fontsize, max_lines=2, bold=False):
    words=str(text or '').strip().split()
    if not words:return []
    lines=[];line=''
    font=_font(fontsize,bold)
    for word in words:
        test=(line+' '+word).strip()
        if line and draw.textbbox((0,0),test,font=font)[2]>width:
            lines.append(line);line=word
        else:line=test
    if line:lines.append(line)
    if len(lines)>max_lines:
        lines=lines[:max_lines]
        lines[-1],_= _fit_line(draw,lines[-1]+'…',width,fontsize,fontsize,bold)
    return lines

def _color(hexstr):
    return tuple(int(hexstr[i:i+2],16) for i in (1,3,5))

def _photo_panel(canvas, photo, box, radius=18):
    x,y,w,h=map(int,box)
    if min(w,h)<2:return
    # Kucuk bir kenar kirpmasi olabilir; nesne ekleme/silme veya mimari degisim yok.
    image=ImageOps.fit(photo,(w,h),Image.Resampling.LANCZOS,centering=(.5,.5))
    mask=Image.new('L',(w,h),0)
    ImageDraw.Draw(mask).rounded_rectangle((0,0,w-1,h-1),radius=radius,fill=255)
    canvas.paste(image,(x,y),mask)

def _render_cover(photos,prompt,format_name,style,main_index):
    W,H=COVER_SIZES[format_name]; th=COVER_THEMES[_pick_cover_theme(prompt,style)]
    canvas=Image.new('RGB',(W,H),_color(th['bg']))
    d=ImageDraw.Draw(canvas)
    accent,text,muted=th['accent'],th['text'],th['muted']
    fields=_listing_fields(prompt)
    gap=14 if W==1080 else 12
    margin=42 if W==1080 else 34
    hero=photos[main_index]
    others=[p for n,p in enumerate(photos) if n!=main_index]
    # Ayrica profesyonel kompozisyon: yatay formatta fotograflar solda, bilgiler sagda.
    if format_name=='landscape':
        left_w=int(W*.64)
        d.rectangle((left_w+12,0,W,H),fill=_color(th['panel']))
        hero_h=int(H*.67)
        _photo_panel(canvas,hero,(margin,margin,left_w-margin*2,hero_h-margin),14)
        ty=hero_h+gap
        tw=(left_w-margin*2-gap*(len(others)-1))//len(others)
        for i,p in enumerate(others):
            _photo_panel(canvas,p,(margin+i*(tw+gap),ty,tw,H-ty-margin),12)
        x=left_w+42;y=60;cw=W-x-34
        label,f=_fit_line(d,fields['type'],cw,28,20,True)
        d.text((x,y),label,font=f,fill=accent);y+=55
        for line in _wrap_lines(d,fields['title'],cw,35,3,True):
            d.text((x,y),line,font=_font(35,True),fill=text);y+=50
        y+=14
        if fields['price']:
            title,f=_fit_line(d,fields['price'],cw,40,25,True)
            d.text((x,y),title,font=f,fill=accent);y+=62
        for value in (fields['location'],fields['features']):
            if not value: continue
            for line in _wrap_lines(d,value,cw,23,2):
                d.text((x,y),line,font=_font(23),fill=muted);y+=36
        return canvas
    # Dikey/kare format: ustte hero, altinda kucuk fotograflar ve ilan bilgileri.
    top=122 if H>=1350 else 94
    hero_h=int(H*(0.43 if H>=1800 else 0.43 if H>=1300 else 0.39))
    thumb_h=int(H*(0.145 if H>=1800 else 0.13))
    d.text((margin,30), 'PROPERTY  /  LISTING',font=_font(23,True),fill=muted)
    lab,labfont=_fit_line(d,fields['type'],int(W*.45),30,20,True)
    lab_w=int(d.textbbox((0,0),lab,font=labfont)[2])+38
    d.rounded_rectangle((W-margin-lab_w,27,W-margin,75),radius=22,fill=_color(th['panel']))
    d.text((W-margin-lab_w+19,34),lab,font=labfont,fill=accent)
    _photo_panel(canvas,hero,(margin,top,W-margin*2,hero_h),19)
    thumb_y=top+hero_h+gap
    tw=(W-margin*2-gap*(len(others)-1))//len(others)
    for i,p in enumerate(others):
        _photo_panel(canvas,p,(margin+i*(tw+gap),thumb_y,tw,thumb_h),12)
    y=thumb_y+thumb_h+max(24,int(H*.03))
    if fields['price']:
        value,font=_fit_line(d,fields['price'],W-margin*2,60 if H>=1300 else 49,26,True)
        d.text((margin,y),value,font=font,fill=accent)
        y+=75 if H>=1300 else 61
    title_max_lines=3 if H>=1800 else 2
    title_fontsize=45 if H>=1800 else (39 if H>=1300 else 35)
    for line in _wrap_lines(d,fields['title'],W-margin*2,title_fontsize,title_max_lines,True):
        d.text((margin,y),line,font=_font(title_fontsize,True),fill=text)
        y+=title_fontsize+17
    y+=10
    for value in (fields['location'], fields['features']):
        if not value:continue
        for line in _wrap_lines(d,value,W-margin*2,26 if H>=1300 else 22,2):
            if y > H-95:break
            d.text((margin,y),line,font=_font(26 if H>=1300 else 22),fill=muted)
            y+=40
    d.line((margin,H-70,W-margin,H-70),fill=_color(th['panel']),width=2)
    d.text((margin,H-57),'CREATED USING ORIGINAL PROPERTY PHOTOS',font=_font(15),fill=muted)
    return canvas

@app.post('/api/estate-cover', dependencies=[Depends(authorize)],
          responses={200: {'content': {'image/png': {}}, 'description':'PNG real estate cover'}})
async def estate_cover(
    files: list[UploadFile] = File(..., description='3-6 adet PNG/JPEG/WEBP'),
    prompt: str = Form(..., min_length=8, max_length=4000),
    format: Literal['story','post','square','landscape'] = Form('story'),
    style: Literal['auto','navy','black','blue','light'] = Form('auto'),
    main_index: int = Form(0, ge=0, le=5)
):
    if not (MIN_COVER_FILES <= len(files) <= MAX_COVER_FILES):
        raise HTTPException(422, 'Select 3 to 6 photos')
    if main_index >= len(files):
        raise HTTPException(422, 'Invalid main photo selection')
    photos=[];total=0
    try:
        for upload in files:
            if upload.content_type not in ('image/png','image/jpeg','image/webp'):
                raise HTTPException(415, 'Only PNG, JPEG, and WEBP photos are supported')
            blob=await upload.read(COVER_MAX_PER_FILE+1)
            total+=len(blob)
            if len(blob)>COVER_MAX_PER_FILE or total>COVER_MAX_TOTAL:
                raise HTTPException(413, 'The photo exceeds 12 MB or the total exceeds 48 MB')
            try:
                with Image.open(io.BytesIO(blob)) as inp:
                    if inp.format not in ('PNG','JPEG','WEBP') or inp.width*inp.height>COVER_MAX_PIXELS:
                        raise ValueError('Unsupported image format or pixel limit exceeded')
                    picture=ImageOps.exif_transpose(inp).convert('RGB');picture.load()
                    photos.append(picture)
            except (OSError,ValueError,Image.DecompressionBombError) as exc:
                raise HTTPException(400, 'Unable to read photo: '+str(exc)[:140])
        artwork=await run_in_threadpool(_render_cover, photos, prompt, format, style, main_index)
        output=io.BytesIO(); artwork.save(output,'PNG',optimize=True)
        return Response(content=output.getvalue(),media_type='image/png',headers={
            'Content-Disposition':'attachment; filename="emlak-ilan-kapagi.png"',
            'Cache-Control':'no-store'
        })
    finally:
        for upload in files: await upload.close()

HTML_PAGE = __INLINE_HTML__

# Uvicorn ve ngrok ayni worker prosesinde calisir.
cfg = uvicorn.Config(app, host='127.0.0.1', port=PORT, log_level='warning')
_BRIDGE_SERVER = uvicorn.Server(cfg)
threading.Thread(target=_BRIDGE_SERVER.run, daemon=True).start()
for _ in range(120):
    if _BRIDGE_SERVER.started: break
    time.sleep(0.25)
if not _BRIDGE_SERVER.started:
    raise RuntimeError('FastAPI did not start; port may be occupied')

# Eski config'de authtoken olsa bile yeni giris denemesi yapma.
# Token Colab Secrets'tan gelir ve bu ngrok prosesine ozel olarak kullanilir.
# PyngrokConfig.auth_token mevcut ngrok.yml'deki kaydi override eder.
from pyngrok import conf
bridge_ngrok_config = conf.PyngrokConfig(auth_token=NGROK_TOKEN)
try:
    tunnel = ngrok.connect(addr=PORT, proto='http', pyngrok_config=bridge_ngrok_config)
except Exception as exc:
    if 'ERR_NGROK_334' in str(exc):
        print('\nNGROK TUNNEL CONFLICT (ERR_NGROK_334)', flush=True)
        print('Your ngrok domain is already being used by another active session.', flush=True)
        print('1) Stop the previous Colab notebook if it is still running.', flush=True)
        print('2) Or stop the old session in ngrok dashboard > Agents / Tunnel Sessions.', flush=True)
        print('3) Rerun the single cell. Tokens and passwords stay in Colab Secrets.', flush=True)
        print('Do not enable pooling: requests might be split between old and new sessions.', flush=True)
        raise RuntimeError('ERR_NGROK_334: Stop the old ngrok tunnel session') from None
    raise
PUBLIC_URL = tunnel.public_url
from pathlib import Path
Path(os.environ['BRIDGE_URL_FILE']).write_text(PUBLIC_URL, encoding='utf-8')
print('WEB: ', PUBLIC_URL, flush=True)
print('API: ', PUBLIC_URL+'/api/health', flush=True)
print('DOCS:', PUBLIC_URL+'/docs', flush=True)
print('Load a model in the web interface.', flush=True)
try:
    while not _BRIDGE_SERVER.should_exit:
        time.sleep(1)
finally:
    ngrok.disconnect(PUBLIC_URL)
    ngrok.kill()
