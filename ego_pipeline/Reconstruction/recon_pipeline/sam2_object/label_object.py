#!/usr/bin/env python3
"""Interactive SAM2 object labeling: headed local browser or HTTP remote browser.

⚠ 人工点选是 **FALLBACK**, 不是唯一入口 —— 物体 mask 已有全自动路线, 别据本文件推断
"每条视频都要人点一次":
  - ego_pipeline/bin/auto_label_v17a.py   reconstruct.sh 默认自动调用: v17A(HOI-DETR+SAM2)
    自动发现实例并选帧, 取 mask 内切极点当点击, **只写 label_prompt.json** —— 本管线的
    SAM2 像有人点过一样传播全片(2026-08-10 定稿; 早期直接搬 v17A mask 的做法已废弃,
    v17A 的选帧质量门会把全片覆盖砍成几帧, screw27 实测 5/188 → 改后 188/188)
  - tools/v17a_to_label_prompt.py         同机制的独立工具(多物体版 v17a_multi_object_prompt.py)
只有自动路线失败/需要人工裁决时才用本工具(reconstruct.sh --web)。
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

RECON_ROOT = Path(__file__).resolve().parents[1]
SAM2_OBJ_DIR = Path(__file__).resolve().parent
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))
if str(SAM2_OBJ_DIR) not in sys.path:
    sys.path.insert(0, str(SAM2_OBJ_DIR))

from _common.dataset import VideoJob, discover_videos, resolve_video_job  # noqa: E402
from _common.io import count_video_frames, read_video_frame  # noqa: E402
from _common.paths import interim_step_dir, resolve_repo_path  # noqa: E402
from sam2_object_common import (  # noqa: E402
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_MODEL_CFG,
    LabelPrompt,
    ObjectPrompt,
    build_object_predictor,
    encode_object_mask_png,
    ensure_object_preview_video_state,
    preview_object_mask_on_frame,
    save_label_prompt,
)

HTML_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAM2 Object Label</title>
<style>
html,body{height:100%;margin:0}
body{font-family:sans-serif;background:#111;color:#eee;box-sizing:border-box;padding:10px 0 16px;overflow-y:auto}
header,.toolbar,.objectbar,.legend,.framebar,#status{padding:0 12px}
header h2{margin:0 0 6px;font-size:1.15rem}
header p{margin:0 0 8px;font-size:13px;color:#bbb;line-height:1.35}
.toolbar{margin:6px 0;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button{padding:8px 14px;cursor:pointer}
.pos{background:#2ecc71;color:#000;border:none}.neg{background:#e74c3c;color:#fff;border:none}
.objectbar,.framebar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:6px 0 8px}
.objectbar select{padding:8px 10px;min-width:130px}
.legend{display:flex;flex-wrap:wrap;gap:8px 14px;color:#bbb;font-size:12px;margin:4px 0 8px}
.legend span{white-space:nowrap}
kbd{background:#262626;border:1px solid #555;border-radius:3px;padding:1px 5px;color:#fff}
.framebar input[type=range]{flex:1;min-width:160px}
#status{color:#aaa;font-size:13px;margin:0 0 8px}
#stage{display:flex;justify-content:center;padding:0 12px}
#wrap{width:75vw;max-width:100%;line-height:0}
canvas{cursor:crosshair;border:1px solid #444;display:block;width:100%;height:auto}
</style></head><body>
<header>
<h2>Object labeling</h2>
<p id="videoLabel">Video __VIDEO_NUM__ / __VIDEO_TOTAL__: __VIDEO_ID__</p>
<p>Left click = object · Right click = background · <kbd>1</kbd>-<kbd>9</kbd> select object · <kbd>o</kbd> new object · <kbd>x</kbd>/<kbd>Del</kbd> delete object · <kbd>l</kbd> save/lock object · <kbd>m</kbd> edit object · <kbd>A</kbd>/<kbd>D</kbd> = ±1 frame · <kbd>Q</kbd>/<kbd>E</kbd> = ±5 frames · <kbd>u</kbd> undo · <kbd>c</kbd> clear · <kbd>s</kbd> save/next</p>
<p id="previewNote" style="display:__PREVIEW_DISPLAY__">Blue overlay = active SAM2 preview. Colored overlays = locally saved/locked object previews on their prompt frames.</p>
</header>
<div class="toolbar">
<button class="pos" id="modeObj">Object (+)</button>
<button class="neg" id="modeBg">Background (-)</button>
<button id="undoBtn">Undo</button>
<button id="clearBtn">Clear all</button>
<button id="saveBtn">Save &amp; next</button>
</div>
<div class="objectbar">
<label>Object <select id="objectSelect"></select></label>
<button id="newObjectBtn">New object</button>
<button id="deleteObjectBtn">Delete object</button>
<button id="lockObjectBtn">Save object</button>
<button id="editObjectBtn">Edit object</button>
<span id="objectStatus"></span>
</div>
<div class="legend">
<span><kbd>1</kbd>-<kbd>9</kbd> select object</span>
<span><kbd>o</kbd> new object</span>
<span><kbd>x</kbd>/<kbd>Del</kbd> delete object</span>
<span><kbd>l</kbd> save/lock object</span>
<span><kbd>m</kbd> edit object</span>
<span><kbd>s</kbd> save &amp; next video</span>
<span><kbd>a</kbd>/<kbd>d</kbd> ±1 frame</span>
<span><kbd>q</kbd>/<kbd>e</kbd> ±5 frames</span>
<span><kbd>p</kbd>/<kbd>n</kbd> object/background</span>
<span><kbd>u</kbd> undo</span>
<span><kbd>c</kbd> clear</span>
</div>
<div class="framebar">
<button id="prevBtn" title="Previous frame A">◀</button>
<input type="range" id="slider" min="0" max="__MAX_FRAME__" value="__FRAME_IDX__">
<button id="nextBtn" title="Next frame D">▶</button>
<span id="frameLabel">__FRAME_IDX__ / __MAX_FRAME__</span>
</div>
<div id="status">Loading…</div>
<div id="stage"><div id="wrap"><canvas id="c" width="320" height="180"></canvas></div></div>
<script>
let mode=1, points=[], labels=[], img=new Image(), maskImg=new Image(), maskReady=false, previewSeq=0;
let objects=[], activeObjectIndex=0;
const lockedMaskImgs=new Map();
const objectMaskColors=[[80,220,80],[255,120,40],[220,80,220],[40,220,220],[180,180,60],[255,80,120],[120,180,255],[160,100,40],[210,210,210]];
let currentFrame=__FRAME_IDX__, maxFrame=__MAX_FRAME__, videoIndex=__VIDEO_INDEX__, videoTotal=__VIDEO_TOTAL__, videoId="__VIDEO_ID__";
const c=document.getElementById('c'), ctx=c.getContext('2d'), statusEl=document.getElementById('status');
const slider=document.getElementById('slider'), frameLabel=document.getElementById('frameLabel');
const videoLabel=document.getElementById('videoLabel');
const objectSelect=document.getElementById('objectSelect'), objectStatus=document.getElementById('objectStatus');
function setMode(m){ mode=m; document.getElementById('modeObj').style.outline=m? '2px solid #fff':'none';
  document.getElementById('modeBg').style.outline=m? 'none':'2px solid #fff'; }
function layoutCanvas(){
  document.getElementById('wrap').style.width='75vw';
  c.style.width='100%';
  c.style.height='auto';
}
function makeObject(idx){
  return {object_id:'object_'+idx, frame_idx:currentFrame, points:[], labels:[], locked:false, name:null};
}
function objectColor(idx){ return objectMaskColors[idx % objectMaskColors.length]; }
function invalidateActivePreview(){
  const obj=activeObject();
  delete obj.preview_mask_b64;
  delete obj.preview_frame_idx;
  lockedMaskImgs.delete(obj.object_id);
}
function nextObjectIndex(){
  const used=new Set(objects.map(obj=>{
    const m=String(obj.object_id||'').match(/^object_(\\d+)$/);
    return m? parseInt(m[1],10): -1;
  }));
  let idx=0;
  while(used.has(idx)) idx++;
  return idx;
}
function ensureObjects(){
  if(!objects.length){ objects=[makeObject(0)]; activeObjectIndex=0; }
}
function activeObject(){ ensureObjects(); return objects[activeObjectIndex]; }
function syncActiveObject(){
  const obj=activeObject();
  obj.frame_idx=currentFrame;
  obj.points=points.map(p=>[p[0],p[1]]);
  obj.labels=labels.map(v=>v);
}
function loadActiveObject(){
  const obj=activeObject();
  points=(obj.points||[]).map(p=>[p[0],p[1]]);
  labels=(obj.labels||[]).map(v=>v);
  currentFrame=Math.max(0, Math.min(maxFrame, parseInt(obj.frame_idx||0,10)));
  updateFrameLabel(currentFrame);
  maskReady=false; previewSeq++;
}
function updateObjectUI(){
  ensureObjects();
  objectSelect.innerHTML='';
  objects.forEach((obj, idx)=>{
    const opt=document.createElement('option');
    opt.value=idx;
    opt.textContent=(idx+1)+': '+obj.object_id+(obj.locked?' ✓':' *');
    objectSelect.appendChild(opt);
  });
  objectSelect.value=activeObjectIndex;
  const obj=activeObject();
  objectStatus.textContent=(obj.locked?'locked':'editing')+' · '+points.length+' point(s) · frame '+currentFrame;
}
function selectObject(idx){
  if(idx<0 || idx>=objects.length) return;
  syncActiveObject();
  activeObjectIndex=idx;
  loadActiveObject();
  updateObjectUI();
  loadFrame(currentFrame,{keepPoints:true});
}
function newObject(){
  syncActiveObject();
  objects.push(makeObject(nextObjectIndex()));
  activeObjectIndex=objects.length-1;
  points=[]; labels=[]; currentFrame=Math.max(0, Math.min(maxFrame, currentFrame));
  activeObject().frame_idx=currentFrame;
  maskReady=false; previewSeq++; updateObjectUI(); draw();
}
function deleteActiveObject(){
  ensureObjects();
  const obj=activeObject();
  const label=obj.object_id||('object '+(activeObjectIndex+1));
  if(!confirm('Delete '+label+'?')) return;
  lockedMaskImgs.delete(obj.object_id);
  objects.splice(activeObjectIndex,1);
  if(!objects.length){
    objects=[makeObject(0)];
    activeObjectIndex=0;
  }else{
    activeObjectIndex=Math.max(0, Math.min(activeObjectIndex, objects.length-1));
  }
  loadActiveObject();
  updateObjectUI();
  loadFrame(currentFrame,{keepPoints:true});
  statusEl.textContent='Deleted '+label;
}
async function lockActiveObject(){
  if(!points.length || !labels.some(v=>v)){ alert('Add at least one object click for this object'); return; }
  syncActiveObject();
  if(__MASK_PREVIEW__ && !maskReady){
    statusEl.textContent='Computing saved object preview…';
    await refreshPreview();
  }
  activeObject().locked=true;
  updateObjectUI();
  statusEl.textContent=activeObject().object_id+' saved locally';
  draw();
}
function editActiveObject(){
  activeObject().locked=false;
  updateObjectUI();
  statusEl.textContent='Editing '+activeObject().object_id;
}
function drawTintedMask(maskImage, color, alpha){
  if(!maskImage || !maskImage.complete || !maskImage.naturalWidth) return;
  const off=document.createElement('canvas');
  off.width=c.width; off.height=c.height;
  const octx=off.getContext('2d',{willReadFrequently:true});
  octx.drawImage(maskImage,0,0,c.width,c.height);
  const imgData=octx.getImageData(0,0,c.width,c.height);
  const d=imgData.data;
  for(let i=0;i<d.length;i+=4){
    if(d[i+3]>127){
      d[i]=color[0]; d[i+1]=color[1]; d[i+2]=color[2]; d[i+3]=alpha;
    } else {
      d[i+3]=0;
    }
  }
  octx.putImageData(imgData,0,0);
  ctx.drawImage(off,0,0);
}
function getLockedMaskImage(obj){
  if(!obj.preview_mask_b64) return null;
  const cached=lockedMaskImgs.get(obj.object_id);
  if(cached && cached.src.endsWith(obj.preview_mask_b64)) return cached;
  const im=new Image();
  im.onload=()=>draw();
  im.src='data:image/png;base64,'+obj.preview_mask_b64;
  lockedMaskImgs.set(obj.object_id, im);
  return im;
}
function drawLockedObjectMasks(){
  objects.forEach((obj, idx)=>{
    if(!obj.locked || idx===activeObjectIndex || obj.preview_frame_idx!==currentFrame) return;
    drawTintedMask(getLockedMaskImage(obj), objectColor(idx), 125);
  });
}
function drawMaskOverlay(){
  drawLockedObjectMasks();
  const active=activeObject();
  if(active.locked && active.preview_mask_b64 && active.preview_frame_idx===currentFrame){
    drawTintedMask(getLockedMaskImage(active), objectColor(activeObjectIndex), 145);
    return;
  }
  if(maskReady) drawTintedMask(maskImg, [40,120,255], 140);
}
function drawPoints(){
  for(let i=0;i<points.length;i++){
    const [x,y]=points[i]; ctx.beginPath(); ctx.arc(x,y,6,0,Math.PI*2);
    ctx.fillStyle=labels[i]? '#2ecc71':'#e74c3c'; ctx.fill(); ctx.strokeStyle='#fff'; ctx.stroke();
  }
}
function draw(){
  if(!img.complete || !img.naturalWidth) return;
  ctx.drawImage(img,0,0,c.width,c.height);
  drawMaskOverlay();
  drawPoints();
}
function clearPoints(){
  if(activeObject().locked){ statusEl.textContent='Object is locked; press Edit object first'; return; }
  points=[]; labels=[]; maskReady=false; previewSeq++; invalidateActivePreview(); draw();
  syncActiveObject(); updateObjectUI();
}
function updateVideoLabel(){
  videoLabel.textContent='Video '+(videoIndex+1)+' / '+videoTotal+': '+videoId;
}
function updateFrameLabel(idx){ frameLabel.textContent=idx+' / '+maxFrame; slider.value=idx; }
function applyVideoMeta(meta){
  videoIndex=meta.video_index; videoTotal=meta.video_total; videoId=meta.video_id;
  maxFrame=meta.max_frame; currentFrame=meta.frame_idx;
  slider.max=maxFrame; slider.value=currentFrame;
  objects=[makeObject(0)]; activeObjectIndex=0; points=[]; labels=[];
  lockedMaskImgs.clear();
  updateVideoLabel(); updateFrameLabel(currentFrame);
  updateObjectUI();
}
async function refreshPreview(){
  if(!__MASK_PREVIEW__) return;
  if(!points.length || !labels.some(v=>v)){
    maskReady=false; draw(); return;
  }
  const seq=++previewSeq;
  statusEl.textContent='Computing mask preview…';
  try{
    const r=await fetch('/preview',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({frame_idx:currentFrame,points,labels})});
    const data=await r.json();
    if(seq!==previewSeq) return;
    if(!r.ok) throw new Error(data.error||('HTTP '+r.status));
    if(!data.has_mask){
      maskReady=false; draw();
      statusEl.textContent='No mask detected — try another object point';
      return;
    }
    await new Promise((resolve, reject)=>{
      maskImg.onload=()=>resolve();
      maskImg.onerror=()=>reject(new Error('decode failed'));
      maskImg.src='data:image/png;base64,'+data.mask_b64;
    });
    if(seq!==previewSeq) return;
    activeObject().preview_mask_b64=data.mask_b64;
    activeObject().preview_frame_idx=currentFrame;
    maskReady=true; draw();
    statusEl.textContent='Frame '+currentFrame+' — click to label';
  }catch(err){
    if(seq!==previewSeq) return;
    maskReady=false; draw();
    statusEl.textContent='Preview failed: '+err.message;
  }
}
async function loadFrame(idx, opts={}){
  idx=Math.max(0, Math.min(maxFrame, idx));
  if(!opts.keepPoints) syncActiveObject();
  currentFrame=idx; updateFrameLabel(idx);
  maskReady=false; previewSeq++;
  statusEl.textContent='Loading frame '+idx+'…';
  try{
    const r=await fetch('/frame?idx='+idx+'&_='+Date.now());
    if(!r.ok) throw new Error('HTTP '+r.status);
    const blob=await r.blob();
    if(!blob.size) throw new Error('empty image');
    const url=URL.createObjectURL(blob);
    await new Promise((resolve, reject)=>{
      img.onload=()=>{ URL.revokeObjectURL(url); resolve(); };
      img.onerror=()=>{ URL.revokeObjectURL(url); reject(new Error('decode failed')); };
      img.src=url;
    });
    c.width=img.naturalWidth; c.height=img.naturalHeight;
    layoutCanvas();
    if(!opts.keepPoints){
      points=[]; labels=[]; activeObject().frame_idx=currentFrame; activeObject().points=[]; activeObject().labels=[];
      invalidateActivePreview();
    }
    updateObjectUI(); draw();
    statusEl.textContent='Frame '+idx+' — click to label';
  }catch(err){
    statusEl.textContent='Failed to load frame '+idx+': '+err.message;
  }
}
function stepFrame(delta){
  if(activeObject().locked){ statusEl.textContent='Object is locked; press Edit object first'; return; }
  loadFrame(currentFrame+delta);
}
c.addEventListener('contextmenu',e=>e.preventDefault());
c.addEventListener('mousedown',e=>{
  if(activeObject().locked){ statusEl.textContent='Object is locked; press Edit object first'; return; }
  const r=c.getBoundingClientRect();
  const x=(e.clientX-r.left)*c.width/r.width, y=(e.clientY-r.top)*c.height/r.height;
  points.push([x,y]); labels.push(e.button===0? mode:0); maskReady=false; previewSeq++; invalidateActivePreview(); draw();
  syncActiveObject(); updateObjectUI();
  refreshPreview();
});
document.getElementById('modeObj').onclick=()=>setMode(1);
document.getElementById('modeBg').onclick=()=>setMode(0);
document.getElementById('undoBtn').onclick=()=>{ if(activeObject().locked){ statusEl.textContent='Object is locked; press Edit object first'; return; } if(points.length){ points.pop(); labels.pop(); maskReady=false; previewSeq++; invalidateActivePreview(); draw(); syncActiveObject(); updateObjectUI(); refreshPreview(); }};
document.getElementById('clearBtn').onclick=()=>clearPoints();
document.getElementById('prevBtn').onclick=()=>stepFrame(-1);
document.getElementById('nextBtn').onclick=()=>stepFrame(1);
document.getElementById('newObjectBtn').onclick=()=>newObject();
document.getElementById('deleteObjectBtn').onclick=()=>deleteActiveObject();
document.getElementById('lockObjectBtn').onclick=()=>lockActiveObject();
document.getElementById('editObjectBtn').onclick=()=>editActiveObject();
objectSelect.addEventListener('change',()=>selectObject(parseInt(objectSelect.value,10)));
slider.addEventListener('input', ()=>updateFrameLabel(parseInt(slider.value,10)));
slider.addEventListener('change', ()=>loadFrame(parseInt(slider.value,10)));
document.getElementById('saveBtn').onclick=()=>{
  syncActiveObject();
  for(const obj of objects){
    if(!(obj.points||[]).length || !(obj.labels||[]).some(v=>v)){ alert('Object '+obj.object_id+' needs at least one positive object click'); return; }
  }
  fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({schema_version:'sam2_object_prompt_v2',objects})})
    .then(r=>r.json()).then(d=>{
      if(!d.ok){ alert(d.message||'save failed'); return; }
      if(d.done){ alert(d.message||'All videos labeled'); window.close(); return; }
      applyVideoMeta(d.next);
      clearPoints();
      loadFrame(currentFrame);
      statusEl.textContent=d.message||('Saved. Loaded '+videoId);
    });
};
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT') return;
  const key=e.key.toLowerCase();
  if(e.key==='ArrowLeft' || e.key==='[' || key==='a') stepFrame(-1);
  else if(e.key==='ArrowRight' || e.key===']' || key==='d') stepFrame(1);
  else if(key==='q') stepFrame(-5);
  else if(key==='e') stepFrame(5);
  else if(key==='u'){ document.getElementById('undoBtn').click(); }
  else if(key==='c') clearPoints();
  else if(key==='s') document.getElementById('saveBtn').click();
  else if(key==='o') newObject();
  else if(key==='x' || e.key==='Delete' || e.key==='Backspace'){ e.preventDefault(); deleteActiveObject(); }
  else if(key==='l') lockActiveObject();
  else if(key==='m') editActiveObject();
  else if(/^[1-9]$/.test(key)) selectObject(parseInt(key,10)-1);
  else if(key==='p') setMode(1);
  else if(key==='n') setMode(0);
});
window.addEventListener('resize', layoutCanvas);
setMode(1); ensureObjects(); layoutCanvas(); updateVideoLabel(); updateObjectUI(); loadFrame(currentFrame,{keepPoints:true});
</script></body></html>
"""


def _render_label_page(
    frame_idx: int,
    max_frame: int,
    *,
    mask_preview: bool = True,
    video_id: str,
    video_index: int,
    video_total: int,
) -> str:
    """Inject frame metadata without str.format (CSS/JS contain literal braces)."""
    return (
        HTML_PAGE.replace("__FRAME_IDX__", str(frame_idx))
        .replace("__MAX_FRAME__", str(max_frame))
        .replace("__MASK_PREVIEW__", "true" if mask_preview else "false")
        .replace("__PREVIEW_DISPLAY__", "block" if mask_preview else "none")
        .replace("__VIDEO_ID__", video_id)
        .replace("__VIDEO_INDEX__", str(video_index))
        .replace("__VIDEO_NUM__", str(video_index + 1))
        .replace("__VIDEO_TOTAL__", str(video_total))
    )


class _LabelState:
    def __init__(
        self,
        jobs: list[VideoJob],
        default_frame: int,
        *,
        mask_preview: bool,
        predictor,
        sam2_session: dict,
    ):
        if not jobs:
            raise ValueError("At least one video job is required")
        self.jobs = jobs
        self.job_index = 0
        self.default_frame = default_frame
        self.frame_idx = default_frame
        self.num_frames = 0
        self.mask_preview = mask_preview
        self.predictor = predictor
        self.sam2_session = sam2_session
        self.lock = threading.Lock()
        self.done = False
        self._load_current_video()

    @property
    def job(self) -> VideoJob:
        return self.jobs[self.job_index]

    @property
    def video_path(self) -> Path:
        return resolve_repo_path(self.job.video_path)

    @property
    def step_dir(self) -> Path:
        return interim_step_dir(self.job.dataset, self.job.video_id, "sam2_object")

    @property
    def max_frame(self) -> int:
        return max(0, self.num_frames - 1)

    def _load_current_video(self) -> None:
        self.num_frames = count_video_frames(self.video_path)
        self.frame_idx = max(0, min(self.default_frame, self.max_frame))
        self.step_dir.mkdir(parents=True, exist_ok=True)

    def meta(self) -> dict[str, Any]:
        return {
            "video_index": self.job_index,
            "video_total": len(self.jobs),
            "video_id": self.job.video_id,
            "video": str(self.video_path),
            "frame_idx": self.frame_idx,
            "num_frames": self.num_frames,
            "max_frame": self.max_frame,
            "mask_preview": self.mask_preview,
        }

    def advance(self) -> bool:
        self.job_index += 1
        if self.job_index >= len(self.jobs):
            self.done = True
            return False
        _close_sam2_session(self.predictor, self.sam2_session)
        self.sam2_session["_lock"] = threading.Lock()
        self._load_current_video()
        return True


def _try_open_browser(url: str) -> None:
    """Best-effort browser launch after the HTTP server is listening."""
    import shutil
    import subprocess

    if shutil.which("xdg-open"):
        try:
            subprocess.Popen(
                ["xdg-open", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except OSError:
            pass
    try:
        if webbrowser.open(url, new=2):
            return
    except Exception as exc:
        print(f"Could not open browser automatically: {exc}", flush=True)


def _close_sam2_session(predictor, session_cache: dict) -> None:
    state = session_cache.get("sam2_state")
    if predictor is not None and state is not None and hasattr(predictor, "reset_state"):
        try:
            predictor.reset_state(state)
        except Exception:
            pass
    session_cache.clear()


def _compute_mask_preview(
    predictor,
    session_cache: dict,
    *,
    video_path: Path,
    frame_idx: int,
    points: list[tuple[float, float]],
    labels: list[int],
) -> tuple[Any, Any | None]:
    frame = read_video_frame(video_path, frame_idx)
    if predictor is None or not points or not any(labels):
        return frame, None
    with session_cache.get("_lock", threading.Lock()):
        mask = preview_object_mask_on_frame(
            predictor,
            video_path=video_path,
            frame_idx=frame_idx,
            points=points,
            labels=labels,
            session_cache=session_cache,
        )
    return frame, mask


def _start_sam2_prewarm(state: "_LabelState") -> None:
    if not state.mask_preview or state.predictor is None:
        return

    def _prewarm() -> None:
        try:
            video_path = state.video_path
            video_id = state.job.video_id
            with state.lock:
                ensure_object_preview_video_state(
                    state.predictor,
                    video_path=video_path,
                    session_cache=state.sam2_session,
                )
            print(f"SAM2 preview state ready for {video_id}", flush=True)
        except Exception as exc:  # noqa: BLE001 - preview can still initialize on click and report errors there
            print(f"SAM2 preview prewarm failed: {exc}", flush=True)

    threading.Thread(target=_prewarm, daemon=True).start()


def _run_http(
    jobs: list[VideoJob],
    frame_idx: int,
    host: str,
    port: int,
    no_browser: bool,
    *,
    mask_preview: bool,
    gpu_id: int,
    preview_device: str,
    sam2_checkpoint: Path,
    sam2_model_cfg: str,
) -> LabelPrompt:
    import cv2

    sam2_session: dict[str, Any] = {"_lock": threading.Lock()}
    state = _LabelState(
        jobs,
        frame_idx,
        mask_preview=mask_preview,
        predictor=None,
        sam2_session=sam2_session,
    )
    result: dict[str, LabelPrompt | None] = {"prompt": None}
    predictor: Any = None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _fmt, *_args):
            return

        def _send_json(self, code: int, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                page = _render_label_page(
                    state.frame_idx,
                    state.max_frame,
                    mask_preview=state.mask_preview,
                    video_id=state.job.video_id,
                    video_index=state.job_index,
                    video_total=len(state.jobs),
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            if parsed.path == "/meta":
                self._send_json(200, state.meta())
                return
            if parsed.path == "/health":
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "video_id": state.job.video_id,
                        "video_index": state.job_index,
                        "video_total": len(state.jobs),
                        "frame_idx": state.frame_idx,
                        "max_frame": state.max_frame,
                        "mask_preview": state.mask_preview,
                        "sam2_ready": state.predictor is not None,
                    },
                )
                return
            if parsed.path == "/frame":
                qs = parse_qs(parsed.query)
                try:
                    idx = int(qs.get("idx", [state.frame_idx])[0])
                except ValueError:
                    self.send_error(400, "bad idx")
                    return
                idx = max(0, min(state.max_frame, idx))
                try:
                    frame = read_video_frame(state.video_path, idx)
                    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    if not ok:
                        self.send_error(500, "encode failed")
                        return
                    payload = buf.tobytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception as exc:
                    print(f"[label_object] /frame idx={idx} failed: {exc}", flush=True)
                    self.send_error(500, str(exc))
                return
            self.send_error(404)

        def do_POST(self):
            if self.path == "/preview":
                if not state.mask_preview:
                    self.send_error(503, "mask preview disabled")
                    return
                if state.predictor is None:
                    self._send_json(
                        503,
                        {
                            "has_mask": False,
                            "error": "SAM2 still loading — wait a few seconds and click again",
                        },
                    )
                    return
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length).decode())
                try:
                    idx = int(data["frame_idx"])
                    raw_points = data.get("points", [])
                    raw_labels = data.get("labels", [])
                    points = [(float(p[0]), float(p[1])) for p in raw_points]
                    labels = [int(v) for v in raw_labels]
                except (KeyError, TypeError, ValueError):
                    self.send_error(400, "bad preview payload")
                    return
                idx = max(0, min(state.max_frame, idx))
                try:
                    with state.lock:
                        mask = preview_object_mask_on_frame(
                            state.predictor,
                            video_path=state.video_path,
                            frame_idx=idx,
                            points=points,
                            labels=labels,
                            session_cache=state.sam2_session,
                        )
                    payload = {
                        "has_mask": mask is not None,
                        "mask_b64": (
                            base64.b64encode(encode_object_mask_png(mask)).decode("ascii")
                            if mask is not None
                            else ""
                        ),
                    }
                    self._send_json(200, payload)
                except Exception as exc:
                    import traceback

                    print(f"[label_object] /preview idx={idx} failed: {exc}", flush=True)
                    traceback.print_exc()
                    self._send_json(500, {"has_mask": False, "error": str(exc)})
                return
            if self.path != "/save":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length).decode())
            prompt = LabelPrompt.from_json(data)
            saved_path = save_label_prompt(state.step_dir, prompt)
            result["prompt"] = prompt
            current_video_id = state.job.video_id
            has_next = state.advance()
            if has_next:
                _start_sam2_prewarm(state)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "done": False,
                        "message": f"Saved {current_video_id} to {saved_path}. Loaded {state.job.video_id}.",
                        "next": state.meta(),
                    },
                )
            else:
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "done": True,
                        "message": f"Saved {current_video_id} to {saved_path}. All videos labeled.",
                    },
                )

    server = ThreadingHTTPServer((host, port), Handler)
    browser_url = f"http://127.0.0.1:{port}/"
    print(f"Batch: {len(state.jobs)} video(s)", flush=True)
    print(f"Video 1/{len(state.jobs)}: {state.video_path} ({state.num_frames} frames)", flush=True)
    if mask_preview:
        print("Mask preview: enabled (loading SAM2 in background)", flush=True)
    else:
        print("Mask preview: disabled", flush=True)

    def _serve() -> None:
        server.serve_forever()

    def _load_predictor() -> None:
        nonlocal predictor
        if not no_browser:
            _try_open_browser(browser_url)
        print(f"Open labeling UI: {browser_url}", flush=True)
        if host == "0.0.0.0":
            print(
                "Remote access: ssh -L "
                f"{port}:127.0.0.1:{port} user@host",
                flush=True,
            )
        if mask_preview:
            device_label = "CPU" if preview_device == "cpu" else f"GPU {gpu_id}"
            print(f"Loading SAM2 on {device_label} for mask preview…", flush=True)
            predictor = build_object_predictor(
                gpu_id=gpu_id,
                device=preview_device,
                checkpoint=sam2_checkpoint,
                model_cfg=sam2_model_cfg,
                apply_postprocessing=preview_device == "cuda",
            )
            state.predictor = predictor
            print("SAM2 mask preview ready", flush=True)
            _start_sam2_prewarm(state)

    server_thread = threading.Thread(target=_serve, daemon=True)
    server_thread.start()
    loader_thread = threading.Thread(target=_load_predictor, daemon=True)
    loader_thread.start()

    try:
        while not state.done:
            time.sleep(0.1)
    finally:
        server.shutdown()
        server.server_close()
        _close_sam2_session(predictor, sam2_session)
        if predictor is not None:
            del predictor
    if result["prompt"] is None:
        raise RuntimeError("Save failed")
    return result["prompt"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-id", default=None, help="Video id (HOI4D __ name)")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--video-list", type=Path, default=None, help="Text/JSON list of videos or ids for batch labeling")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument(
        "--label-mode",
        choices=("headed", "http"),
        required=True,
        help="headed=local browser UI; http=browser UI for remote SSH",
    )
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id for SAM2 mask preview when --preview-device cuda")
    parser.add_argument(
        "--preview-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device for interactive SAM2 mask preview during labeling (default: cpu)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Alias for --preview-device cpu; keeps labeling preview off the GPU",
    )
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--sam2-model-cfg", default=DEFAULT_SAM2_MODEL_CFG)
    parser.add_argument(
        "--no-mask-preview",
        action="store_true",
        help="Disable SAM2 mask preview while labeling",
    )
    args = parser.parse_args(argv)

    if args.video_list is not None:
        jobs = discover_videos(
            args.dataset,
            dataset_root=args.dataset_root,
            video_list=args.video_list,
        )
    elif args.video is not None:
        from _common.dataset import video_job_from_path

        jobs = [video_job_from_path(args.dataset, args.video, video_id=args.video_id)]
    elif args.video_id is not None:
        jobs = [resolve_video_job(args.dataset, args.video_id, dataset_root=args.dataset_root)]
    else:
        parser.error("Provide --video-list, --video, or --video-id")

    mask_preview = not args.no_mask_preview
    preview_device = "cpu" if args.cpu_only else args.preview_device

    if args.label_mode == "headed":
        print("Headed mode uses the same local browser UI as HTTP mode.", flush=True)
        _run_http(
            jobs,
            args.frame_idx,
            "127.0.0.1",
            args.http_port,
            args.no_browser,
            mask_preview=mask_preview,
            gpu_id=args.gpu,
            preview_device=preview_device,
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_model_cfg=args.sam2_model_cfg,
        )
    else:
        _run_http(
            jobs,
            args.frame_idx,
            args.http_host,
            args.http_port,
            args.no_browser,
            mask_preview=mask_preview,
            gpu_id=args.gpu,
            preview_device=preview_device,
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_model_cfg=args.sam2_model_cfg,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
