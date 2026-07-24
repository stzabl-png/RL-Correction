#!/usr/bin/env python3
"""
annotate_grasp_frames.py — 手动标注视频里左右手的抓取区间(浏览器 GUI)

给一个 MP4(或含 image.mp4 / 散帧的目录),在浏览器里逐帧标注左右手各自的
「抓取开始帧 / 抓取结束帧」。每只手可以有多段抓取,严格 抓-放-抓-放 交替
(Space 自动切换,天然不可能连续两个抓或两个放)。

启动:
  # 单条
  python tools/annotate_grasp_frames.py <mp4或目录> [--port 8770] [--out xxx.json] [--no-browser]
  # 多条队列(S 保存后自动进入下一条);多条必须用 --out-dir,每条输出到各自 take 目录,
  # 避免同目录视频的 grasp_annotation.json 互相覆盖:
  python tools/annotate_grasp_frames.py A/0.mp4 A/1.mp4 ... --out-dir <ReconstructOutput/.../父目录>
  #   -> 每条输出 <out-dir>/<视频名去后缀>/grasp_annotation.json

快捷键(浏览器窗口聚焦后):
  ← / →        上一帧 / 下一帧          ↑ / ↓  快退/快进 10 帧
  Shift        切换当前手(左/右)
  Space        打点:先按=抓取开始帧,再按=抓取结束帧(交替)
  Z / Backspace 撤销当前手最后一次打点
  Home / End   跳到首帧 / 末帧
  S            保存 + 自动进入下一条(队列模式);最后一条则提示完成
  [ / ]        上一条 / 下一条视频(保存当前)

输出 JSON(默认存在视频目录下 grasp_annotation.json):
  {"video":..., "num_frames":N, "fps":..,
   "annotations": {"left": [[s,e],...], "right": [[s,e],...]}}
  其中每个 [s,e] = 一段抓取的 [开始帧, 结束帧](含端点)。
"""
import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2
try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted


def _resolve_input(path: Path):
    """把输入解析成 (frames: List[jpeg bytes], fps, name)。"""
    path = path.resolve()
    frames, fps = [], 15.0
    exts = {".jpg", ".jpeg", ".png"}
    if path.is_dir():
        cand = path / "image.mp4"
        if cand.is_file():
            path = cand
    if path.is_file() and (path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}):
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            ok, buf = cv2.imencode(".jpg", fr)
            if ok:
                frames.append(buf.tobytes())
        cap.release()
    elif path.is_dir():
        files = natsorted([p for p in path.iterdir() if p.suffix.lower() in exts])
        for fp in files:
            frames.append(fp.read_bytes())
    if not frames:
        raise SystemExit(f"没读到任何帧: {path}")
    return frames, float(fps), (path.parent.name + "/" + path.name)


def _load_existing(out_path: Path):
    try:
        return json.loads(Path(out_path).read_text(encoding="utf-8")).get("annotations")
    except Exception:
        return None


class Session:
    """一个标注队列:多条 (视频, 输出json),当前指针,懒加载帧。"""

    def __init__(self, jobs, start=0):
        self.jobs = jobs                      # list[(Path video, Path out)]
        self.lock = threading.Lock()
        self._cache = {}                      # i -> (frames, fps, name)
        self.i = -1
        self.frames, self.fps, self.name, self.out_path, self.existing = [], 15.0, "", None, None
        self.load(max(0, min(len(jobs) - 1, start)))

    def load(self, i):
        i = max(0, min(len(self.jobs) - 1, i))
        with self.lock:
            self.i = i
            video, out = self.jobs[i]
            if i not in self._cache:
                self._cache[i] = _resolve_input(video)
            self.frames, self.fps, self.name = self._cache[i]
            self.out_path = out
            self.existing = _load_existing(out)

    def save(self, annotations):
        with self.lock:
            payload = {"video": self.name, "num_frames": len(self.frames),
                       "fps": self.fps, "annotations": annotations}
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self.out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
            self.existing = annotations

    def has_next(self):
        return self.i < len(self.jobs) - 1

    def has_prev(self):
        return self.i > 0

    def frame(self, i):
        with self.lock:
            i = max(0, min(len(self.frames) - 1, i))
            return self.frames[i]

    def meta(self):
        with self.lock:
            return {"num_frames": len(self.frames), "fps": self.fps, "name": self.name,
                    "annotations": self.existing, "index": self.i, "total": len(self.jobs),
                    "has_next": self.has_next(), "has_prev": self.has_prev(),
                    "out": str(self.out_path)}


SESSION: Session = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, "application/json", json.dumps(obj).encode())

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif u.path == "/meta":
            self._json(SESSION.meta())
        elif u.path == "/frame":
            i = int(parse_qs(u.query).get("i", ["0"])[0])
            self._send(200, "image/jpeg", SESSION.frame(i))
        else:
            self._send(404, "text/plain", b"not found")

    def _read_annotations(self):
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        return data.get("annotations", {"left": [], "right": []})

    def do_POST(self):
        p = urlparse(self.path).path
        if p == "/save":
            SESSION.save(self._read_annotations())
            self._json({"status": "ok"})
        elif p in ("/next", "/prev"):
            SESSION.save(self._read_annotations())          # 先存当前
            if p == "/next":
                if SESSION.has_next():
                    SESSION.load(SESSION.i + 1)
                else:
                    self._json({"done": True, **SESSION.meta()})
                    return
            else:
                if SESSION.has_prev():
                    SESSION.load(SESSION.i - 1)
            self._json(SESSION.meta())
        else:
            self._send(404, "text/plain", b"not found")


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>grasp 标注</title>
<style>
  body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}
  #bar{position:sticky;top:0;background:#1a1a1a;padding:10px 14px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
  #wrap{position:relative;text-align:center}
  img{max-width:100%;max-height:calc(100vh - 90px);object-fit:contain}
  button{background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:6px;padding:6px 10px;cursor:pointer}
  button:hover{background:#3a3a3a}
  .hand{font-weight:700;padding:4px 10px;border-radius:6px}
  .left{background:#2563eb}.right{background:#16a34a}
  #queue{font-weight:700;color:#fb0}
  #segbar{position:relative;height:26px;background:#000;margin:6px 14px;border-radius:4px;overflow:hidden}
  .seg{position:absolute;top:0;height:100%;opacity:.75}
  .seg.left{background:#2563eb}.seg.right{background:#16a34a}
  #cur{position:absolute;top:0;height:100%;width:2px;background:#fff}
  #toast{position:fixed;left:50%;bottom:30px;transform:translateX(-50%);background:#222;border:1px solid #555;padding:8px 16px;border-radius:6px;opacity:0;transition:opacity .3s;pointer-events:none}
  #help{font-size:12px;color:#999;margin-left:8px}
</style></head><body>
<div id="bar">
  <span id="queue">0 / 0</span>
  <span id="vid" style="color:#9cf"></span>
  <span class="hand left" id="hand">左手 LEFT</span>
  <span id="pos">0 / 0</span>
  <button onclick="nav('/prev')">[ 上条</button>
  <button onclick="j(-10)">⏪10</button><button onclick="j(-1)">◀</button>
  <button onclick="j(1)">▶</button><button onclick="j(10)">10⏩</button>
  <button onclick="nav('/next')">下条 ]</button>
  <span id="help">Space=抓/放  Shift=换手  Z=撤销  S=保存+下一条</span>
  <span id="stat" style="margin-left:auto;color:#6f6"></span>
</div>
<div id="segbar"></div>
<div id="wrap"><img id="im" src="/frame?i=0"><div id="cur" style="left:0"></div></div>
<div id="toast"></div>
<script>
let N=0,fps=15,cur=0,hand="left",segs={left:[],right:[]};
function toast(m){let t=document.getElementById("toast");t.textContent=m;t.style.opacity=1;clearTimeout(t._t);t._t=setTimeout(()=>t.style.opacity=0,1100);}
function draw(){
  document.getElementById("pos").textContent=cur+" / "+(N-1);
  document.getElementById("cur").style.left=(cur/(N-1||1)*100)+"%";
  let sb=document.getElementById("segbar");sb.querySelectorAll(".seg").forEach(e=>e.remove());
  for(let h of ["left","right"])for(let s of segs[h]){
    let a=s[0],b=s[1]==null?s[0]:s[1];
    let d=document.createElement("div");d.className="seg "+h;
    d.style.left=(a/(N-1||1)*100)+"%";d.style.width=((b-a)/(N-1||1)*100||0.5)+"%";
    sb.appendChild(d);
  }
}
function show(){document.getElementById("im").src="/frame?i="+cur;draw();}
function setHand(h){hand=h;let e=document.getElementById("hand");e.className="hand "+h;e.textContent=h==="left"?"左手 LEFT":"右手 RIGHT";}
function j(d){cur=Math.max(0,Math.min(N-1,cur+d));show();}
function payload(){let out={left:[],right:[]};for(let h of ["left","right"])for(let s of segs[h])if(s[1]!=null)out[h].push([s[0],s[1]]);return out;}
function apply(m){
  N=m.num_frames;fps=m.fps;cur=0;segs={left:[],right:[]};
  if(m.annotations){segs.left=(m.annotations.left||[]).map(x=>[x[0],x[1]]);segs.right=(m.annotations.right||[]).map(x=>[x[0],x[1]]);}
  document.getElementById("queue").textContent=(m.index+1)+" / "+m.total;
  document.getElementById("vid").textContent=m.name;
  document.title=(m.index+1)+"/"+m.total+" "+m.name;
  setHand("left");show();
}
function save(silent){
  return fetch("/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({annotations:payload()})})
    .then(()=>{if(!silent)toast("已保存 ✔");});
}
function nav(where){
  fetch(where,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({annotations:payload()})})
    .then(r=>r.json()).then(m=>{
      if(m.done){toast("✅ 已保存,全部标注完成!");document.getElementById("stat").textContent="全部完成";return;}
      apply(m);toast((where==="/next"?"→ ":"← ")+(m.index+1)+"/"+m.total+"  "+m.name);
    });
}
function grab(){
  let a=segs[hand];
  if(a.length&&a[a.length-1][1]==null){a[a.length-1][1]=cur;toast(hand+" 抓取结束 @"+cur);}
  else{a.push([cur,null]);toast(hand+" 抓取开始 @"+cur);}
  draw();save(true);
}
document.addEventListener("keydown",e=>{
  if(e.code==="Space"){e.preventDefault();grab();}
  else if(e.key==="Shift"){setHand(hand==="left"?"right":"left");}
  else if(e.key==="z"||e.key==="Z"||e.key==="Backspace"){e.preventDefault();let a=segs[hand];if(a.length){a.pop();draw();save(true);toast("撤销");}}
  else if(e.key==="ArrowLeft")j(-1);
  else if(e.key==="ArrowRight")j(1);
  else if(e.key==="ArrowUp")j(-10);
  else if(e.key==="ArrowDown")j(10);
  else if(e.key==="Home"){cur=0;show();}
  else if(e.key==="End"){cur=N-1;show();}
  else if(e.key==="s"||e.key==="S"){e.preventDefault();nav("/next");}
  else if(e.key==="[")nav("/prev");
  else if(e.key==="]")nav("/next");
});
window.addEventListener("resize",draw);
fetch("/meta").then(r=>r.json()).then(apply);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="一个或多个 MP4/目录;多个时配合 --out-dir")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--out", default=None, help="单个输入时的输出 JSON(默认:视频目录/grasp_annotation.json)")
    ap.add_argument("--out-dir", default=None,
                    help="多个输入时:每条输出到 <out-dir>/<视频名去后缀>/grasp_annotation.json")
    ap.add_argument("--start", type=int, default=0, help="从队列第几条开始(0-based)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    inputs = [Path(x) for x in args.inputs]
    if args.out_dir:
        out_dir = Path(args.out_dir)
        jobs = [(p, out_dir / p.stem / "grasp_annotation.json") for p in inputs]
    elif len(inputs) == 1:
        p = inputs[0]
        base = p if p.is_dir() else p.parent
        out = Path(args.out) if args.out else base / "grasp_annotation.json"
        jobs = [(p, out)]
    else:
        raise SystemExit("[grasp] 多个输入必须用 --out-dir(否则同目录视频的标注会互相覆盖)")

    global SESSION
    SESSION = Session(jobs, start=args.start)

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"[grasp] 队列 {len(jobs)} 条,从第 {SESSION.i + 1} 条开始")
    print(f"[grasp] 打开: {url}   (Ctrl-C 结束)")
    for k, (v, o) in enumerate(jobs):
        print(f"  {k+1:>3}. {v}  ->  {o}")
    if not args.no_browser:
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[grasp] 已退出")


if __name__ == "__main__":
    main()
