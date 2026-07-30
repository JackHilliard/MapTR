"""Minimal local web viewer for a finished MapTRv2 training run.

Shows, on one page, at http://127.0.0.1:<port>:
  - a loss/eval-metric curve (parsed from ALL of the run's *.log.json --
    mmcv writes a new timestamped log file every time training (re)starts,
    e.g. after a resume, so a single run's history can span several files)
  - TensorBoard, embedded (auto-launches `tensorboard --logdir` for you)
  - the CARLA BEV prediction-vs-GT gallery (PNGs from
    tools/maptrv2/carla_bev_vis.py, if you've run it against this work_dir)

Runs entirely outside the ML container -- only needs `flask`, `matplotlib`,
`seaborn` (only because it reuses analyze_logs.py's log parser, which
imports it), and `tensorboard`. Point-in-time only: reflects whatever's on
disk when the page loads/refreshes, no live-polling of an in-progress run.

Usage (run from the repo root, so the `tools.analysis_tools` import
resolves):
  pip install flask matplotlib seaborn tensorboard
  python3 tools/maptrv2/webviewer.py --work-dir work_dirs/<run>

Then open http://127.0.0.1:5000 (or wherever it's port-forwarded to).

Note: TensorBoard is launched as `sys.executable -m tensorboard.main`
(the same Python interpreter running this script), not the bare
`tensorboard` command off $PATH -- on a system with multiple Python
installs, a stale/system-packaged `tensorboard` binary can end up first on
$PATH and hang indefinitely on startup (observed in this project's own dev
environment: an apt-installed tensorboard hung with no useful error,
while `pip install tensorboard` into the interpreter actually running this
script fixed it immediately). Make sure `pip install tensorboard` targets
that same interpreter. Also: this script's own venv/container Python must
be the one with flask/tensorboard installed -- running a host-built venv
*inside* an apptainer container (or vice versa) silently breaks this,
since the venv's interpreter isn't visible inside the container's own
filesystem. If viewing remotely (SSH), both --port and --tb-port need to
be forwarded -- the TensorBoard iframe's src is resolved by your browser,
not the remote server, so forwarding only --port leaves it unable to load.
"""
import argparse
import atexit
import glob
import io
import os.path as osp
import socket
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')  # headless -- no display in a server process
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from flask import Flask, abort, jsonify, send_file

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from tools.analysis_tools.analyze_logs import load_json_logs  # noqa: E402

app = Flask(__name__)
STATE = {}  # populated in main() -- work_dir, tb_port, tb_log_path

# GitHub-dark-ish palette, shared between the page CSS and the matplotlib
# figure so the rendered plot doesn't look like a pasted-in white rectangle.
BG = '#0d1117'
BG_PANEL = '#161b22'
BORDER = '#30363d'
TEXT = '#c9d1d9'
TEXT_MUTED = '#8b949e'
ACCENT = '#58a6ff'
LOSS_RAW = '#3d5a80'
LOSS_SMOOTH = '#58a6ff'
EVAL_COLOR = '#3fb950'


def find_log_jsons(work_dir):
    """All *.log.json files, oldest first (filenames are
    YYYYMMDD_HHMMSS.log.json, so lexicographic sort == chronological)."""
    return sorted(glob.glob(osp.join(work_dir, '*.log.json')))


def find_config_file(work_dir):
    pys = [f for f in glob.glob(osp.join(work_dir, '*.py'))]
    return osp.basename(pys[0]) if pys else None


def find_checkpoints(work_dir):
    ckpts = sorted(glob.glob(osp.join(work_dir, '*.pth')))
    # best_*/*.pth -- see CLAUDE.md gotcha #2: metric names with a "/" in
    # them (e.g. CarlaMap_chamfer/mAP) create a real subdirectory here.
    ckpts += sorted(glob.glob(osp.join(work_dir, 'best_*', '*.pth')))
    return [osp.relpath(c, work_dir) for c in ckpts]


def find_vis_images(work_dir):
    return sorted(osp.basename(f)
                  for f in glob.glob(osp.join(work_dir, 'vis', '*.png')))


def merge_log_dicts(log_json_paths):
    """Merge every *.log.json in the work_dir into one {epoch: data} dict.
    Training resumes produce a new log file each time, and mmcv's epoch/iter
    counters correctly continue from the checkpoint (only the LR schedule
    formula has the resume gotcha -- see CLAUDE.md), so epoch numbers are
    directly comparable/mergeable across files.

    NOT a plain "later file wins" dict.update(): a resumed run's new log
    file also logs a near-empty bookkeeping entry (a checkpoint-save
    marker, e.g. {'epoch': 1, 'iter': 259} with no 'loss') for the epoch it
    resumed *from* -- verified directly against a real resume. Blindly
    preferring the later file would let that sparse marker silently
    replace the earlier file's complete data for that same epoch. Instead,
    for any epoch appearing in more than one file, keep whichever entry
    actually has more recorded iterations.
    """
    log_dicts = load_json_logs(log_json_paths)
    merged = {}
    for d in log_dicts:
        for epoch, data in d.items():
            existing = merged.get(epoch)
            if existing is None or len(data.get('iter', [])) >= len(existing.get('iter', [])):
                merged[epoch] = data
    return merged


def ema_smooth(ys, alpha=0.9):
    """Simple causal EMA smoothing, similar in spirit to TensorBoard's own
    scalar-smoothing slider."""
    smoothed = []
    avg = None
    for y in ys:
        avg = y if avg is None else alpha * avg + (1 - alpha) * y
        smoothed.append(avg)
    return smoothed


def build_loss_figure(log_json_paths):
    merged = merge_log_dicts(log_json_paths)
    epochs = sorted(merged.keys())
    if not epochs:
        return None

    with plt.style.context('dark_background'):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        fig.patch.set_facecolor(BG)

        # per-epoch iter count: take the max across epochs rather than just
        # the first, so one truncated/partial epoch (e.g. an interrupted
        # run) doesn't throw off the global-iteration x-axis for every
        # later epoch.
        iters_per_epoch = max(
            (d['iter'][-1] for d in merged.values() if d.get('iter')),
            default=0)

        # --- train loss vs global iteration, across ALL log files ---
        ax = axes[0]
        ax.set_facecolor(BG)
        xs, ys = [], []
        for epoch in epochs:
            d = merged[epoch]
            if 'loss' not in d:
                continue
            n = len(d['loss'])
            iters = d['iter'][:n]
            xs.extend([(epoch - 1) * iters_per_epoch + it for it in iters])
            ys.extend(d['loss'][:n])
        if xs:
            ax.plot(xs, ys, linewidth=0.6, color=LOSS_RAW, alpha=0.5,
                     label='raw')
            ax.plot(xs, ema_smooth(ys), linewidth=1.6, color=LOSS_SMOOTH,
                     label='smoothed')
            ax.legend(fontsize=7, facecolor=BG_PANEL, edgecolor=BORDER,
                       labelcolor=TEXT)
        ax.set_xlabel('iter', color=TEXT)
        ax.set_ylabel('loss', color=TEXT)
        ax.set_title('training loss (all logs)', color=TEXT)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color(BORDER)

        # --- eval metric vs epoch, across ALL log files ---
        ax = axes[1]
        ax.set_facecolor(BG)
        metric_key = None
        for epoch in epochs:
            for k in merged[epoch]:
                if k.endswith('/mAP'):
                    metric_key = k
                    break
            if metric_key:
                break
        if metric_key:
            xs, ys = [], []
            for epoch in epochs:
                vals = merged[epoch].get(metric_key)
                if vals:
                    xs.append(epoch)
                    ys.append(vals[-1])
            ax.plot(xs, ys, marker='o', markersize=4, color=EVAL_COLOR)
            ax.set_xlabel('epoch', color=TEXT)
            ax.set_ylabel(metric_key, color=TEXT)
            ax.set_title('eval metric (all logs)', color=TEXT)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
            ax.tick_params(colors=TEXT, labelsize=8)
            ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.6)
            for spine in ax.spines.values():
                spine.set_color(BORDER)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'no eval metric logged yet', ha='center',
                     va='center', transform=ax.transAxes, fontsize=9,
                     color=TEXT_MUTED)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, facecolor=BG)
        plt.close(fig)
        buf.seek(0)
        return buf


def tensorboard_is_up(port, timeout=0.5):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=timeout):
            return True
    except OSError:
        return False


PAGE_TEMPLATE = """<!doctype html>
<html><head>
<title>MapTRv2 run viewer</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{
    font-family: system-ui, sans-serif; max-width: 1150px; margin: 2em auto;
    padding: 0 1em; background: {bg}; color: {text};
  }}
  h1 {{ font-size: 1.3em; }}
  h2 {{
    font-size: 1.05em; margin-top: 2em; border-bottom: 1px solid {border};
    padding-bottom: 0.3em;
  }}
  a {{ color: {accent}; }}
  .meta {{ color: {text_muted}; font-size: 0.9em; }}
  .meta code {{ background: {bg_panel}; padding: 1px 4px; border-radius: 3px; }}
  img.plot {{ max-width: 100%; border-radius: 4px; }}
  .gallery {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .gallery figure {{
    margin: 0; width: 220px; background: {bg_panel}; border: 1px solid {border};
    border-radius: 4px; padding: 6px;
  }}
  .gallery img {{ width: 100%; display: block; border-radius: 2px; }}
  .gallery figcaption {{
    font-size: 0.75em; color: {text_muted}; margin-top: 4px;
    word-break: break-all; text-align: center;
  }}
  iframe {{ width: 100%; height: 700px; border: 1px solid {border}; border-radius: 4px; background: #fff; }}
  ul {{ font-size: 0.9em; }}
  code {{ color: {text}; }}
  #tb-warning {{
    display: none; background: #3d1f1f; border: 1px solid #f85149;
    border-radius: 4px; padding: 0.8em 1em; font-size: 0.9em;
  }}
</style>
</head><body>
<h1>MapTRv2 run: {work_dir}</h1>
<p class="meta">config: <code>{config}</code></p>

<h2>Loss / eval curve</h2>
{loss_section}

<h2>TensorBoard</h2>
<p class="meta">
  <a href="http://127.0.0.1:{tb_port}" target="_blank">http://127.0.0.1:{tb_port}</a>
  &mdash; if viewing this page over SSH, this port needs its own
  <code>-L {tb_port}:127.0.0.1:{tb_port}</code> forward (the iframe below is
  resolved by your browser, not the server this script runs on).
</p>
<div id="tb-warning">
  TensorBoard doesn't seem to be reachable on port {tb_port}. Check the
  server console output, or its log file: <code>{tb_log_path}</code>
</div>
<iframe id="tb-frame" src="http://127.0.0.1:{tb_port}"></iframe>

<h2>BEV predictions ({n_vis} images)</h2>
<div class="gallery">
{gallery}
</div>

<h2>Checkpoints</h2>
<ul>{ckpt_list}</ul>

<script>
async function checkTensorboard(attemptsLeft) {{
  try {{
    const resp = await fetch('/tb_status');
    const data = await resp.json();
    if (data.up) {{ return; }}
  }} catch (e) {{}}
  if (attemptsLeft <= 0) {{
    document.getElementById('tb-warning').style.display = 'block';
    return;
  }}
  setTimeout(() => checkTensorboard(attemptsLeft - 1), 2000);
}}
checkTensorboard(5);
</script>

</body></html>
"""


@app.route('/')
def index():
    work_dir = STATE['work_dir']
    log_jsons = find_log_jsons(work_dir)
    loss_section = ('<img class="plot" src="/plot.png">' if log_jsons
                     else '<p class="meta">no *.log.json found in this work_dir yet</p>')

    images = find_vis_images(work_dir)
    gallery = '\n'.join(
        f'<figure><a href="/vis/{f}" target="_blank"><img src="/vis/{f}"></a>'
        f'<figcaption>{f}</figcaption></figure>'
        for f in images
    ) or '<p class="meta">no images in vis/ -- run tools/maptrv2/carla_bev_vis.py against this work_dir first</p>'

    ckpts = find_checkpoints(work_dir)
    ckpt_list = '\n'.join(f'<li><code>{c}</code></li>' for c in ckpts) or '<li>none found</li>'

    return PAGE_TEMPLATE.format(
        bg=BG, bg_panel=BG_PANEL, border=BORDER, text=TEXT,
        text_muted=TEXT_MUTED, accent=ACCENT,
        work_dir=osp.abspath(work_dir),
        config=find_config_file(work_dir) or 'unknown',
        loss_section=loss_section,
        tb_port=STATE['tb_port'],
        tb_log_path=STATE['tb_log_path'],
        n_vis=len(images),
        gallery=gallery,
        ckpt_list=ckpt_list,
    )


@app.route('/plot.png')
def plot_png():
    log_jsons = find_log_jsons(STATE['work_dir'])
    if not log_jsons:
        abort(404)
    buf = build_loss_figure(log_jsons)
    if buf is None:
        abort(404)
    return send_file(buf, mimetype='image/png')


@app.route('/vis/<path:filename>')
def vis_image(filename):
    path = osp.join(STATE['work_dir'], 'vis', filename)
    if not osp.isfile(path):
        abort(404)
    return send_file(osp.abspath(path))


@app.route('/tb_status')
def tb_status():
    return jsonify(up=tensorboard_is_up(STATE['tb_port']))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir', required=True)
    p.add_argument('--port', type=int, default=5000)
    p.add_argument('--tb-port', type=int, default=6006)
    return p.parse_args()


def main():
    args = parse_args()
    if not osp.isdir(args.work_dir):
        raise SystemExit(f'no such directory: {args.work_dir}')

    STATE['work_dir'] = args.work_dir
    STATE['tb_port'] = args.tb_port

    tb_logdir = osp.join(args.work_dir, 'tf_logs')
    # Not the work_dir (may be root-owned / read-only for this process --
    # see CLAUDE.md) and not DEVNULL (silent failures were exactly the
    # "tensorboard doesn't work, no idea why" problem) -- a real temp file
    # whose path gets surfaced in the console and in the page itself.
    tb_log_file = tempfile.NamedTemporaryFile(
        prefix='webviewer_tensorboard_', suffix='.log', delete=False)
    STATE['tb_log_path'] = tb_log_file.name

    tb_proc = subprocess.Popen(
        [sys.executable, '-m', 'tensorboard.main', '--logdir', tb_logdir,
         '--port', str(args.tb_port), '--host', '127.0.0.1'],
        stdout=tb_log_file, stderr=subprocess.STDOUT,
    )
    atexit.register(tb_proc.terminate)
    print(f'TensorBoard starting on http://127.0.0.1:{args.tb_port} '
          f'(logdir={tb_logdir}, log={tb_log_file.name})')

    print(f'Viewer on http://127.0.0.1:{args.port}')
    app.run(host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
