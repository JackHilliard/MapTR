"""Minimal local web viewer for a finished MapTRv2 training run.

Shows, on one page, at http://127.0.0.1:<port>:
  - a loss/eval-metric curve (parsed from the run's *.log.json)
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
that same interpreter.
"""
import argparse
import atexit
import glob
import io
import os.path as osp
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')  # headless -- no display in a server process
import matplotlib.pyplot as plt
from flask import Flask, Response, abort, send_file

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from tools.analysis_tools.analyze_logs import load_json_logs  # noqa: E402

app = Flask(__name__)
STATE = {}  # populated in main() -- work_dir, tb_port


def find_log_json(work_dir):
    logs = sorted(glob.glob(osp.join(work_dir, '*.log.json')))
    return logs[-1] if logs else None  # most recent invocation


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


def build_loss_figure(log_json_path):
    """Mirrors analyze_logs.py's own x-axis construction for train/val
    modes, but renders directly to an in-memory PNG instead of a CLI-
    specified output file (that script's plot_curve() is argparse/file-
    oriented, awkward to reuse as-is inside a web route)."""
    log_dicts = load_json_logs([log_json_path])
    log_dict = log_dicts[0]
    epochs = sorted(log_dict.keys())
    if not epochs:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))

    # --- train loss vs global iteration ---
    ax = axes[0]
    iters_per_epoch = log_dict[epochs[0]]['iter'][-1] if log_dict[epochs[0]]['iter'] else 0
    xs, ys = [], []
    for epoch in epochs:
        d = log_dict[epoch]
        if 'loss' not in d:
            continue
        n = len(d['loss'])
        iters = d['iter'][:n]
        xs.extend([(epoch - 1) * iters_per_epoch + it for it in iters])
        ys.extend(d['loss'][:n])
    if xs:
        ax.plot(xs, ys, linewidth=0.7)
    ax.set_xlabel('iter')
    ax.set_ylabel('loss')
    ax.set_title('training loss')

    # --- eval metric vs epoch, if any eval entries are present ---
    ax = axes[1]
    metric_key = None
    for epoch in epochs:
        for k in log_dict[epoch]:
            if k.endswith('/mAP'):
                metric_key = k
                break
        if metric_key:
            break
    if metric_key:
        xs, ys = [], []
        for epoch in epochs:
            vals = log_dict[epoch].get(metric_key)
            if vals:
                xs.append(epoch)
                ys.append(vals[-1])
        ax.plot(xs, ys, marker='o')
        ax.set_xlabel('epoch')
        ax.set_ylabel(metric_key)
        ax.set_title('eval metric')
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'no eval metric logged yet', ha='center', va='center',
                 transform=ax.transAxes, fontsize=9, color='gray')

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110)
    plt.close(fig)
    buf.seek(0)
    return buf


PAGE_TEMPLATE = """<!doctype html>
<html><head>
<title>MapTRv2 run viewer</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; }}
  h1 {{ font-size: 1.3em; }}
  h2 {{ font-size: 1.05em; margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 0.3em; }}
  .meta {{ color: #555; font-size: 0.9em; }}
  .meta code {{ background: #f2f2f2; padding: 1px 4px; }}
  img.plot {{ max-width: 100%; }}
  .gallery {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .gallery a img {{ width: 220px; border: 1px solid #ddd; }}
  iframe {{ width: 100%; height: 700px; border: 1px solid #ddd; }}
  ul {{ font-size: 0.9em; }}
</style>
</head><body>
<h1>MapTRv2 run: {work_dir}</h1>
<p class="meta">config: <code>{config}</code></p>

<h2>Loss / eval curve</h2>
{loss_section}

<h2>TensorBoard</h2>
<iframe src="http://127.0.0.1:{tb_port}"></iframe>

<h2>BEV predictions ({n_vis} images)</h2>
<div class="gallery">
{gallery}
</div>

<h2>Checkpoints</h2>
<ul>{ckpt_list}</ul>

</body></html>
"""


@app.route('/')
def index():
    work_dir = STATE['work_dir']
    log_json = find_log_json(work_dir)
    loss_section = ('<img class="plot" src="/plot.png">' if log_json
                     else '<p class="meta">no *.log.json found in this work_dir yet</p>')

    images = find_vis_images(work_dir)
    gallery = '\n'.join(
        f'<a href="/vis/{f}"><img src="/vis/{f}" title="{f}"></a>' for f in images
    ) or '<p class="meta">no images in vis/ -- run tools/maptrv2/carla_bev_vis.py against this work_dir first</p>'

    ckpts = find_checkpoints(work_dir)
    ckpt_list = '\n'.join(f'<li><code>{c}</code></li>' for c in ckpts) or '<li>none found</li>'

    return PAGE_TEMPLATE.format(
        work_dir=osp.abspath(work_dir),
        config=find_config_file(work_dir) or 'unknown',
        loss_section=loss_section,
        tb_port=STATE['tb_port'],
        n_vis=len(images),
        gallery=gallery,
        ckpt_list=ckpt_list,
    )


@app.route('/plot.png')
def plot_png():
    log_json = find_log_json(STATE['work_dir'])
    if not log_json:
        abort(404)
    buf = build_loss_figure(log_json)
    if buf is None:
        abort(404)
    return send_file(buf, mimetype='image/png')


@app.route('/vis/<path:filename>')
def vis_image(filename):
    path = osp.join(STATE['work_dir'], 'vis', filename)
    if not osp.isfile(path):
        abort(404)
    return send_file(osp.abspath(path))


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
    tb_proc = subprocess.Popen(
        [sys.executable, '-m', 'tensorboard.main', '--logdir', tb_logdir,
         '--port', str(args.tb_port), '--host', '127.0.0.1'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    atexit.register(tb_proc.terminate)
    print(f'TensorBoard starting on http://127.0.0.1:{args.tb_port} '
          f'(logdir={tb_logdir})')

    print(f'Viewer on http://127.0.0.1:{args.port}')
    app.run(host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
