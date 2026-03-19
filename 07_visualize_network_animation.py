"""
IEEE 57-Bus Network Animation - Neural Network AC-OPF
=====================================================
Generates an animated GIF showing the 57-bus power grid as the
trained neural network dispatches generators across varying load conditions.

The animation runs the actual trained NN (Zamzam et al. 2019):
  1. NN predicts alpha (active power) and beta (voltage setpoints)
  2. Power flow solved to find reactive power Qg
  3. Any Qg violations are clipped and power flow re-solved (Algorithm 1)
  4. Generator nodes that triggered correction are highlighted red

Node color  = voltage magnitude (p.u.) from final power flow solution
Gen node    = diamond; red border = Qg correction was applied this scenario
Load bar    = system load level (% of nominal)

Usage:
    python 07_visualize_network_animation.py

Output:
    Outputs_V3/network_animation.gif
"""

import os, sys
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from PIL import Image
import io, json, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

from pypower.api import case57, runpf, ppoption
from pypower.idx_bus import PD, QD, VM, VA, BUS_TYPE, PQ, PV
from pypower.idx_gen import PG, QG, VG, GEN_BUS
from pypower.idx_brch import F_BUS, T_BUS, PF as BR_PF

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "Outputs_V3")
os.makedirs(OUTPUT_DIR, exist_ok=True)
GIF_PATH     = os.path.join(OUTPUT_DIR, "network_animation.gif")

# Model lives in the archive (not committed to repo)
MODEL_PATH = (
    r"C:\Users\jborr\.vscode\Workspaces\archive\SciML\Project Codes"
    r"\First Attempt\Version 2\Outputs_v2\best_opf_model_v3_improved.pth"
)
LIMITS_PATH = os.path.join(
    r"C:\Users\jborr\.vscode\Workspaces\archive\SciML\Project Codes"
    r"\First Attempt\Version 2\Outputs_v2", "generator_limits.json"
)

# ── Animation parameters ───────────────────────────────────────────────
N_FRAMES  = 48
FPS       = 10
DPI       = 110

LOAD_LEVELS = 0.90 + 0.35 * np.sin(np.linspace(0, 2 * np.pi, N_FRAMES))

# ── Colour scheme ──────────────────────────────────────────────────────
BG_COLOR      = "#0d1117"
PANEL_COLOR   = "#161b22"
EDGE_DIM      = "#2a3140"
EDGE_HI       = "#58a6ff"
TEXT_COLOR    = "#e6edf3"
MUTED_COLOR   = "#8b949e"
ACCENT_BLUE   = "#58a6ff"
ACCENT_GREEN  = "#3fb950"
ACCENT_ORANGE = "#d29922"
ACCENT_RED    = "#f85149"

CMAP_VOLTAGES = cm.plasma
VM_MIN_PLOT   = 0.92
VM_MAX_PLOT   = 1.08
norm_vm       = mcolors.Normalize(vmin=VM_MIN_PLOT, vmax=VM_MAX_PLOT)

# ── NN model definition (must match training) ──────────────────────────
class OPF_NN_Zamzam(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, input_size),  nn.Sigmoid(),
            nn.Linear(input_size, input_size),  nn.Sigmoid(),
            nn.Linear(input_size, output_size), nn.Sigmoid(),
            nn.Linear(output_size, output_size),nn.Sigmoid(),
        )
    def forward(self, x):
        return self.network(x)

# ── Load model ─────────────────────────────────────────────────────────
print("Loading trained NN model ...")
if not os.path.exists(MODEL_PATH):
    sys.exit(f"Model not found: {MODEL_PATH}")

DEVICE = torch.device('cpu')
ckpt   = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model  = OPF_NN_Zamzam(ckpt['input_size'], ckpt['output_size']).to(DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
input_scaler = ckpt['input_scaler']
print(f"  input_size={ckpt['input_size']}, output_size={ckpt['output_size']}")

# ── Load generator limits ──────────────────────────────────────────────
print("Loading generator limits ...")
with open(LIMITS_PATH) as f:
    lim = json.load(f)

n_gens      = lim['n_gens']           # 7
pg_limits   = lim['pg_limits']        # list of [min, max] per gen
qg_limits   = lim['qg_limits']        # list of [min, max] per gen
vm_min_arr  = np.array(lim['vm_min_orig'])
vm_max_arr  = np.array(lim['vm_max_orig'])
lam         = lim.get('lambda', 0.005)

# Restricted voltage limits (interior, with lambda margin)
vm_min = vm_min_arr[0] + lam * (vm_max_arr[0] - vm_min_arr[0])
vm_max = vm_max_arr[0] - lam * (vm_max_arr[0] - vm_min_arr[0])

gen_qg_min = np.array([q[0] for q in qg_limits])
gen_qg_max = np.array([q[1] for q in qg_limits])

# ── Build network ──────────────────────────────────────────────────────
print("Building IEEE 57-bus graph ...")
ppc_base  = case57()
n_bus     = ppc_base['bus'].shape[0]
branch    = ppc_base['branch']
gen_arr   = ppc_base['gen']
gen_buses = list(gen_arr[:, GEN_BUS].astype(int))   # ordered list
gen_set   = set(gen_buses)
slack_bus = gen_buses[0]

base_pd = ppc_base['bus'][:, PD].copy()
base_qd = ppc_base['bus'][:, QD].copy()

# All load bus indices (buses with Pd > 0)
load_bus_mask    = base_pd > 0
load_bus_indices = np.where(load_bus_mask)[0]

G = nx.Graph()
G.add_nodes_from(range(1, n_bus + 1))
for br in branch:
    G.add_edge(int(br[F_BUS]), int(br[T_BUS]))

np.random.seed(7)
pos = nx.spring_layout(G, k=2.2, iterations=120, seed=7)

node_list = sorted(G.nodes())
edge_list  = list(G.edges())

pf_opt = ppoption(VERBOSE=0, OUT_ALL=0)


# ── NN inference + Algorithm 1 recovery ────────────────────────────────
def run_nn_and_recover(load_scale):
    """
    Run the full Zamzam pipeline for a single load scenario.

    Returns:
        vm_final   : voltage magnitudes (57,) after recovery
        pf_final   : branch active power flows (80,)
        qg_violated: bool array (n_gens,) — True if Qg was clipped
        qg_clip_amt: float array (n_gens,) — amount clipped (MVAr)
        pg_slack   : slack bus active power after recovery (MW)
        qg_final   : final Qg values (n_gens,) after recovery
        converged  : bool
    """
    pd_scaled = base_pd * load_scale
    qd_scaled = base_qd * load_scale

    # ── Step 1: NN prediction ──────────────────────────────────────
    # Input: [pd_1..pd_42, qd_1..qd_42]  (only load buses with Pd > 0)
    pd_loads = pd_scaled[load_bus_mask]
    qd_loads = qd_scaled[load_bus_mask]
    x_raw    = np.concatenate([pd_loads, qd_loads]).reshape(1, -1)
    x_scaled = input_scaler.transform(x_raw)
    x_tensor = torch.FloatTensor(x_scaled)

    with torch.no_grad():
        out = model(x_tensor).numpy()[0]   # shape (13,): 6 alpha + 7 beta

    alphas = out[:n_gens - 1]   # alpha for non-slack gens (6)
    betas  = out[n_gens - 1:]   # beta for all gen buses   (7)

    # ── Step 2: Decode alpha/beta → Pg, Vm ────────────────────────
    pg_non_slack = []
    for i, a in enumerate(alphas):
        pg_min, pg_max = pg_limits[i + 1]
        pg_non_slack.append(pg_min + a * (pg_max - pg_min))

    vm_gens = [vm_min + b * (vm_max - vm_min) for b in betas]

    # ── Step 3: Initial power flow (get Qg from physics) ──────────
    ppc = case57()
    ppc['bus'][load_bus_indices, PD] = pd_loads
    ppc['bus'][load_bus_indices, QD] = qd_loads
    for i, pg in enumerate(pg_non_slack):
        ppc['gen'][i + 1, PG] = pg
    for i, vm in enumerate(vm_gens):
        ppc['gen'][i, VG] = vm

    res, ok = runpf(ppc, pf_opt)
    if not ok:
        vm_fallback = ppc_base['bus'][:, VM].copy()
        return vm_fallback, np.zeros(len(branch)), np.zeros(n_gens, bool), \
               np.zeros(n_gens), None, np.zeros(n_gens), False

    qg_init    = res['gen'][:, QG].copy()
    vm_init    = res['bus'][:, VM].copy()

    # ── Step 4: Check Qg limits ────────────────────────────────────
    violated   = (qg_init < gen_qg_min) | (qg_init > gen_qg_max)
    qg_clipped = np.clip(qg_init, gen_qg_min, gen_qg_max)
    clip_amt   = np.abs(qg_init - qg_clipped)

    if not violated.any():
        pf_flows = res['branch'][:, BR_PF]
        return vm_init, pf_flows, violated, clip_amt, res['gen'][0, PG], qg_init, True

    # ── Step 5: Re-solve with violated generators as PQ buses ──────
    ppc2 = case57()
    ppc2['bus']    = res['bus'].copy()
    ppc2['gen']    = res['gen'].copy()
    ppc2['branch'] = res['branch'].copy()
    ppc2['bus'][load_bus_indices, PD] = pd_loads
    ppc2['bus'][load_bus_indices, QD] = qd_loads

    for i in range(n_gens):
        if violated[i] and i != 0:
            bus_idx = gen_buses[i] - 1
            ppc2['bus'][bus_idx, BUS_TYPE] = PQ
            ppc2['gen'][i, QG] = qg_clipped[i]

    res2, ok2 = runpf(ppc2, pf_opt)
    if not ok2:
        pf_flows = res['branch'][:, BR_PF]
        return vm_init, pf_flows, violated, clip_amt, res['gen'][0, PG], qg_clipped, True

    vm_final  = res2['bus'][:, VM]
    pf_final  = res2['branch'][:, BR_PF]
    pg_slack  = res2['gen'][0, PG]
    qg_final  = res2['gen'][:, QG]
    return vm_final, pf_final, violated, clip_amt, pg_slack, qg_final, True


# ── Pre-compute all frames ─────────────────────────────────────────────
print(f"Running NN + Algorithm 1 for {N_FRAMES} load scenarios ...")
vm_frames, pf_frames, viol_frames, clip_frames, pgslack_frames, qg_frames = \
    [], [], [], [], [], []

for i, scale in enumerate(LOAD_LEVELS):
    vm, pf, viol, clip, pgslack, qg_fin, _ = run_nn_and_recover(scale)
    vm_frames.append(vm)
    pf_frames.append(pf)
    viol_frames.append(viol)
    clip_frames.append(clip)
    pgslack_frames.append(pgslack if pgslack is not None else 0.0)
    qg_frames.append(qg_fin)
    if (i + 1) % 12 == 0:
        print(f"  {i+1}/{N_FRAMES} done")

vm_arr     = np.array(vm_frames)    # (N_FRAMES, 57)
pf_arr     = np.array(pf_frames)    # (N_FRAMES, 80)
viol_arr   = np.array(viol_frames)  # (N_FRAMES, 7)  bool
clip_arr   = np.array(clip_frames)  # (N_FRAMES, 7)
pslack_arr = np.array(pgslack_frames)

pf_p95   = np.percentile(np.abs(pf_arr), 95)
qg_arr   = np.array(qg_frames)     # (N_FRAMES, 7)


def node_sizes(load_scale):
    load = base_pd * load_scale
    sz   = 90 + 190 * (load / (base_pd.max() + 1e-6))
    sz[base_pd < 1] = 110
    return sz


# ── Frame renderer ─────────────────────────────────────────────────────
FIG_W, FIG_H = 13.0, 8.0

def make_frame(idx):
    vm      = vm_arr[idx]
    pf      = pf_arr[idx]
    viol    = viol_arr[idx]       # (7,) bool — which gens had Qg clipped
    clip    = clip_arr[idx]       # (7,) float — clip amounts
    qg      = qg_arr[idx]         # (7,) float — final Qg values (MVAr)
    scale   = LOAD_LEVELS[idx]
    pg_slk  = pslack_arr[idx]
    n_viol  = int(viol.sum())

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG_COLOR)
    ax  = fig.add_axes([0.01, 0.08, 0.64, 0.88])
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    sp  = fig.add_axes([0.66, 0.08, 0.33, 0.88])
    sp.set_facecolor(PANEL_COLOR)
    sp.set_xlim(0, 1); sp.set_ylim(0, 1)
    sp.axis('off')

    # ── Edges ──────────────────────────────────────────────────────
    pf_mag  = np.abs(pf)
    pf_norm = np.clip(pf_mag / (pf_p95 + 1e-6), 0, 1)
    for ei, (u, v) in enumerate(edge_list):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        r   = pf_norm[ei]
        col = (1-r)*np.array(mcolors.to_rgb(EDGE_DIM)) + r*np.array(mcolors.to_rgb(EDGE_HI))
        ax.plot([x0,x1],[y0,y1], color=col, linewidth=0.7+2.0*r,
                zorder=1, solid_capstyle='round')

    # ── Load buses ─────────────────────────────────────────────────
    sizes      = node_sizes(scale)
    load_nodes = [b for b in node_list if b not in gen_set]
    lxy  = np.array([pos[b] for b in load_nodes])
    lcol = [CMAP_VOLTAGES(norm_vm(vm[b-1])) for b in load_nodes]
    lsz  = [sizes[b-1] for b in load_nodes]
    ax.scatter(lxy[:,0], lxy[:,1], c=lcol, s=lsz,
               edgecolors=BG_COLOR, linewidths=0.5, zorder=3)

    # ── Generator buses ────────────────────────────────────────────
    for gi, gbus in enumerate(gen_buses):
        x, y   = pos[gbus]
        gcol   = CMAP_VOLTAGES(norm_vm(vm[gbus-1]))
        is_viol = viol[gi]
        ring_col = ACCENT_RED if is_viol else ACCENT_ORANGE
        # Halo
        ax.scatter(x, y, c=ring_col, s=420, marker='D',
                   alpha=0.35, zorder=4)
        # Diamond body
        ax.scatter(x, y, c=[gcol], s=280, marker='D',
                   edgecolors=ring_col, linewidths=2.2, zorder=5)

        # Small pulse ring for violated generators
        if is_viol:
            ax.scatter(x, y, c='none', s=560, marker='D',
                       edgecolors=ACCENT_RED, linewidths=1.0,
                       alpha=0.5, zorder=4)

    # ── SLACK label ────────────────────────────────────────────────
    sx, sy = pos[slack_bus]
    ax.text(sx, sy+0.055, 'SLACK', fontsize=5.5, ha='center', va='bottom',
            color=ACCENT_ORANGE, fontweight='bold', zorder=6,
            bbox=dict(boxstyle='round,pad=0.18', fc=BG_COLOR,
                      ec=ACCENT_ORANGE, alpha=0.8, lw=0.9))

    # ── Legend ─────────────────────────────────────────────────────
    handles = [
        Line2D([0],[0], marker='o', color='none',
               markerfacecolor='#c060c0', markersize=7,
               label='Load / transit bus'),
        Line2D([0],[0], marker='D', color='none',
               markerfacecolor=ACCENT_ORANGE,
               markeredgecolor=ACCENT_ORANGE,
               markersize=7, label='Generator (feasible)'),
        Line2D([0],[0], marker='D', color='none',
               markerfacecolor='#c04040',
               markeredgecolor=ACCENT_RED,
               markersize=7, label='Generator (Qg corrected)'),
        Line2D([0],[0], color=EDGE_HI, linewidth=1.8,
               label='High-loading branch'),
    ]
    ax.legend(handles=handles, loc='lower left', fontsize=6.5,
              facecolor=PANEL_COLOR, edgecolor="#30363d",
              labelcolor=TEXT_COLOR, framealpha=0.9)

    ax.set_title('IEEE 57-Bus System  \u00b7  Neural Network AC-OPF Dispatch',
                 color=TEXT_COLOR, fontsize=11, fontweight='bold',
                 pad=6, loc='center')

    # ── Colorbar ───────────────────────────────────────────────────
    ax_cb = fig.add_axes([0.02, 0.02, 0.60, 0.034])
    cb = plt.colorbar(
        cm.ScalarMappable(norm=norm_vm, cmap=CMAP_VOLTAGES),
        cax=ax_cb, orientation='horizontal'
    )
    cb.set_label('Voltage Magnitude (p.u.)', color=MUTED_COLOR, fontsize=7.5, labelpad=2)
    cb.ax.tick_params(colors=MUTED_COLOR, labelsize=7)

    # ── Stats panel ────────────────────────────────────────────────
    load_pct = scale * 100
    load_col = (ACCENT_GREEN if load_pct < 88
                else ACCENT_ORANGE if load_pct < 112
                else ACCENT_RED)

    def divider(y):
        sp.axhline(y=y, color="#30363d", linewidth=0.7, xmin=0.05, xmax=0.95)

    def row(y, txt, val, vc=ACCENT_BLUE, fs_v=9.5):
        sp.text(0.08, y, txt, transform=sp.transAxes,
                fontsize=7.5, color=MUTED_COLOR, va='top')
        sp.text(0.92, y, val, transform=sp.transAxes,
                fontsize=fs_v, color=vc, va='top', ha='right', fontweight='bold')

    # ── Header row ─────────────────────────────────────────────────
    sp.text(0.5, 0.975, 'LIVE METRICS', transform=sp.transAxes,
            fontsize=8, color=MUTED_COLOR, ha='center', va='top', fontweight='bold')
    divider(0.935)

    row(0.915, 'System load',  f'{load_pct:.1f}%',                    vc=load_col)
    row(0.845, 'Total demand', f'{(base_pd*scale).sum():.0f} MW',      vc=TEXT_COLOR)
    vm_min_v, vm_max_v = vm.min(), vm.max()
    vm_col = ACCENT_GREEN if (vm_min_v > 0.94 and vm_max_v < 1.06) else ACCENT_ORANGE
    row(0.775, 'Vm range', f'{vm_min_v:.3f}\u2013{vm_max_v:.3f} pu',  vc=vm_col)

    divider(0.725)

    # ── Algorithm 1 pipeline ───────────────────────────────────────
    sp.text(0.5, 0.705, 'ALGORITHM 1  \u2014  RECOVERY PIPELINE',
            transform=sp.transAxes, fontsize=6.8, color=MUTED_COLOR,
            ha='center', va='top', fontweight='bold')

    # Three pipeline steps drawn as coloured boxes with arrows
    # Step 3 colour scales with correction count: 1=orange, 2=orange-red, 3=red
    step3_colors = {0: ACCENT_GREEN, 1: '#e3a030', 2: '#e05530', 3: ACCENT_RED}
    step3_col = step3_colors.get(n_viol, ACCENT_RED)
    step3_lbl = f'3. Qg Check\n({n_viol} corrected)'

    box_y  = 0.585
    box_h  = 0.095
    box_w  = 0.25
    bx     = [0.07, 0.385, 0.70]
    labels = ['1. NN\nPredict', '2. Solve\nPower Flow', step3_lbl]
    colors = [ACCENT_GREEN, ACCENT_GREEN, step3_col]

    for bi, (bxi, lbl, col) in enumerate(zip(bx, labels, colors)):
        sp.add_patch(plt.Rectangle(
            (bxi, box_y - box_h), box_w, box_h,
            transform=sp.transAxes,
            fc=col + '33', ec=col, linewidth=1.4, zorder=3
        ))
        sp.text(bxi + box_w/2, box_y - box_h/2, lbl,
                transform=sp.transAxes, fontsize=6.0, color=col,
                ha='center', va='center', fontweight='bold', linespacing=1.4)
        if bi < 2:
            sp.annotate('', xy=(bx[bi+1] - 0.005, box_y - box_h/2),
                        xytext=(bxi + box_w + 0.005, box_y - box_h/2),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color=MUTED_COLOR, lw=1.0))

    # Status line below pipeline
    names = ', '.join(f'G{gi+1}' for gi in range(n_gens) if viol[gi])
    status_txt = f'\u26a0  corrected: {names}' if n_viol > 0 else '\u2713  All generators feasible'
    status_col = step3_col

    sp.text(0.5, box_y - box_h - 0.025, status_txt,
            transform=sp.transAxes, fontsize=6.5, color=status_col,
            ha='center', va='top', fontweight='bold')

    divider(0.445)

    # ── Per-generator Qg gauges ────────────────────────────────────
    sp.text(0.5, 0.428, 'REACTIVE POWER  Qg  vs  LIMITS  (MVAr)',
            transform=sp.transAxes, fontsize=6.8, color=MUTED_COLOR,
            ha='center', va='top', fontweight='bold')

    gauge_top  = 0.395    # top y of first gauge row
    gauge_h    = 0.046    # height of each gauge bar
    gauge_gap  = 0.058    # spacing between rows
    gx0, gw   = 0.22, 0.68   # bar x start and width

    for gi in range(n_gens):
        qg_val  = qg[gi]
        qg_lo   = gen_qg_min[gi]
        qg_hi   = gen_qg_max[gi]
        q_range = qg_hi - qg_lo if (qg_hi - qg_lo) > 0 else 1.0
        fill    = float(np.clip((qg_val - qg_lo) / q_range, 0, 1))
        is_v    = bool(viol[gi])
        bar_col = ACCENT_RED if is_v else ACCENT_BLUE
        gy      = gauge_top - gi * gauge_gap

        # Generator label
        sp.text(0.06, gy, f'G{gi+1}', transform=sp.transAxes,
                fontsize=6.5, color=ACCENT_ORANGE if gi == 0 else TEXT_COLOR,
                va='center', fontweight='bold')

        # Track
        sp.add_patch(plt.Rectangle(
            (gx0, gy - gauge_h/2), gw, gauge_h,
            transform=sp.transAxes, fc='#1f2937', ec='#30363d',
            linewidth=0.6, zorder=2
        ))
        # Fill
        sp.add_patch(plt.Rectangle(
            (gx0, gy - gauge_h/2), gw * fill, gauge_h,
            transform=sp.transAxes, fc=bar_col, ec='none',
            alpha=0.80, zorder=3
        ))
        # Clipped-limit marker line at right edge if violated
        if is_v:
            sp.plot([gx0 + gw * fill, gx0 + gw * fill],
                    [gy - gauge_h/2, gy + gauge_h/2],
                    transform=sp.transAxes, color=ACCENT_RED,
                    lw=1.5, zorder=4)

        # Qg value text
        sp.text(gx0 + gw + 0.03, gy, f'{qg_val:.1f}',
                transform=sp.transAxes, fontsize=5.8,
                color=bar_col if is_v else MUTED_COLOR,
                va='center', fontweight='bold' if is_v else 'normal')

    # Min/Max axis labels under gauges
    last_gy = gauge_top - (n_gens - 1) * gauge_gap
    sp.text(gx0, last_gy - gauge_h/2 - 0.02, 'min',
            transform=sp.transAxes, fontsize=5.5, color=MUTED_COLOR,
            ha='left', va='top')
    sp.text(gx0 + gw, last_gy - gauge_h/2 - 0.02, 'max',
            transform=sp.transAxes, fontsize=5.5, color=MUTED_COLOR,
            ha='right', va='top')

    divider(0.038)
    sp.text(0.5, 0.022, 'Zamzam & Baker (2019)  \u00b7  IEEE 57-bus  \u00b7  Johns Hopkins SciML',
            transform=sp.transAxes, fontsize=5.2,
            color=MUTED_COLOR, ha='center', va='top', style='italic')

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor=BG_COLOR,
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── Render + save ──────────────────────────────────────────────────────
print(f"\nRendering {N_FRAMES} frames ...")
frames = []
for i in range(N_FRAMES):
    frames.append(make_frame(i))
    if (i + 1) % 12 == 0:
        print(f"  frame {i+1}/{N_FRAMES}")

print(f"\nSaving GIF -> {GIF_PATH}")
frames[0].save(
    GIF_PATH, save_all=True, append_images=frames[1:],
    duration=int(1000 / FPS), loop=0, optimize=False,
)
size_mb = os.path.getsize(GIF_PATH) / 1024**2
print(f"  Done! File size: {size_mb:.1f} MB")
print(f"\nEmbed in README.md with:")
print(f"  ![Network Animation](Outputs_V3/network_animation.gif)")
