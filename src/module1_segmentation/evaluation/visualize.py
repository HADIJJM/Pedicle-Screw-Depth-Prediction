"""Overlays of reference vs predicted masks on CT, and summary plots.

Case figures are drawn in RAS+ canonical orientation so every scan (LPS, PIR,
...) looks the same: sagittal with anterior on the left, coronal in radiological
convention (patient's right on the viewer's left), pixel extents in millimetres
so anisotropic voxels keep their true aspect. Research prototype.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.module1_segmentation.evaluation.metrics import LEVEL_NAMES  # noqa: E402

# Fixed categorical order, one hue per level (adjacent vertebrae = adjacent slots,
# validated for colour-vision deficiency on adjacent pairs). Dark-surface steps
# are used on CT, light-surface steps on white plots.
LEVEL_COLORS_ON_CT = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767")
SERIES_1 = "#2a78d6"
STATUS_COLORS = {"correct": "#0ca30c", "missed": "#fab219", "wrong level": "#ec835a", "false positive": "#d03b3b"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ_BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
CT_WINDOW_HU = (-500.0, 1300.0)  # bone window, level 400 / width 1800
CROP_MARGIN_MM = 25.0
LABEL_ALPHA = 0.45


def _hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ------------------------------------------------------------ orientation
def to_ras(arr: np.ndarray, affine: np.ndarray, zooms) -> tuple[np.ndarray, np.ndarray]:
    """Reorient a voxel array to RAS+ (views, no copy); returns (array, voxel sizes)."""
    from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform
    start = io_orientation(affine)
    transform = ornt_transform(start, axcodes2ornt(("R", "A", "S")))
    out = apply_orientation(arr, transform)
    new_zooms = np.zeros(3)
    for i in range(3):
        new_zooms[int(transform[i, 0])] = zooms[i]
    return out, new_zooms


def _slab_index(fg: np.ndarray, axis: int) -> int:
    other = tuple(a for a in range(3) if a != axis)
    counts = fg.sum(axis=other)
    return int(np.argmax(counts)) if counts.any() else fg.shape[axis] // 2


def _crop(fg2d: np.ndarray, zooms2d) -> tuple[slice, slice]:
    """Crop (rows, cols) of a 2D slice around foreground with a margin in mm."""
    if not fg2d.any():
        return slice(None), slice(None)
    rr, cc = np.nonzero(fg2d)
    mr, mc = (int(np.ceil(CROP_MARGIN_MM / z)) for z in zooms2d)
    return (slice(max(rr.min() - mr, 0), rr.max() + mr + 1),
            slice(max(cc.min() - mc, 0), cc.max() + mc + 1))


def _views(vol: np.ndarray, x: int, y: int) -> dict[str, np.ndarray]:
    """2D display arrays (rows = superior at top after origin='lower', cols per convention)."""
    sag = vol[x, :, :].T[:, ::-1]    # rows S, cols A reversed -> anterior on the left
    cor = vol[:, y, :].T[:, ::-1]    # rows S, cols R reversed -> patient's right on the left
    return {"sagittal": sag, "coronal": cor}


def _edges(lab: np.ndarray) -> np.ndarray:
    e = np.zeros(lab.shape, dtype=bool)
    e[1:, :] |= lab[1:, :] != lab[:-1, :]
    e[:-1, :] |= lab[1:, :] != lab[:-1, :]
    e[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    e[:, :-1] |= lab[:, 1:] != lab[:, :-1]
    return e & (lab > 0)


def _label_rgba(lab: np.ndarray) -> np.ndarray:
    lut = np.zeros((9, 4))
    for c, h in enumerate(LEVEL_COLORS_ON_CT, 1):
        lut[c, :3] = _hex_rgb(h)
        lut[c, 3] = LABEL_ALPHA
    rgba = lut[lab]
    edge = _edges(lab)
    rgba[edge, 3] = 1.0
    return rgba


def _error_rgba(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    rgba = np.zeros(gt.shape + (4,))
    cats = {
        "correct": (gt > 0) & (pred == gt),
        "wrong level": (gt > 0) & (pred > 0) & (pred != gt),
        "missed": (gt > 0) & (pred == 0),
        "false positive": (gt == 0) & (pred > 0),
    }
    for name, m in cats.items():
        rgba[m, :3] = _hex_rgb(STATUS_COLORS[name])
        rgba[m, 3] = 0.75
    return rgba


def _direct_labels(ax, lab: np.ndarray, extent_scale) -> None:
    sx, sy = extent_scale
    for c in range(1, 9):
        rr, cc = np.nonzero(lab == c)
        if rr.size < 30:
            continue
        t = ax.text((np.median(cc) + 0.5) * sx, (np.median(rr) + 0.5) * sy, LEVEL_NAMES[c - 1],
                    color="white", fontsize=7.5, ha="center", va="center", fontweight="bold")
        t.set_path_effects([patheffects.withStroke(linewidth=2.2, foreground="black")])


def _orient_letters(ax, view: str, w: float, h: float) -> None:
    left, right = ("A", "P") if view == "sagittal" else ("R", "L")
    style = dict(color="#c3c2b7", fontsize=8, fontweight="bold")
    ax.text(0.01 * w, 0.5 * h, left, ha="left", va="center", **style)
    ax.text(0.99 * w, 0.5 * h, right, ha="right", va="center", **style)
    ax.text(0.5 * w, 0.99 * h, "S", ha="center", va="top", **style)
    ax.text(0.5 * w, 0.01 * h, "I", ha="center", va="bottom", **style)


def render_case_figure(ct_hu: np.ndarray | None, gt: np.ndarray, pred: np.ndarray, affine: np.ndarray,
                       out_path: str | Path, case: dict[str, Any]) -> Path:
    """Sagittal + coronal: reference overlay, prediction overlay, error map."""
    zooms = np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))
    gt_r, z = to_ras(gt, affine, zooms)
    pred_r, _ = to_ras(pred, affine, zooms)
    ct_r = to_ras(ct_hu, affine, zooms)[0] if ct_hu is not None else None
    fg = gt_r > 0 if gt_r.any() else pred_r > 0
    x, y = _slab_index(fg, 0), _slab_index(fg, 1)

    v_gt, v_pred = _views(gt_r, x, y), _views(pred_r, x, y)
    v_ct = _views(ct_r, x, y) if ct_r is not None else None
    fg_union = {k: (v_gt[k] > 0) | (v_pred[k] > 0) for k in v_gt}
    pix = {"sagittal": (z[2], z[1]), "coronal": (z[2], z[0])}  # (row size, col size) in mm

    fig, axes = plt.subplots(2, 3, figsize=(13, 10.5), facecolor="#1a1a19")
    titles = ("Reference", "Prediction", "Error map")
    for r, view in enumerate(("sagittal", "coronal")):
        rs, cs = _crop(fg_union[view], pix[view])
        g, p = v_gt[view][rs, cs], v_pred[view][rs, cs]
        h_mm, w_mm = g.shape[0] * pix[view][0], g.shape[1] * pix[view][1]
        extent = (0, w_mm, 0, h_mm)
        for col in range(3):
            ax = axes[r, col]
            ax.set_facecolor("black")
            if v_ct is not None:
                ax.imshow(v_ct[view][rs, cs], cmap="gray", vmin=CT_WINDOW_HU[0], vmax=CT_WINDOW_HU[1],
                          origin="lower", extent=extent, interpolation="bilinear")
            if col == 0:
                ax.imshow(_label_rgba(g), origin="lower", extent=extent, interpolation="nearest")
                _direct_labels(ax, g, (pix[view][1], pix[view][0]))
            elif col == 1:
                ax.imshow(_label_rgba(p), origin="lower", extent=extent, interpolation="nearest")
                _direct_labels(ax, p, (pix[view][1], pix[view][0]))
            else:
                ax.imshow(_error_rgba(g, p), origin="lower", extent=extent, interpolation="nearest")
            ax.set_xlim(0, w_mm)
            ax.set_ylim(0, h_mm)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            _orient_letters(ax, view, w_mm, h_mm)
            ax.set_title(f"{titles[col]} · {view}", color="#f0efec", fontsize=10)

    def f(v):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"

    title = (f"{case.get('scan_id', '')}  ·  mean Dice {f(case.get('mean_dice'))}  ·  "
             f"spine Dice {f(case.get('spine_dice'))}  ·  {case.get('failure_category', '')}")
    if case.get("best_shift"):
        title += f"  ·  best shift {int(case['best_shift']):+d} → Dice {f(case.get('mean_dice_at_best_shift'))}"
    if ct_r is None:
        title += "  ·  CT not shown (not on reference grid)"
    fig.suptitle(title, color="#f0efec", fontsize=11.5)
    lvl = [Patch(facecolor=c, edgecolor="none", label=n) for n, c in zip(LEVEL_NAMES, LEVEL_COLORS_ON_CT)]
    err = [Patch(facecolor=c, edgecolor="none", label=n) for n, c in STATUS_COLORS.items()]
    kw = dict(frameon=False, labelcolor="#f0efec", fontsize=9, handlelength=1.2, columnspacing=1.0)
    fig.legend(handles=lvl, loc="lower left", ncol=8, bbox_to_anchor=(0.02, 0.005), **kw)
    fig.legend(handles=err, loc="lower right", ncol=4, bbox_to_anchor=(0.98, 0.005), **kw)
    fig.tight_layout(rect=(0, 0.04, 1, 0.965))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


# --------------------------------------------------------------- summaries
def plot_dice_per_level(levels: list[dict[str, Any]], out_path: str | Path, split: str) -> Path:
    data = [[r["dice"] for r in levels if r["level"] == n and not np.isnan(r["dice"])] for n in LEVEL_NAMES]
    n_cases = len({r["scan_id"] for r in levels})
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
    _style(ax)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        if vals:
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(vals)), vals, s=16, color=SERIES_1, alpha=0.55,
                       linewidths=0, zorder=2)
    ax.boxplot([d if d else [np.nan] for d in data], positions=range(len(LEVEL_NAMES)), widths=0.5,
               showfliers=False, medianprops=dict(color=INK, linewidth=2),
               boxprops=dict(color=INK_2, linewidth=1), whiskerprops=dict(color=INK_2, linewidth=1),
               capprops=dict(color=INK_2, linewidth=1), zorder=3)
    for i, vals in enumerate(data):
        if vals:
            ax.text(i, 1.035, f"{np.mean(vals):.2f}", ha="center", va="bottom", fontsize=9, color=INK_2)
    ax.set_xticks(range(len(LEVEL_NAMES)), LEVEL_NAMES)
    ax.set_ylim(-0.02, 1.1)
    ax.set_ylabel("Dice", color=INK_2)
    ax.set_title(f"Dice per vertebral level ({split}, {n_cases} cases); mean above each box",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return Path(out_path)


def plot_level_assignment(assignment: np.ndarray, out_path: str | Path, split: str) -> Path:
    """Rows: reference level. Columns: label the prediction mostly gave it (or missed)."""
    cols = list(LEVEL_NAMES) + ["missed"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", (SURFACE,) + SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(8.5, 6.5), facecolor=SURFACE)
    vmax = max(int(assignment.max()), 1)
    ax.imshow(assignment, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(assignment.shape[0]):
        for j in range(assignment.shape[1]):
            v = int(assignment[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                        color="white" if v > 0.55 * vmax else INK)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(LEVEL_NAMES)), LEVEL_NAMES)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(LEVEL_NAMES)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0, colors=INK_2, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Label the prediction mostly gave this vertebra", color=INK_2)
    ax.set_ylabel("Reference level", color=INK_2)
    ax.set_title(f"Level assignment, cases per cell ({split}); diagonal = correct",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return Path(out_path)


def plot_spine_vs_level_dice(cases: list[dict[str, Any]], out_path: str | Path, split: str,
                             label_gap: float = 0.3) -> Path:
    """Spine Dice (any vertebra) vs per-level mean Dice: points far right of the diagonal are naming errors."""
    pts = [(c["spine_dice"], c["mean_dice"], c["scan_id"]) for c in cases
           if not (np.isnan(c["spine_dice"]) or np.isnan(c["mean_dice"]))]
    fig, ax = plt.subplots(figsize=(7, 6.5), facecolor=SURFACE)
    _style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle="--", zorder=1)
    if pts:
        xs, ys, names = zip(*pts)
        ax.scatter(xs, ys, s=36, color=SERIES_1, alpha=0.8, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        gaps = sorted(((x - y, x, y, n) for x, y, n in pts if x - y >= label_gap), reverse=True)[:8]
        placed: list[tuple[float, float]] = []
        for _, x, y, n in gaps:  # skip labels that would overlap one already drawn
            if any(abs(y - py) < 0.025 and abs(x - px) < 0.15 for px, py in placed):
                continue
            placed.append((x, y))
            ax.annotate(n, (x, y), xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                        fontsize=7.5, color=INK_2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Spine Dice (all target vertebrae as one class)", color=INK_2)
    ax.set_ylabel("Mean Dice over levels", color=INK_2)
    ax.set_title(f"Segmentation vs naming ({split}, {len(pts)} cases)\n"
                 "Below the diagonal: bone found, levels misnamed", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return Path(out_path)
