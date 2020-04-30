# Most of the code here comes from
# https://github.com/VisualComputingInstitute/DROW/blob/master/v2/utils/__init__.py
from collections import defaultdict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import auc

# For plotting using lab cluster server
# https://github.com/matplotlib/matplotlib/issues/3466/
plt.switch_backend('agg')


def prec_rec_2d(det_scores, det_coords, det_frames, gt_coords, gt_frames, gt_radii):
    """ Computes full precision-recall curves at all possible thresholds.

    Arguments:
    - `det_scores` (D,) array containing the scores of the D detections.
    - `det_coords` (D,2) array containing the (x,y) coordinates of the D detections.
    - `det_frames` (D,) array containing the frame number of each of the D detections.
    - `gt_coords` (L,2) array containing the (x,y) coordinates of the L labels (ground-truth detections).
    - `gt_frames` (L,) array containing the frame number of each of the L labels.
    - `gt_radii` (L,) array containing the radius at which each of the L labels should consider detection associations.
                      This will typically just be an np.full_like(gt_frames, 0.5) or similar,
                      but could vary when mixing classes, for example.

    Returns: (recs, precs, threshs)
    - `threshs`: (D,) array of sorted thresholds (scores), from higher to lower.
    - `recs`: (D,) array of recall scores corresponding to the thresholds.
    - `precs`: (D,) array of precision scores corresponding to the thresholds.
    """
    # This means that all reported detection frames which are not in ground-truth frames
    # will be counted as false-positives.
    # TODO: do some sanity-checks in the "linearization" functions before calling `prec_rec_2d`.
    frames = np.unique(np.r_[det_frames, gt_frames])

    det_accepted_idxs = defaultdict(list)
    tps = np.zeros(len(frames), dtype=np.uint32)
    fps = np.zeros(len(frames), dtype=np.uint32)
    fns = np.array([np.sum(gt_frames == f) for f in frames], dtype=np.uint32)

    precs = np.full_like(det_scores, np.nan)
    recs = np.full_like(det_scores, np.nan)
    threshs = np.full_like(det_scores, np.nan)

    indices = np.argsort(det_scores, kind='mergesort')  # mergesort for determinism.
    for i, idx in enumerate(reversed(indices)):
        frame = det_frames[idx]
        iframe = np.where(frames == frame)[0][0]  # Can only be a single one.

        # Accept this detection
        dets_idxs = det_accepted_idxs[frame]
        dets_idxs.append(idx)
        threshs[i] = det_scores[idx]

        dets = det_coords[dets_idxs]

        gts_mask = gt_frames == frame
        gts = gt_coords[gts_mask]
        radii = gt_radii[gts_mask]

        if len(gts) == 0:  # No GT, but there is a detection.
            fps[iframe] += 1
        else:              # There is GT and detection in this frame.
            not_in_radius = radii[:,None] < cdist(gts, dets)  # -> ngts x ndets, True (=1) if too far, False (=0) if may match.
            igt, idet = linear_sum_assignment(not_in_radius)

            tps[iframe] = np.sum(np.logical_not(not_in_radius[igt, idet]))  # Could match within radius
            fps[iframe] = len(dets) - tps[iframe]  # NB: dets is only the so-far accepted.
            fns[iframe] = len(gts) - tps[iframe]

        tp, fp, fn = np.sum(tps), np.sum(fps), np.sum(fns)
        precs[i] = tp/(fp+tp) if fp+tp > 0 else np.nan
        recs[i] = tp/(fn+tp) if fn+tp > 0 else np.nan

    return recs, precs, threshs


def eval_prec_rec(rec, prec):
    # make sure the x-input to auc is sorted
    assert np.sum(np.diff(rec)>=0) == len(rec) - 1
    # compute error matrices
    return auc(rec, prec), peakf1(rec, prec), eer(rec, prec)


def peakf1(recs, precs):
    return np.max(2 * precs * recs / np.clip(precs + recs, 1e-16, 2 + 1e-16))


def eer(recs, precs):
    # Find the first nonzero or else (0,0) will be the EER :)
    def first_nonzero_idx(arr):
        return np.where(arr != 0)[0][0]

    p1 = first_nonzero_idx(precs)
    r1 = first_nonzero_idx(recs)
    idx = np.argmin(np.abs(precs[p1:] - recs[r1:]))
    return (precs[p1 + idx] + recs[r1 + idx]) / 2  # They are often the exact same, but if not, use average.


def plot_prec_rec(wds, wcs, was, wps, figsize=(15,10), title=None):
    fig, ax = plt.subplots(figsize=figsize)

    # make sure the x-input to auc is sorted
    assert np.sum(np.diff(wds[0])>=0) == len(wds[0]) - 1
    assert np.sum(np.diff(wcs[0])>=0) == len(wcs[0]) - 1
    assert np.sum(np.diff(was[0])>=0) == len(was[0]) - 1
    assert np.sum(np.diff(wps[0])>=0) == len(wps[0]) - 1

    ax.plot(*wds[:2], label='agn (AUC: {:.1%}, F1: {:.1%}, EER: {:.1%})'.format(auc(*wds[:2]), peakf1(*wds[:2]), eer(*wds[:2])), c='#E24A33')
    ax.plot(*wcs[:2], label='wcs (AUC: {:.1%}, F1: {:.1%}, EER: {:.1%})'.format(auc(*wcs[:2]), peakf1(*wcs[:2]), eer(*wcs[:2])), c='#348ABD')
    ax.plot(*was[:2], label='was (AUC: {:.1%}, F1: {:.1%}, EER: {:.1%})'.format(auc(*was[:2]), peakf1(*was[:2]), eer(*was[:2])), c='#988ED5')
    ax.plot(*wps[:2], label='wps (AUC: {:.1%}, F1: {:.1%}, EER: {:.1%})'.format(auc(*wps[:2]), peakf1(*wps[:2]), eer(*wps[:2])), c='#8EBA42')

    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.91)

    _prettify_pr_curve(ax)
    _lbplt_fatlegend(ax, loc='upper right')

    return fig, ax


def plot_prec_rec_wps_only(wps, figsize=(15,10), title=None):
    fig, ax = plt.subplots(figsize=figsize)

    # make sure the x-input to auc is sorted
    assert np.sum(np.diff(wps[0])>=0) == len(wps[0]) - 1

    ax.plot(*wps[:2], label='wps (AUC: {:.1%}, F1: {:.1%}, EER: {:.1%})'.format(auc(*wps[:2]), peakf1(*wps[:2]), eer(*wps[:2])), c='#8EBA42')

    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.91)

    _prettify_pr_curve(ax)
    _lbplt_fatlegend(ax, loc='upper right')
    return fig, ax


def _prettify_pr_curve(ax):
    ax.plot([0,1], [0,1], ls="--", c=".6")
    ax.set_xlim(-0.02,1.02)
    ax.set_ylim(-0.02,1.02)
    ax.set_xlabel("Recall [%]")
    ax.set_ylabel("Precision [%]")
    ax.axes.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x*100)))
    ax.axes.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x*100)))
    return ax


def _lbplt_fatlegend(ax=None, *args, **kwargs):
    # Copy paste from lbtoolbox.plotting.fatlegend
    if ax is not None:
        leg = ax.legend(*args, **kwargs)
    else:
        leg = plt.legend(*args, **kwargs)

    for l in leg.legendHandles:
        l.set_linewidth(l.get_linewidth()*2.0)
        l.set_alpha(1)
    return leg
