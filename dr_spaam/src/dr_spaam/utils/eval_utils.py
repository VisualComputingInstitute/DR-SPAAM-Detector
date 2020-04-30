import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F

from . import utils as u
from . import prec_rec_utils as pru

# For plotting using lab cluster server https://github.com/matplotlib/matplotlib/issues/3466/
plt.switch_backend('agg')


def cfg_to_model(cfg):
    if cfg['network'] == 'cutout':
        from ..model.drow import DROW
        model = DROW(num_scans=cfg['num_scans'],
                     num_pts=cfg['cutout_kwargs']['num_cutout_pts'],
                     focal_loss_gamma=cfg['focal_loss_gamma'],
                     pedestrian_only=cfg['pedestrian_only'])

    elif cfg['network'] == 'cutout_gating':
        from ..model.drow import TemporalDROW
        model = TemporalDROW(num_scans=cfg['num_scans'],
                             num_pts=cfg['cutout_kwargs']['num_cutout_pts'],
                             focal_loss_gamma=cfg['focal_loss_gamma'],
                             pedestrian_only=cfg['pedestrian_only'])

    elif cfg['network'] == 'cutout_spatial':
        from ..model.drow import SpatialDROW
        model = SpatialDROW(num_scans=cfg['num_scans'],
                            num_pts=cfg['cutout_kwargs']['num_cutout_pts'],
                            focal_loss_gamma=cfg['focal_loss_gamma'],
                            alpha=cfg['similarity_kwargs']['alpha'],
                            window_size=cfg['similarity_kwargs']['window_size'],
                            pedestrian_only=cfg['pedestrian_only'])

    elif cfg['network'] == 'fc2d':
        from ..model.polar_drow import PolarDROW
        model = PolarDROW(in_channel=1)

    elif cfg['network'] == 'fc2d_fea':
        raise NotImplementedError
        from ..model.polar_drow import PolarDROW
        model = PolarDROW(in_channel=cfg['cutout_kwargs']['num_cutout_pts'])

    elif cfg['network'] == 'fc1d':
        from ..model.fconv_drow import FConvDROW
        model = FConvDROW(in_channel=1)

    elif cfg['network'] == 'fc1d_fea':
        from ..model.fconv_drow import FConvDROW
        model = FConvDROW(in_channel=cfg['cutout_kwargs']['num_cutout_pts'])

    else:
        raise RuntimeError

    return model


def model_fn(model, data, rtn_result=False):
    tb_dict, rtn_dict = {}, {}

    net_input = data['input']
    net_input = torch.from_numpy(net_input).cuda(non_blocking=True).float()

    # Forward pass
    model_rtn = model(net_input)
    spatial_drow = len(model_rtn) == 3
    if spatial_drow:
        pred_cls, pred_reg, pred_sim = model_rtn
    else:
        pred_cls, pred_reg = model_rtn

    target_cls, target_reg = data['target_cls'], data['target_reg']
    target_cls = torch.from_numpy(target_cls).cuda(non_blocking=True).long()
    target_reg = torch.from_numpy(target_reg).cuda(non_blocking=True).float()

    n_batch, n_pts = target_cls.shape[:2]

    # cls loss
    target_cls = target_cls.view(n_batch * n_pts)
    pred_cls = pred_cls.view(n_batch * n_pts, -1)
    if pred_cls.shape[1] == 1:
        cls_loss = model.cls_loss(torch.sigmoid(pred_cls.squeeze(-1)),
                                  target_cls.float(),
                                  reduction='mean')
    else:
        cls_loss = model.cls_loss(pred_cls, target_cls, reduction='mean')
    total_loss = cls_loss
    tb_dict['cls_loss'] = cls_loss.item()

    # number fg points
    fg_mask = target_cls.ne(0)
    fg_ratio = torch.sum(fg_mask).item() / (n_batch * n_pts)
    tb_dict['fg_ratio'] = fg_ratio

    # reg loss
    if fg_ratio > 0.0:
        target_reg = target_reg.view(n_batch * n_pts, -1)
        pred_reg = pred_reg.view(n_batch * n_pts, -1)
        reg_loss = F.mse_loss(pred_reg[fg_mask], target_reg[fg_mask],
                              reduction='none')
        reg_loss = torch.sqrt(torch.sum(reg_loss, dim=1)).mean()
        total_loss = total_loss + reg_loss
        tb_dict['reg_loss'] = reg_loss.item()

    # # regularization loss for spatial attention
    # if spatial_drow:
    #     att_loss = (-torch.log(pred_sim + 1e-5) * pred_sim).sum(dim=2).mean()  # shannon entropy
    #     tb_dict['att_loss'] = att_loss.item()
    #     total_loss = total_loss + att_loss

    if rtn_result:
        rtn_dict["pred_reg"] = pred_reg.view(n_batch, n_pts, -1)
        rtn_dict["pred_cls"] = pred_cls.view(n_batch, n_pts, -1)

    return total_loss, tb_dict, rtn_dict


def eval_batch(model, data, vote_kwargs, full_eval=True):
    # forward pass
    _, tb_dict, rtn_dict = model_fn(model, data, rtn_result=full_eval)

    # only compute lost, not ap
    if not full_eval:
        return tb_dict, rtn_dict

    # get inference result to cpu
    pred_cls, pred_reg = rtn_dict['pred_cls'], rtn_dict['pred_reg']
    if pred_cls.shape[-1] == 1:
        pred_cls = torch.sigmoid(pred_cls).data.cpu().numpy()
    else:
        pred_cls = F.softmax(pred_cls, dim=-1).data.cpu().numpy()
    pred_reg = pred_reg.data.cpu().numpy()

    # grouping
    scan_grid, phi_grid = data['scans'], data['phi_grid']
    dets_xy_list, dets_cls_list, dets_inds_list = [], [], []
    for i, (s_g, p_g, p_cls, p_reg) in enumerate(
            zip(scan_grid, phi_grid, pred_cls, pred_reg)):
        # dets_xy, dets_cls, _ = u.group_predicted_center(s_g[-1], p_g, p_cls, p_reg,
        #                                                 **vote_kwargs)
        dets_xy, dets_cls, _ = u.nms_predicted_center(s_g[-1], p_g, p_cls, p_reg)
        if len(dets_xy) > 0:
            dets_xy_list.append(dets_xy)
            dets_cls_list.append(dets_cls)
            dets_inds_list = dets_inds_list + [i] * len(dets_cls)

    if len(dets_xy_list) > 0:
        rtn_dict.update({'dets_xy': np.concatenate(dets_xy_list, axis=0),
                        'dets_cls': np.concatenate(dets_cls_list, axis=0),
                        'dets_inds': np.array(dets_inds_list, dtype=np.int32)})

    return tb_dict, rtn_dict


def eval_epoch(model, test_loader, vote_kwargs, full_eval=True):
    model.eval()

    # hold all detections
    dets_xy_list, dets_cls_list, dets_inds_list = [], [], []

    # hold all ground truth
    gts_xy, gts_inds = {}, {}
    gts_xy['wc'], gts_xy['wa'], gts_xy['wp'], gts_xy['all'] = [], [], [], []
    gts_inds['wc'], gts_inds['wa'], gts_inds['wp'], gts_inds['all'] = [], [], [], []

    # hold all items for tb logging
    tb_dict = {}

    # inference over the whole test set, and collect results
    for it, data in enumerate(tqdm.tqdm(test_loader, desc='eval')):
        n_batch = len(data['scans'])
        it_global = it * n_batch

        # inference
        batch_tb_dict, batch_rtn_dict = eval_batch(model, data, vote_kwargs, full_eval)

        # store tb log
        for k, v in batch_tb_dict.items():
            tb_dict.setdefault(k, []).append(v)

        if not full_eval:
            continue

        # store detection
        if 'dets_xy' in batch_rtn_dict:
            dets_xy_list.append(batch_rtn_dict['dets_xy'])
            dets_cls_list.append(batch_rtn_dict['dets_cls'])
            dets_inds_list.append(batch_rtn_dict['dets_inds'] + it_global)

        # store gt
        for k in ['wc', 'wa', 'wp']:
            for j, j_gts in enumerate(data['dets_'+k]):
                for r, phi in j_gts:
                    xy = u.rphi_to_xy(r, phi)
                    gts_xy[k].append(xy)
                    gts_xy['all'].append(xy)
                    gts_inds[k].append(j + it_global)
                    gts_inds['all'].append(j + it_global)

    # compute loss
    for k, v in tb_dict.items():
        tb_dict[k] = np.array(v).mean()

    # only log training loss
    if not full_eval:
        return tb_dict, None, None

    # dets for the whole epoch
    dets_xy = np.concatenate(dets_xy_list, axis=0)  # (N, 2)
    dets_cls = np.concatenate(dets_cls_list, axis=0)  # (N, cls)
    dets_inds = np.concatenate(dets_inds_list)  # (N)

    # gts for the whole epoch
    for k, v in gts_xy.items():
        gts_xy[k] = np.array(v)
        gts_inds[k] = np.array(gts_inds[k], dtype=np.int32)

    # evaluation
    rpt_dict = {}
    dist_thresh = [0.3, 0.5, 0.7]
    for dt in dist_thresh:
        rpt_dict[dt] = {}

    # pedestrian only
    if dets_cls.shape[1] == 1:
        for dt in dist_thresh:
            rpt_dict[dt]['wp'] = compute_prec_rec(dets_xy, dets_cls[:, 0], dets_inds,
                                                  gts_xy['wp'], gts_inds['wp'], dt)
            ap, f1, eer = eval_prec_rec(*rpt_dict[dt]['wp'][:2])

            tb_dict["ap_wp_t%s" % dt] = ap
            tb_dict["f1_wp_t%s" % dt] = f1
            tb_dict["eer_wp_t%s" % dt] = eer

    # multi-class
    else:
        for dt in dist_thresh:
            for k in gts_xy.keys():
                if k == 'wc': d_cls = dets_cls[:, 1]
                elif k == 'wa': d_cls = dets_cls[:, 2]
                elif k == 'wp': d_cls = dets_cls[:, 3]
                elif k == 'all': d_cls = np.sum(dets_cls[:, 1:], axis=1)
                else: raise RuntimeError

                rpt_dict[dt][k] = compute_prec_rec(dets_xy, d_cls, dets_inds,
                                                   gts_xy[k], gts_inds[k], dt)
                ap, f1, eer = eval_prec_rec(*rpt_dict[dt][k][:2])

                tb_dict["ap_%s_t%s" % (k, dt)] = ap
                tb_dict["f1_%s_t%s" % (k, dt)] = f1
                tb_dict["eer_%s_t%s" % (k, dt)] = eer

    # also return network inference results
    fwd_dict = {}
    fwd_dict['dets'] = dets_xy
    fwd_dict['dets_inds'] = dets_inds
    fwd_dict['dets_cls'] = dets_cls
    fwd_dict['gts'] = gts_xy
    fwd_dict['gts_inds'] = gts_inds

    return tb_dict, rpt_dict, fwd_dict


def compute_prec_rec(dets, dets_cls, dets_inds, gts, gts_inds, dt):
    dt = dt * np.ones(len(gts_inds), dtype=np.float32)
    return pru.prec_rec_2d(dets_cls, dets, dets_inds, gts, gts_inds, dt)


def eval_prec_rec(rec, prec):
    return pru.eval_prec_rec(rec, prec)


def plot_prec_rec(rpt_dict, plot_title=None, output_file=None):
    pedestrian_only = 'all' not in rpt_dict.keys()
    if pedestrian_only:
        fig, ax = pru.plot_prec_rec_wps_only(wps=rpt_dict['wp'],
                                             title=plot_title)
    else:
        fig, ax = pru.plot_prec_rec(wds=rpt_dict['all'],
                                    wcs=rpt_dict['wc'],
                                    was=rpt_dict['wa'],
                                    wps=rpt_dict['wp'],
                                    title=plot_title)

    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight')

    return fig, ax


def eval_epoch_with_output(model, test_loader, epoch, it, root_result_dir, split, tag,
                           vote_kwargs, full_eval=True, writing=False, plotting=False,
                           save_pkl=False, tb_log=None):
    tb_dict, rpt_dict, fwd_dict = eval_epoch(
            model, test_loader, vote_kwargs=vote_kwargs,
            full_eval=full_eval)

    if writing:
        ap_dir = os.path.join(root_result_dir, 'results')
        os.makedirs(ap_dir, exist_ok=True)
        ap_file = os.path.join(ap_dir, '%s.csv' % split)
        for k, v in tb_dict.items():
            with open(ap_file, "a") as f:
                s = "%s, %s, %s, %s, %s, %s\n" % (tag, it, epoch, split, k, v)
                f.write(s)
            if tb_log is not None:
                stag = ("eval_%s" % split) if tag.startswith('eval_') else split
                tb_log.add_scalar("%s_%s" % (stag, k), v, it)

    if not full_eval:
        tb_log.flush()
        return

    if save_pkl:
        pkl_dir = os.path.join(root_result_dir, 'pkl')
        os.makedirs(pkl_dir, exist_ok=True)

        s = '%s_e%s_%s.pkl' % (tag, epoch, split)
        with open(os.path.join(pkl_dir, 'rpt_'+s), "wb") as f:
            pickle.dump(rpt_dict, f)
        with open(os.path.join(pkl_dir, 'fwd_'+s), "wb") as f:
            pickle.dump(fwd_dict, f)

    if plotting:
        for k, v in rpt_dict.items():
            fig_dir = os.path.join(root_result_dir, 'figs', split, 't_%s' % k)
            os.makedirs(fig_dir, exist_ok=True)
            plot_file = '%s_e%s_%s_t%s.png' % (tag, epoch, split, k)
            fig, ax = plot_prec_rec(v, output_file=os.path.join(fig_dir, plot_file))

            if tb_log is not None:
                fig.canvas.draw()
                im = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                im = im.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
                im = im.transpose(2, 0, 1)  # (3, H, W)
                im = im.astype(np.float32) / 255.0
                tb_log.add_image("pr_curve_t%s" % k, im, it)
                plt.close(fig)
            else:
                plt.close(fig)

    if tb_log is not None:
        tb_log.flush()
