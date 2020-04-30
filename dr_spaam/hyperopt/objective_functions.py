import json
import hyperopt as hp
import numpy as np

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import utils.utils as u
import utils.eval_utils as eu


def objective(vote_kwargs, scans, pred_cls, pred_reg, gts_xy, gts_inds):
    # get detection
    scan_phi = u.get_laser_phi()
    dets_xy_list, dets_cls_list, dets_inds_list = [], [], []
    for i, (scan, p_cls, p_reg) in enumerate(zip(scans, pred_cls, pred_reg)):
        dets_xy, dets_cls, _ = u.group_predicted_center(
                scan, scan_phi, p_cls, p_reg, **vote_kwargs)

        for xy, c in zip(dets_xy, dets_cls):
            dets_xy_list.append(xy)
            dets_cls_list.append(c)
            dets_inds_list.append(i)

    dets_xy = np.array(dets_xy_list)
    dets_cls = np.array(dets_cls_list)
    dets_inds = np.array(dets_inds_list)

    # compute precision recall
    eval_radius = 0.5
    rpt_tuple = eu.compute_prec_rec(dets_xy, dets_cls[:, 0], dets_inds,
                                    gts_xy, gts_inds, eval_radius)
    ap, f1, eer = eu.eval_prec_rec(*rpt_tuple[:2])

    # objective, maximize AP_0.5 for pedestrian class
    rtn_dict = {'loss': -ap,
                'status': hp.STATUS_OK,
                'real_attachments': {'kw': json.dumps(vote_kwargs).encode('utf-8'),
                                     'auc': json.dumps(ap).encode('utf-8')}}

    return rtn_dict

