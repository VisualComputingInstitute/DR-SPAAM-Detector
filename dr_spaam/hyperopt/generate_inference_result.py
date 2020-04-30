import pickle
from tqdm import tqdm
import yaml
import numpy as np
import torch

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import utils.utils as u
import utils.eval_utils as eu
from utils.dataset import create_test_dataloader


if __name__=='__main__':
    cfg_file = './cfgs/NCT_cfgs/STEP_bl_5.yaml'
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg['name'] = os.path.basename(cfg_file).split(".")[0] + cfg['tag']

    model = eu.cfg_to_model(cfg)
    model.cuda()

    ckpt_file = './output/%s/ckpts/ckpt_e40.pth' % cfg['name']
    ckpt = torch.load(ckpt_file)
    model.load_state_dict(ckpt['model_state'])

    test_loader = create_test_dataloader(data_path="../data/DROWv2-data",
                                         num_scans=cfg['num_scans'],
                                         network_type=cfg['network'],
                                         cutout_kwargs=cfg['cutout_kwargs'],
                                         polar_grid_kwargs=cfg['polar_grid_kwargs'],
                                         pedestrian_only=cfg['pedestrian_only'],
                                         split='val',
                                         scan_stride=1,
                                         pt_stride=1)

    scan_list, pred_cls_list, pred_reg_list, gts_xy_list, gts_inds_list = [], [], [], [], []
    for i, data in enumerate(tqdm(test_loader)):
        model.eval()

        input = torch.from_numpy(data['input']).cuda(non_blocking=True).float()
        with torch.no_grad():
            model_rtn = model(input)

        if len(model_rtn) == 3:
            pred_cls, pred_reg, _ = model_rtn
        else:
            pred_cls, pred_reg = model_rtn

        pred_cls = torch.sigmoid(pred_cls[0]).data.cpu().numpy()
        pred_reg = pred_reg[0].data.cpu().numpy()

        pred_cls_list.append(pred_cls)
        pred_reg_list.append(pred_reg)

        for gt in data['dets_wp'][0]:
            xy = u.rphi_to_xy(gt[0], gt[1])
            gts_xy_list.append(np.array(xy))
            gts_inds_list.append(i)

        scan_list.append(data['scans'][0][-1])

    scans = np.stack(scan_list, axis=0)
    pred_cls = np.stack(pred_cls_list, axis=0)
    pred_reg = np.stack(pred_reg_list, axis=0)
    gts_xy = np.stack(gts_xy_list, axis=0)
    gts_inds = np.array(gts_inds_list)

    pkl_file = './hyperopt/inference_result_%s.pkl' % cfg['name']
    with open(pkl_file, 'wb') as f:
        pickle.dump([scans, pred_cls, pred_reg, gts_xy, gts_inds], f)
