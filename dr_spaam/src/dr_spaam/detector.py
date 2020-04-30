import torch
import numpy as np
from .utils import utils as u
from .model.drow import DROW, SpatialDROW


class Detector(object):
    def __init__(self, ckpt_file, original_drow=False, gpu=True, stride=1):
        self._gpu, self._scan_phi, self._stride = gpu, None, stride
        self._original_drow = original_drow

        self._ct_kwargs = {
            "fixed": False,
            "centered": True,
            "window_width": 1.0,
            "window_depth": 0.5,
            "num_cutout_pts": 56,
            "padding_val": 29.99,
            "area_mode": True
        }

        if original_drow:
            model = DROW(num_scans=1, 
                         num_pts=self._ct_kwargs['num_cutout_pts'], 
                         pedestrian_only=True)
        else:
            model = SpatialDROW(num_pts=self._ct_kwargs['num_cutout_pts'],
                                pedestrian_only=True,
                                alpha=0.5,
                                window_size=7)

        ckpt = torch.load(ckpt_file)
        model.load_state_dict(ckpt['model_state'])

        model.eval()
        self._model = model.cuda() if gpu else model

        if not original_drow:
            self._fea, self._sim = None, None
            # for tracking
            self._prev_dets_xy, self._prev_dets_cls = None, None
            self._prev_instance_mask, self._prev_dets_to_tracks = None, None
            self._tracks, self._tracks_cls, self._tracks_age = [], [], []
            self._max_track_age = 100
            self._max_assoc_dist = 0.7

    def __call__(self, scan, tracking=False):
        assert self.laser_spec_set(), "Need to call set_laser_spec() first."

        # preprocess
        ct = u.scans_to_cutout(
            scan[None, ...], self._scan_phi,
            stride=self._stride, **self._ct_kwargs)
        ct = torch.from_numpy(ct).float()

        if self._gpu:
            ct = ct.cuda()

        # inference
        with torch.no_grad():
            if self._original_drow:
                pred_cls, pred_reg = self._model(ct.unsqueeze(dim=0))  # one dim for batch
            else:
                pred_cls, pred_reg, self._fea, self._sim = self._model(
                    ct.unsqueeze(dim=0), testing=True, fea_template=self._fea)
        pred_cls = torch.sigmoid(pred_cls[0]).data.cpu().numpy()
        pred_reg = pred_reg[0].data.cpu().numpy()

        # postprocess
        dets_xy, dets_cls, instance_mask = u.nms_predicted_center(
            scan, self._scan_phi, pred_cls, pred_reg)

        if tracking:
            self._update_tracklets(dets_xy, dets_cls, instance_mask)

        return dets_xy, dets_cls, instance_mask

    def get_tracklets(self):
        tracks, tracks_cls = [], []
        for i in range(len(self._tracks)):
            if self._tracks_age[i] < self._max_track_age:
                tracks.append(np.stack(self._tracks[i], axis=0))
                tracks_cls.append(np.array(self._tracks_cls[i]).mean())
        return tracks, tracks_cls

    def _update_tracklets(self, dets_xy, dets_cls, instance_mask):
        assert not self._original_drow, "Must use DR-SPAAM model"

        # first frame
        if self._prev_dets_xy is None:
            self._prev_dets_xy = dets_xy
            self._prev_dets_cls = dets_cls
            self._prev_instance_mask = instance_mask
            self._prev_dets_to_tracks = np.arange(len(dets_xy), dtype=np.int32)

            for d_xy, d_cls in zip(dets_xy, dets_cls):
                self._tracks.append([d_xy])
                self._tracks_cls.append([np.asscalar(d_cls)])
                self._tracks_age.append(0)

            return

        # associate detections
        prev_dets_inds = self._associate_prev_det(
            dets_xy, dets_cls, instance_mask)

        # mapping from detection indices to tracklets indices
        dets_to_tracks = []

        # assign current detections to tracks based on assocation with previous
        # detections
        for d_idx, (d_xy, d_cls, prev_d_idx) in enumerate(
                zip(dets_xy, dets_cls, prev_dets_inds)):
            # distance between assocated detections
            dxy = self._prev_dets_xy[prev_d_idx] - d_xy
            dxy = np.hypot(dxy[0], dxy[1])

            if dxy < self._max_assoc_dist and prev_d_idx >= 0:
                # if current detection is close to the associated detection,
                # append to the tracklet
                ti = self._prev_dets_to_tracks[prev_d_idx]
                self._tracks[ti].append(d_xy)
                self._tracks_cls[ti].append(np.asscalar(d_cls))
                self._tracks_age[ti] = -1
                dets_to_tracks.append(ti)
            else:
                # otherwise start a new tracklet
                self._tracks.append([d_xy])
                self._tracks_cls.append([np.asscalar(d_cls)])
                self._tracks_age.append(-1)
                dets_to_tracks.append(len(self._tracks) - 1)

        # tracklet age
        for i in range(len(self._tracks_age)):
            self._tracks_age[i] += 1

        # # prune inactive tracks
        # pop_inds = []
        # for i in range(len(self._tracks_age)):
        #     self._tracks_age[i] = self._tracks_age[i] + 1
        #     if self._tracks_age[i] > self._track_len:
        #         pop_inds.append(i)

        # if len(pop_inds) > 0:
        #     pop_inds.reverse()
        #     for pi in pop_inds:
        #         for j in range(len(dets_to_tracks)):
        #             if dets_to_tracks[j] == pi:
        #                 dets_to_tracks[j] = -1
        #             elif dets_to_tracks[j] > pi:
        #                 dets_to_tracks[j] = dets_to_tracks[j] - 1
        #         self._tracks.pop(pi)
        #         self._tracks_cls.pop(pi)
        #         self._tracks_age.pop(pi)

        # update
        self._prev_dets_xy = dets_xy
        self._prev_dets_cls = dets_cls
        self._prev_instance_mask = instance_mask
        self._prev_dets_to_tracks = dets_to_tracks

    def _associate_prev_det(self, dets_xy, dets_cls, instance_mask):
        prev_dets_inds = []
        occupied_flag = np.zeros(len(self._prev_dets_xy), dtype=np.bool)
        sim = self._sim[0].data.cpu().numpy()
        for d_idx, (d_xy, d_cls) in enumerate(zip(dets_xy, dets_cls)):
            inst_id = d_idx + 1  # instance is 1-based

            # For all the points that belong to the current instance, find their
            # most similar points in the previous scans and take the point with
            # highest support as the associated point of this instance in the 
            # previous scan.
            inst_sim = sim[instance_mask == inst_id].argmax(axis=1)
            assoc_prev_pt_inds = np.bincount(inst_sim).argmax()

            # associated detection
            prev_d_idx = self._prev_instance_mask[assoc_prev_pt_inds] - 1  # instance is 1-based

            # only associate one detection
            if occupied_flag[prev_d_idx]:
                prev_dets_inds.append(-1)
            else:
                prev_dets_inds.append(prev_d_idx)
                occupied_flag[prev_d_idx] = True

        return prev_dets_inds

    def set_laser_spec(self, angle_inc, num_pts):
        self._scan_phi = u.get_laser_phi(angle_inc, num_pts)[::self._stride]

    def laser_spec_set(self):
        return self._scan_phi is not None
