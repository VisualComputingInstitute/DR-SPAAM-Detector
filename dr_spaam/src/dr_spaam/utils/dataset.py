from glob import glob
import os

import json
import numpy as np
from torch.utils.data import Dataset, DataLoader

from . import utils as u


def create_dataloader(data_path, num_scans, batch_size, num_workers, network_type="cutout",
                      train_with_val=False, use_data_augumentation=False,
                      cutout_kwargs=None, polar_grid_kwargs=None,
                      pedestrian_only=False):
    train_set = DROWDataset(data_path=data_path,
                            split='train',
                            num_scans=num_scans,
                            network_type=network_type,
                            train_with_val=train_with_val,
                            use_data_augumentation=use_data_augumentation,
                            cutout_kwargs=cutout_kwargs,
                            polar_grid_kwargs=polar_grid_kwargs,
                            pedestrian_only=pedestrian_only)
    eval_set = DROWDataset(data_path=data_path,
                           split='val',
                           num_scans=num_scans,
                           network_type=network_type,
                           train_with_val=False,
                           use_data_augumentation=False,
                           cutout_kwargs=cutout_kwargs,
                           polar_grid_kwargs=polar_grid_kwargs,
                           pedestrian_only=pedestrian_only)
    train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True,
                              num_workers=num_workers, shuffle=True,
                              collate_fn=train_set.collate_batch)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, pin_memory=True,
                             num_workers=num_workers, shuffle=True,
                             collate_fn=eval_set.collate_batch)
    return train_loader, eval_loader


def create_test_dataloader(data_path, num_scans, network_type="cutout",
                           cutout_kwargs=None, polar_grid_kwargs=None,
                           pedestrian_only=False, split='test',
                           scan_stride=1, pt_stride=1):
    test_set = DROWDataset(data_path=data_path,
                           split=split,
                           num_scans=num_scans,
                           network_type=network_type,
                           train_with_val=False,
                           use_data_augumentation=False,
                           cutout_kwargs=cutout_kwargs,
                           polar_grid_kwargs=polar_grid_kwargs,
                           pedestrian_only=pedestrian_only,
                           scan_stride=scan_stride,
                           pt_stride=pt_stride)
    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True,
                             num_workers=1, shuffle=False,
                             collate_fn=test_set.collate_batch)
    return test_loader


class DROWDataset(Dataset):
    def __init__(self, data_path, split='train', num_scans=5, network_type="cutout",
                 train_with_val=False, cutout_kwargs=None, polar_grid_kwargs=None,
                 use_data_augumentation=False, pedestrian_only=False,
                 scan_stride=1, pt_stride=1):
        self._num_scans = num_scans
        self._use_data_augmentation = use_data_augumentation
        self._cutout_kwargs = cutout_kwargs
        self._network_type = network_type
        self._polar_grid_kwargs = polar_grid_kwargs
        self._pedestrian_only = pedestrian_only
        self._scan_stride = scan_stride
        self._pt_stride = pt_stride  # @TODO remove pt_stride

        if train_with_val:
            seq_names = [f[:-4] for f in glob(os.path.join(data_path, 'train', '*.csv'))]
            seq_names += [f[:-4] for f in glob(os.path.join(data_path, 'val', '*.csv'))]
        else:
            seq_names = [f[:-4] for f in glob(os.path.join(data_path, split, '*.csv'))]

        # seq_names = seq_names[:1]
        self.seq_names = seq_names

        # Pre-load scans and annotations
        self.scans_ns, self.scans_t, self.scans = zip(*[self._load_scan_file(f) for f in seq_names])
        self.dets_ns, self.dets_wc, self.dets_wa, self.dets_wp = zip(*map(
            lambda f: self._load_det_file(f), seq_names))

        # Pre-compute mappings from detection index to scan index
        # such that idet2iscan[seq_idx][det_idx] = scan_idx
        self.idet2iscan = [{i: np.where(ss == d)[0][0] for i, d in enumerate(ds)}
                for ss, ds in zip(self.scans_ns, self.dets_ns)]

        # Look-up list for sequence indices and annotation indices.
        self.flat_seq_inds, self.flat_det_inds = [], []
        for seq_idx, det_ns in enumerate(self.dets_ns):
            num_samples = len(det_ns)
            self.flat_seq_inds += [seq_idx] * num_samples
            self.flat_det_inds += range(num_samples)

    def __len__(self):
        return len(self.flat_det_inds)

    def __getitem__(self, idx):
        seq_idx = self.flat_seq_inds[idx]
        det_idx = self.flat_det_inds[idx]
        dets_ns = self.dets_ns[seq_idx][det_idx]

        rtn_dict = {}
        rtn_dict['seq_name'] = self.seq_names[seq_idx]
        rtn_dict['dets_ns'] = dets_ns

        # Annotation
        rtn_dict['dets_wc'] = self.dets_wc[seq_idx][det_idx]
        rtn_dict['dets_wa'] = self.dets_wa[seq_idx][det_idx]
        rtn_dict['dets_wp'] = self.dets_wp[seq_idx][det_idx]

        # Scan
        scan_idx = self.idet2iscan[seq_idx][det_idx]
        inds_tmp = (np.arange(self._num_scans) * self._scan_stride)[::-1]
        scan_inds = [max(0, scan_idx - i) for i in inds_tmp]
        scans = np.array([self.scans[seq_idx][i] for i in scan_inds])
        scans = scans[:, ::self._pt_stride]
        scans_ns = [self.scans_ns[seq_idx][i] for i in scan_inds]
        rtn_dict['scans'] = scans
        rtn_dict['scans_ns'] = scans_ns

        # angle
        scan_phi = u.get_laser_phi()[::self._pt_stride]
        rtn_dict['phi_grid'] = scan_phi

        # Regression target
        target_cls, target_reg = u.get_regression_target(
                scans[-1],
                scan_phi,
                rtn_dict['dets_wc'],
                rtn_dict['dets_wa'],
                rtn_dict['dets_wp'],
                pedestrian_only=self._pedestrian_only)

        rtn_dict['target_cls'] = target_cls
        rtn_dict['target_reg'] = target_reg

        if self._use_data_augmentation:
            rtn_dict = u.data_augmentation(rtn_dict)

        # polar grid or cutout
        if self._network_type == "cutout" \
                or self._network_type == "cutout_gating" \
                or self._network_type == "cutout_spatial":
            if "area_mode" not in self._cutout_kwargs:
                cutout = u.scans_to_cutout_original(
                    scans, scan_phi[1] - scan_phi[0],
                    **self._cutout_kwargs)
            else:
                cutout = u.scans_to_cutout(scans, scan_phi, stride=1,
                                           **self._cutout_kwargs)
            rtn_dict['input'] = cutout
        elif self._network_type == "fc1d":
            rtn_dict['input'] = np.expand_dims(scans, axis=1)
        elif self._network_type == 'fc1d_fea':
            cutout = u.scans_to_cutout(rtn_dict['scans'],
                                       scan_phi[1] - scan_phi[0],
                                       **self._cutout_kwargs)
            rtn_dict['input'] = np.transpose(cutout, (1, 2, 0))
        elif self._network_type == "fc2d":
            polar_grid = u.scans_to_polar_grid(rtn_dict['scans'],
                                               **self._polar_grid_kwargs)
            rtn_dict['input'] = np.expand_dims(polar_grid, axis=1)
        elif self._network_type == 'fc2d_fea':
            raise NotImplementedError

        return rtn_dict

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, _ in batch[0].items():
            if k in ["target_cls", "target_reg", "input"]:
                rtn_dict[k] = np.array([sample[k] for sample in batch])
            else:
                rtn_dict[k] = [sample[k] for sample in batch]

        return rtn_dict

    def _load_scan_file(self, seq_name):
        data = np.genfromtxt(seq_name + '.csv', delimiter=",")
        seqs = data[:, 0].astype(np.uint32)
        times = data[:, 1].astype(np.float32)
        scans = data[:, 2:].astype(np.float32)
        return seqs, times, scans

    def _load_det_file(self, seq_name):
        def do_load(f_name):
            seqs, dets = [], []
            with open(f_name) as f:
                for line in f:
                    seq, tail = line.split(',', 1)
                    seqs.append(int(seq))
                    dets.append(json.loads(tail))
            return seqs, dets

        s1, wcs = do_load(seq_name + '.wc')
        s2, was = do_load(seq_name + '.wa')
        s3, wps = do_load(seq_name + '.wp')
        assert all(a == b == c for a, b, c in zip(s1, s2, s3))

        return np.array(s1), wcs, was, wps


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = DROWDataset(data_path='../data/DROWv2-data')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for sample in dataset:
        target_cls, target_reg = sample['target_cls'], sample['target_reg']
        scans = sample['scans']
        scan_phi = u.get_laser_phi()

        num_scans = scans.shape[0]
        for scan_idx in range(1):
            scan_x, scan_y = u.scan_to_xy(scans[-scan_idx])

            plt.cla()
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.scatter(scan_x, scan_y, s=1, c='black')

            colors = ['blue', 'green', 'red']
            cls_labels = [1, 2, 3]
            for cls_label, c in zip(cls_labels, colors):
                canonical_dxy = target_reg[target_cls==cls_label]
                dets_r, dets_phi = u.canonical_to_global(
                        scans[-1][target_cls==cls_label],
                        scan_phi[target_cls==cls_label],
                        canonical_dxy[:, 0],
                        canonical_dxy[:, 1])
                dets_x, dets_y = u.rphi_to_xy(dets_r, dets_phi)
                ax.scatter(dets_x, dets_y, s=5, c=c)

            plt.pause(0.1)

    plt.show()
