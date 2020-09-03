import argparse
import time
import numpy as np

# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

from dr_spaam.detector import Detector
import dr_spaam.utils.utils as u


def inference_time():
    seq_name = './data/DROWv2-data/test/run_t_2015-11-26-11-55-45.bag.csv'
    scans = np.genfromtxt(seq_name, delimiter=',')[:, 2:]

    # inference time
    use_gpu = True
    model_names = ("DR-SPAAM", "DROW", "DROW-T5")
    ckpts = (
        "./ckpts/dr_spaam_e40.pth",
        "./ckpts/drow_e40.pth",
        "./ckpts/drow5_e40.pth"
    )
    for model_name, ckpt in zip(model_names, ckpts):
        detector = Detector(model_name=model_name, ckpt_file=ckpt, gpu=use_gpu, stride=1)
        detector.set_laser_spec(angle_inc=np.radians(0.5), num_pts=450)

        t_list = []
        for i in range(60):
            s = scans[i:i+5] if model_name == "DROW-T5" else scans[i]
            t0 = time.time()
            dets_xy, dets_cls, instance_mask = detector(s)
            t_list.append(1e3 * (time.time() - t0))

        t = np.array(t_list[10:]).mean()
        print("inference time (model: %s, gpu: %s): %f ms (%.1f FPS)" % (
            model_name, use_gpu, t, 1e3 / t))


def play_sequence():
    # scans
    seq_name = './data/DROWv2-data/test/run_t_2015-11-26-11-22-03.bag.csv'
    # seq_name = './data/DROWv2-data/val/run_2015-11-26-15-52-55-k.bag.csv'
    scans_data = np.genfromtxt(seq_name, delimiter=',')
    scans_t = scans_data[:, 1]
    scans = scans_data[:, 2:]
    scan_phi = u.get_laser_phi()    

    # odometry, used only for plotting
    odo_name = seq_name[:-3] + 'odom2'
    odos = np.genfromtxt(odo_name, delimiter=',')
    odos_t = odos[:, 1]
    odos_phi = odos[:, 4]

    # detector
    ckpt = './ckpts/dr_spaam_e40.pth'
    detector = Detector(model_name="DR-SPAAM", ckpt_file=ckpt, gpu=True, stride=1)
    detector.set_laser_spec(angle_inc=np.radians(0.5), num_pts=450)

    # scanner location
    rad_tmp = 0.5 * np.ones(len(scan_phi), dtype=np.float)
    xy_scanner = u.rphi_to_xy(rad_tmp, scan_phi)
    xy_scanner = np.stack(xy_scanner, axis=1)

    # plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    _break = False

    def p(event):
        nonlocal _break
        _break = True
    fig.canvas.mpl_connect('key_press_event', p)

    # video sequence
    odo_idx = 0
    for i in range(len(scans)):
    # for i in range(0, len(scans), 20):
        plt.cla()

        ax.set_aspect('equal')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)

        # ax.set_title('Frame: %s' % i)
        ax.set_title('Press any key to exit.')
        ax.axis("off")

        # find matching odometry
        while odo_idx < len(odos_t) - 1 and odos_t[odo_idx] < scans_t[i]:
            odo_idx += 1
        odo_phi = odos_phi[odo_idx]
        odo_rot = np.array([[np.cos(odo_phi), np.sin(odo_phi)],
                            [-np.sin(odo_phi), np.cos(odo_phi)]], dtype=np.float32)

        # plot scanner location
        xy_scanner_rot = np.matmul(xy_scanner, odo_rot.T)
        ax.plot(xy_scanner_rot[:, 0], xy_scanner_rot[:, 1], c='black')
        ax.plot((0, xy_scanner_rot[0, 0] * 1.0), (0, xy_scanner_rot[0, 1] * 1.0), c='black')
        ax.plot((0, xy_scanner_rot[-1, 0] * 1.0), (0, xy_scanner_rot[-1, 1] * 1.0), c='black')

        # plot points
        scan = scans[i]
        scan_x, scan_y = u.rphi_to_xy(scan, scan_phi + odo_phi)
        ax.scatter(scan_x, scan_y, s=1, c='blue')

        # inference
        dets_xy, dets_cls, instance_mask = detector(scan)

        # plot detection
        dets_xy_rot = np.matmul(dets_xy, odo_rot.T)
        cls_thresh = 0.5
        for j in range(len(dets_xy)):
            if dets_cls[j] < cls_thresh:
                continue
            # c = plt.Circle(dets_xy_rot[j], radius=0.5, color='r', fill=False)
            c = plt.Circle(dets_xy_rot[j], radius=0.5, color='r', fill=False, linewidth=2)
            ax.add_artist(c)

        # plt.savefig('/home/dan/tmp/det_img/frame_%04d.png' % i)

        plt.pause(0.001)
    
        if _break:
            break


def play_sequence_with_tracking():
    # scans
    seq_name = './data/DROWv2-data/train/lunch_2015-11-26-12-04-23.bag.csv'
    seq0, seq1 = 109170, 109360
    scans, scans_t = [], []
    with open(seq_name) as f:
        for line in f:
            scan_seq, scan_t, scan = line.split(",", 2)
            scan_seq = int(scan_seq)
            if scan_seq < seq0: 
                continue
            scans.append(np.fromstring(scan, sep=','))
            scans_t.append(float(scan_t))
            if scan_seq > seq1: 
                break
    scans = np.stack(scans, axis=0)
    scans_t = np.array(scans_t)
    scan_phi = u.get_laser_phi()

    # odometry, used only for plotting
    odo_name = seq_name[:-3] + 'odom2'
    odos = np.genfromtxt(odo_name, delimiter=',')
    odos_t = odos[:, 1]
    odos_phi = odos[:, 4]

    # detector
    ckpt = './ckpts/dr_spaam_e40.pth'
    detector = Detector(model_name="DR-SPAAM", ckpt_file=ckpt, gpu=True, stride=1, tracking=True)
    detector.set_laser_spec(angle_inc=np.radians(0.5), num_pts=450)

    # scanner location
    rad_tmp = 0.5 * np.ones(len(scan_phi), dtype=np.float)
    xy_scanner = u.rphi_to_xy(rad_tmp, scan_phi)
    xy_scanner = np.stack(xy_scanner, axis=1)

    # plot
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(111)

    _break = False

    def p(event):
        nonlocal _break
        _break = True
    fig.canvas.mpl_connect('key_press_event', p)

    # video sequence
    odo_idx = 0
    for i in range(len(scans)):
        plt.cla()

        ax.set_aspect('equal')
        ax.set_xlim(-10, 5)
        ax.set_ylim(-5, 15)

        # ax.set_title('Frame: %s' % i)
        ax.set_title('Press any key to exit.')
        ax.axis("off")

        # find matching odometry
        while odo_idx < len(odos_t) - 1 and odos_t[odo_idx] < scans_t[i]:
            odo_idx += 1
        odo_phi = odos_phi[odo_idx]
        odo_rot = np.array([[np.cos(odo_phi), np.sin(odo_phi)],
                            [-np.sin(odo_phi), np.cos(odo_phi)]], dtype=np.float32)

        # plot scanner location
        xy_scanner_rot = np.matmul(xy_scanner, odo_rot.T)
        ax.plot(xy_scanner_rot[:, 0], xy_scanner_rot[:, 1], c='black')
        ax.plot((0, xy_scanner_rot[0, 0] * 1.0), (0, xy_scanner_rot[0, 1] * 1.0), c='black')
        ax.plot((0, xy_scanner_rot[-1, 0] * 1.0), (0, xy_scanner_rot[-1, 1] * 1.0), c='black')

        # plot points
        scan = scans[i]
        scan_x, scan_y = u.rphi_to_xy(scan, scan_phi + odo_phi)
        ax.scatter(scan_x, scan_y, s=1, c='blue')

        # inference
        dets_xy, dets_cls, instance_mask = detector(scan)

        # plot detection
        dets_xy_rot = np.matmul(dets_xy, odo_rot.T)
        cls_thresh = 0.3
        for j in range(len(dets_xy)):
            if dets_cls[j] < cls_thresh:
                continue
            c = plt.Circle(dets_xy_rot[j], radius=0.5, color='r', fill=False, linewidth=2)
            ax.add_artist(c)

        # plot track
        cls_thresh = 0.2
        tracks, tracks_cls = detector.get_tracklets()
        for t, tc in zip(tracks, tracks_cls):
            if tc >= cls_thresh and len(t) > 1:
                t_rot = np.matmul(t, odo_rot.T)
                ax.plot(t_rot[:, 0], t_rot[:, 1], color='g', linewidth=2)

        # plt.savefig('/home/dan/tmp/track3_img/frame_%04d.png' % i)

        plt.pause(0.001)
    
        if _break:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--time", default=False, action='store_true')
    parser.add_argument("--dets", default=False, action='store_true')
    parser.add_argument("--tracks", default=False, action='store_true')
    args = parser.parse_args()

    if args.time:
        inference_time()

    if args.dets:
        play_sequence()

    if args.tracks:
        play_sequence_with_tracking()
