import argparse
import glob
import os
import pickle
import yaml

import dr_spaam.utils.eval_utils as eu
from dr_spaam.utils.dataset import create_test_dataloader
from dr_spaam.utils.train_utils import load_checkpoint


def eval(model, cfg, epoch, split, it=0, writing=True, plotting=True,
         save_pkl=True, tb_log=None, scan_stride=1, pt_stride=1):
    root_result_dir = os.path.join('./output', cfg['name'])

    test_loader = create_test_dataloader(data_path="./data/DROWv2-data",
                                         num_scans=cfg['num_scans'],
                                         network_type=cfg['network'],
                                         cutout_kwargs=cfg['cutout_kwargs'],
                                         polar_grid_kwargs=cfg['polar_grid_kwargs'],
                                         pedestrian_only=cfg['pedestrian_only'],
                                         split=split,
                                         scan_stride=scan_stride,
                                         pt_stride=pt_stride)

    eu.eval_epoch_with_output(model, test_loader, epoch=epoch, it=it,
                              vote_kwargs=cfg['vote_kwargs'],
                              root_result_dir=root_result_dir, split=split,
                              tag='eval_%s' % cfg['name'], writing=writing,
                              plotting=plotting, save_pkl=save_pkl, tb_log=tb_log,
                              full_eval=True)


def eval_dir(cfgs_dir, split, epoch):
    cfgs_list = glob.glob(os.path.join(cfgs_dir, '*.yaml'))

    for cfg_file in cfgs_list:
        with open(cfg_file, 'r') as f:
            cfg = yaml.safe_load(f)
            cfg['name'] = os.path.basename(cfg_file).split(".")[0] + cfg['tag']

        ckpt = os.path.join('./output/', cfg['name'], 'ckpts', 'ckpt_e%s.pth' % epoch)
        if not os.path.isfile(ckpt):
            print("Could not load ckpt %s from config %s" % (ckpt, cfg['name']))
            continue

        print("Eval ckpt %s from config %s" % (ckpt, cfg["name"]))
        model = eu.cfg_to_model(cfg)
        model.cuda()

        _, epoch = load_checkpoint(model=model, filename=ckpt)
        eval(model, cfg, epoch, split, writing=True, plotting=False, save_pkl=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--cfg", type=str, required=False, default=None)
    parser.add_argument("--ckpt", type=str, required=False, default=None)
    parser.add_argument("--pkl", type=str, required=False, default=None)
    parser.add_argument("--val", default=False, action='store_true')
    parser.add_argument("--dir", type=str, required=False, default=None)
    parser.add_argument("--epoch", type=int, required=False, default=40)
    parser.add_argument("--pt_stride", type=int, required=False, default=1)
    parser.add_argument("--scan_stride", type=int, required=False, default=1)
    parser.add_argument("--tag", type=str, required=False, default="")
    args = parser.parse_args()

    # load existing results, only plotting
    if args.pkl is not None:
        with open(args.pkl, 'rb') as f:
            _, eval_rpt = pickle.load(f)

        # plot
        for k, v in eval_rpt.items():
            plot_title = args.pkl.split['.'][0] + ('_t%s' % k)
            eu.plot_eval_result(v, plot_title=plot_title,
                                output_file=plot_title + '.png')

    # eval dir
    elif args.dir is not None:
        split = 'val' if args.val else 'test'
        eval_dir(args.dir, split, args.epoch)

    # eval single config
    elif args.cfg is not None:
        with open(args.cfg, 'r') as f:
            cfg = yaml.safe_load(f)
            cfg['name'] = os.path.basename(args.cfg).split(".")[0] + cfg['tag']

        # model
        model = eu.cfg_to_model(cfg)
        model.cuda()

        if args.ckpt is not None:
            # ckpt = os.path.join('./output/', cfg['name'], 'ckpts', args.ckpt)
            ckpt = args.ckpt
        else:
            ckpt = os.path.join('./output/', cfg['name'], 'ckpts', 'ckpt_e%s.pth' % args.epoch)

        _, epoch = load_checkpoint(model=model, filename=ckpt)

        split = 'val' if args.val else 'test'

        if len(args.tag) > 0:
            cfg['name'] = cfg['name'] + "_" + args.tag

        eval(model, cfg, epoch, split, scan_stride=args.scan_stride, pt_stride=args.pt_stride)
