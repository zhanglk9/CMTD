# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import argparse
import json
import os.path as osp

import mmcv
import numpy as np
from PIL import Image


def convert_to_train_id(file):
    # re-assign labels to match the format of Cityscapes
    pil_label = Image.open(file)
    label = np.asarray(pil_label)
    id_to_trainid = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
    sample_class_stats = {}
    for v in id_to_trainid:
        k_mask = label == v
        label_copy[k_mask] = v
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[v] = n
    
    new_file = file.replace('gta2000_rcs1e-2', 'gta')
    sample_class_stats['file'] = new_file
    #Image.fromarray(label_copy, mode='L').save(new_file)
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert GTA annotations to TrainIds')
    parser.add_argument('gta_path', help='gta data path')
    parser.add_argument('--gt-dir', default='labels', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=4, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    # import ipdb;ipdb.set_trace()
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    gta_path = args.gta_path
    out_dir = args.out_dir if args.out_dir else gta_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(gta_path, args.gt_dir)

    poly_files = []
    for poly in mmcv.scandir(
            gt_dir, suffix=tuple('_labelTrainIds.png'),
            recursive=True):
        if not poly.startswith(('a', 'n')):
            if poly.endswith('_labelTrainIds.png'):
                poly_file = osp.join(gt_dir, poly)
                poly_files.append(poly_file)
    poly_files = sorted(poly_files)
    # import ipdb;ipdb.set_trace()

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = mmcv.track_parallel_progress(
                convert_to_train_id, poly_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(convert_to_train_id,
                                                     poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)


if __name__ == '__main__':
    main()
