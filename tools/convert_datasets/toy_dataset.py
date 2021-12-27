# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import os
import os.path as osp

import cv2
import mmcv
import numpy as np


COLORS = dict(
    rectangle=(123, 12, 78),
    circle=(21, 230, 139),
    triangle=(77, 23, 219),
)


def validate_path(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def generate_image_pair(size, colors, max_figures=3, relative_size=0.2):
    assert max_figures > 0
    num_figures = np.random.randint(1, max_figures + 1)

    min_size, max_size = int(0.5 * size * relative_size), int(1.5 * size * relative_size)
    assert 0 < min_size < max_size < size

    image = np.full([size, size, 3], 255, dtype=np.uint8)
    annot = np.zeros([size, size], dtype=np.uint8)

    for _ in range(num_figures):
        # draw rectangle
        height = np.random.randint(min_size, max_size)
        width = np.random.randint(min_size, max_size)

        start_point = (np.random.randint(size - width), np.random.randint(size - height))
        end_point = (start_point[0] + width, start_point[1] + height)

        cv2.rectangle(image, start_point, end_point, colors['rectangle'], -1)
        cv2.rectangle(annot, start_point, end_point, (1,), -1)

        # draw circle
        radius = int(0.5 * np.random.randint(min_size, max_size))
        center = (np.random.randint(radius, size - radius), np.random.randint(radius, size - radius))

        cv2.circle(image, center, radius, colors['circle'], -1)
        cv2.circle(annot, center, radius, (1,), -1)

        # draw triangle
        height = np.random.randint(min_size, max_size)
        width = np.random.randint(min_size, max_size)
        top = (np.random.randint(size - width), np.random.randint(size - height))

        triangle_pts = [(np.random.randint(width) + top[0], np.random.randint(height) + top[1]),
                        (np.random.randint(width) + top[0], np.random.randint(height) + top[1]),
                        (np.random.randint(width) + top[0], np.random.randint(height) + top[1])]
        triangle_pts = np.array(triangle_pts)

        cv2.drawContours(image, [triangle_pts], 0, colors['triangle'], -1)
        cv2.drawContours(annot, [triangle_pts], 0, (1,), -1)

    return image, annot


def generate_pairs(size, num_pairs, images_dir, annot_dir):
    for pair_id in range(num_pairs):
        image, annot = generate_image_pair(size, COLORS)

        file_template = '{:>03}_instance'.format(pair_id)
        mmcv.imwrite(image, osp.join(images_dir, file_template + '.jpg'), auto_mkdir=False)
        mmcv.imwrite(annot, osp.join(annot_dir, file_template + '.png'), auto_mkdir=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Generates toy dataset')
    parser.add_argument('output_dir', help='output dir')
    parser.add_argument('-r', '--resolution', type=int, default=544, help='Image resolution')
    parser.add_argument('-nte', '--num_test', type=int, default=3, help='Number of test images')
    parser.add_argument('-ntr', '--num_train', type=int, default=6, help='Number of train images')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    out_train_images = validate_path(osp.join(args.output_dir, 'images', 'training'))
    out_val_images = validate_path(osp.join(args.output_dir, 'images', 'validation'))
    out_train_annot = validate_path(osp.join(args.output_dir, 'annotations', 'training'))
    out_val_annot = validate_path(osp.join(args.output_dir, 'annotations', 'validation'))

    generate_pairs(args.resolution, args.num_test, out_val_images, out_val_annot)
    generate_pairs(args.resolution, args.num_train, out_train_images, out_train_annot)


if __name__ == '__main__':
    main()
