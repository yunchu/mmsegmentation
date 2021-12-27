# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp

import numpy as np
import matplotlib

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class COCOStuffDataset(CustomDataset):
    """COCO-Stuff dataset.
    """

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
               'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter',
               'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
               'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
               'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
               'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror',
               'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush', 'banner', 'blanket',
               'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other',
               'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt',
               'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 'floor-wood',
               'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel', 'ground-other', 'hill',
               'house', 'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
               'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing',
               'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other',
               'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table',
               'tent', 'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other',
               'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops', 'window-blind',
               'window-other', 'wood')

    PALETTE = None

    def __init__(self, **kwargs):
        super(COCOStuffDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
        assert osp.exists(self.img_dir) and osp.exists(self.ann_dir)

        self.PALETTE = self.get_color_map()

    @staticmethod
    def get_color_map(stuff_start_id=92, stuff_end_id=182, cmap_name='jet'):
        """
        Create a color map for the classes in the COCO Stuff Segmentation Challenge.
        :param stuff_start_id: (optional) index where stuff classes start
        :param stuff_end_id: (optional) index where stuff classes end
        :param cmap_name: (optional) Matlab's name of the color map
        :return: cmap - [c, 3] a color map for c colors where the columns indicate the RGB values
        """

        # Get jet color map from Matlab
        labelCount = stuff_end_id - stuff_start_id + 1
        cmapGen = matplotlib.cm.get_cmap(cmap_name, labelCount)
        cmap = cmapGen(np.arange(labelCount))
        cmap = cmap[:, 0:3]

        # Reduce value/brightness of stuff colors (easier in HSV format)
        cmap = cmap.reshape((-1, 1, 3))
        hsv = matplotlib.colors.rgb_to_hsv(cmap)
        hsv[:, 0, 2] = hsv[:, 0, 2] * 0.7
        cmap = matplotlib.colors.hsv_to_rgb(hsv)
        cmap = cmap.reshape((-1, 3))

        # Permute entries to avoid classes with similar name having similar colors
        st0 = np.random.get_state()
        np.random.seed(42)
        perm = np.random.permutation(labelCount)
        np.random.set_state(st0)
        cmap = cmap[perm, :]

        # Add Things
        thingsPadding = np.zeros((stuff_start_id - 1, 3))
        cmap = np.vstack((thingsPadding, cmap))

        return cmap
