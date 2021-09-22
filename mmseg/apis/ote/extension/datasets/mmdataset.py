# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import json
import os
from copy import deepcopy
from typing import List

import numpy as np
from ote_sdk.entities.annotation import Annotation, AnnotationSceneKind
from ote_sdk.entities.dense_label import DenseLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.subset import Subset
from sc_sdk.entities.annotation import AnnotationScene, NullMediaIdentifier
from sc_sdk.entities.dataset_storage import NullDatasetStorage
from sc_sdk.entities.datasets import Dataset, DatasetItem, NullDataset
from sc_sdk.entities.image import Image

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.pipelines import Compose


def get_annotation_mmseg_format(dataset_item: DatasetItem, label_list: List[str]) -> dict:
    """
    Function to convert a OTE annotation to mmdetection format. This is used both in the OTEDataset class defined in
    this file as in the custom pipeline element 'LoadAnnotationFromOTEDataset'

    :param dataset_item: DatasetItem for which to get annotations
    :param label_list: List of label names in the project
    :return dict: annotation information dict in mmdet format
    """
    width, height = dataset_item.width, dataset_item.height
    gt_seg_map = np.full((height, width), 255, dtype=np.uint8)

    for ann in dataset_item.get_annotations():
        box = ann.shape
        if not isinstance(box, Rectangle):
            continue

        labels = ann.get_labels(include_empty=False)
        for label in labels:
            mask = label.get_mask().numpy
            label_id = label.id

            gt_seg_map[mask] = label_id

    ann_info = dict(gt_semantic_seg=gt_seg_map)

    return ann_info


@DATASETS.register_module()
class OTEDataset(CustomDataset):
    """
    Wrapper that allows using a OTE dataset to train mmsegmentation models. This wrapper is not based on the filesystem,
    but instead loads the items here directly from the OTE Dataset object.

    The wrapper overwrites some methods of the CustomDataset class: prepare_train_img, prepare_test_img and prepipeline
    Naming of certain attributes might seem a bit peculiar but this is due to the conventions set in CustomDataset. For
    instance, CustomDatasets expects the dataset items to be stored in the attribute data_infos, which is why it is
    named like that and not dataset_items.

    """

    class _DataInfoProxy:
        """
        This class is intended to be a wrapper to use it in CustomDataset-derived class as `self.data_infos`.
        Instead of using list `data_infos` as in CustomDataset, our implementation of dataset OTEDataset
        uses this proxy class with overriden __len__ and __getitem__; this proxy class
        forwards data access operations to ote_dataset and converts the dataset items to the view
        convenient for mmsegmentation.
        """
        def __init__(self, ote_dataset, classes):
            self.ote_dataset = ote_dataset
            self.CLASSES = classes

        def __len__(self):
            return len(self.ote_dataset)

        def __getitem__(self, index):
            """
            Prepare a dict 'data_info' that is expected by the mmseg pipeline to handle images and annotations
            :return data_info: dictionary that contains the image and image metadata, as well as the labels of
            the objects in the image
            """

            dataset = self.ote_dataset
            item = dataset[index]

            data_info = dict(dataset_item=item,
                             width=item.width,
                             height=item.height,
                             dataset_id=dataset.id,
                             index=index,
                             ann_info=dict(label_list=self.CLASSES))

            return data_info

    def __init__(self, ote_dataset: Dataset, pipeline, classes=None, palette=None, test_mode: bool = False):
        self.ote_dataset = ote_dataset
        self.test_mode = test_mode

        self.CLASSES, self.PALETTE = self.get_classes_and_palette(classes, palette)

        # Instead of using list data_infos as in CustomDataset, this implementation of dataset
        # uses a proxy class with overriden __len__ and __getitem__; this proxy class
        # forwards data access operations to ote_dataset.
        # Note that list `data_infos` cannot be used here, since OTE dataset class does not have interface to
        # get only annotation of a data item, so we would load the whole data item (including image)
        # even if we need only checking aspect ratio of the image; due to it
        # this implementation of dataset does not uses such tricks as skipping images with wrong aspect ratios or
        # small image size, since otherwise reading the whole dataset during initialization will be required.
        self.data_infos = OTEDataset._DataInfoProxy(ote_dataset, self.CLASSES)

        self.pipeline = Compose(pipeline)

    def prepare_train_img(self, idx: int) -> dict:
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys introduced by pipeline.
        """

        item = self.data_infos[idx]

        self.pre_pipeline(item)
        out = self.pipeline(item)

        return out

    def prepare_test_img(self, idx: int) -> dict:
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by pipeline.
        """

        item = self.data_infos[idx]

        self.pre_pipeline(item)
        out = self.pipeline(item)

        return out

    def get_ann_info(self, idx):
        """
        This method is used for evaluation of predictions. The CustomDataset class implements a method
        CustomDataset.evaluate, which uses the class method get_ann_info to retrieve annotations.

        :param idx: index of the dataset item for which to get the annotations
        :return ann_info: dict that contains the coordinates of the bboxes and their corresponding labels
        """

        label_list = self.CLASSES
        if label_list is None:
            label_list = self.dataset.CLASSES

        dataset_item = self.ote_dataset[idx]
        ann_info = get_annotation_mmseg_format(dataset_item, label_list)

        return ann_info

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""

        gt_seg_maps = []
        for item_id in range(len(self)):
            ann_info = self.get_ann_info(item_id)
            gt_seg_maps.append(ann_info['gt_semantic_seg'])

        return gt_seg_maps


def get_classes_from_annotation(annot_path):
    with open(annot_path) as input_stream:
        content = json.load(input_stream)
        labels_map = content['labels_map']

        categories = [(int(v['id']), v['name']) for v in labels_map]

    return categories


def abs_path_if_valid(value):
    if value:
        return os.path.abspath(value)
    else:
        return None


def split_multiclass_annot(annot):
    unique_labels = np.unique(annot)

    out_masks = []
    for label_id in unique_labels:
        out_masks.append((label_id, annot == label_id))

    return out_masks


class MMDatasetAdapter(Dataset):
    def __init__(self,
                 train_img_dir=None,
                 train_ann_dir=None,
                 val_img_dir=None,
                 val_ann_dir=None,
                 test_img_dir=None,
                 test_ann_dir=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.img_dirs = {
            Subset.TRAINING: abs_path_if_valid(train_img_dir),
            Subset.VALIDATION: abs_path_if_valid(val_img_dir),
            Subset.TESTING: abs_path_if_valid(test_img_dir),
        }
        self.ann_dirs = {
            Subset.TRAINING: abs_path_if_valid(train_ann_dir),
            Subset.VALIDATION: abs_path_if_valid(val_ann_dir),
            Subset.TESTING: abs_path_if_valid(test_ann_dir),
        }

        self.labels = self.get_labels_from_annotation(self.ann_dirs)
        assert self.labels is not None

        self.dataset = None
        self.project_labels = None

    @staticmethod
    def get_labels_from_annotation(ann_dirs):
        out_labels = None
        for subset in (Subset.TRAINING, Subset.VALIDATION, Subset.TESTING):
            dir_path = ann_dirs[subset]
            if dir_path is None:
                continue

            labels_map_path = os.path.join(dir_path, 'meta.json')
            labels = get_classes_from_annotation(labels_map_path)

            if out_labels and out_labels != labels:
                raise RuntimeError('Labels are different from annotation file to annotation file.')

            out_labels = labels

        return out_labels

    def set_project_labels(self, project_labels):
        self.project_labels = project_labels

    def label_id_to_project_label(self, label_id):
        matches = [label for label in self.project_labels if label.id == label_id]

        if len(matches) == 0:
            return None
        else:
            assert len(matches) == 1
            return matches[0]

    def init_as_subset(self, subset: Subset):
        test_mode = subset in {Subset.VALIDATION, Subset.TESTING}
        if self.ann_dirs[subset] is None:
            return False

        pipeline = [dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations')]

        self.dataset = CustomDataset(img_dir=self.img_dirs[subset],
                                     ann_dir=self.ann_dirs[subset],
                                     pipeline=pipeline,
                                     classes=self.labels,
                                     test_mode=test_mode)
        self.dataset.test_mode = False

        return True

    def __getitem__(self, indx) -> DatasetItem:
        def _create_gt_label(label_id, mask):
            label = self.label_id_to_project_label(label_id)
            if label is None:
                return None

            mask = Image(name=None, numpy=mask, dataset_storage=NullDatasetStorage())
            out_label = DenseLabel(label=label, mask=mask)

            return out_label

        def _create_gt_labels(enumerated_masks):
            out_labels = []
            for label_id, mask in enumerated_masks:
                label = _create_gt_label(label_id, mask)
                if label is None:
                    continue

                out_labels.append(label)

            return out_labels

        item = self.dataset[indx]

        splitted_annot = split_multiclass_annot(item['gt_semantic_seg'])
        dense_labels = _create_gt_labels(splitted_annot)
        annotation = Annotation(Rectangle(x1=0, y1=0, x2=1, y2=1),
                                labels=dense_labels)

        image = Image(name=None,
                      numpy=item['img'],
                      dataset_storage=NullDatasetStorage())

        annotation_scene = AnnotationScene(kind=AnnotationSceneKind.ANNOTATION,
                                           media_identifier=NullMediaIdentifier(),
                                           annotations=[annotation])
        dataset_item = DatasetItem(image, annotation_scene)

        return dataset_item

    def __len__(self) -> int:
        assert self.dataset is not None
        return len(self.dataset)

    def get_labels(self) -> list:
        return self.labels

    def get_subset(self, subset: Subset) -> Dataset:
        dataset = deepcopy(self)

        if dataset.init_as_subset(subset):
            return dataset
        else:
            return NullDataset()
