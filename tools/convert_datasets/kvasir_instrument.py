import argparse
from os.path import join
from shutil import copyfile

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Kvasir-Instrument dataset to mmsegmentation format')
    parser.add_argument('images_path',
                        help='the path to images folder')
    parser.add_argument('masks_path',
                        help='the path to masks folder')
    parser.add_argument('train_path',
                        help='the path to train.txt')
    parser.add_argument('test_path',
                        help='the path to test.txt')
    parser.add_argument('-o', '--out_dir',
                        help='output path')
    args = parser.parse_args()

    return args


def read_records(data_path):
    with open(data_path) as input_stream:
        instances = []
        for line in input_stream:
            instance = line.strip()
            if len(instance) > 0:
                instances.append(instance)

    return instances


def process_instances(mode, instances, images_path, masks_path, out_dir):
    assert mode in ['training', 'validation']

    for instance in instances:
        image_name = f'{instance}.jpg'
        mask_name = f'{instance}.png'

        in_image_path = join(images_path, image_name)
        in_mask_path = join(masks_path, mask_name)

        out_image_path = join(out_dir, 'images', mode, image_name)
        out_mask_path = join(out_dir, 'annotations', mode, mask_name)

        copyfile(in_image_path, out_image_path)

        mask = mmcv.imread(in_mask_path)
        mmcv.imwrite(mask[:, :, 0] // 128, out_mask_path)


def main():
    args = parse_args()

    if args.out_dir is None:
        out_dir = join('data', 'kvasir_instrument')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(join(out_dir, 'images'))
    mmcv.mkdir_or_exist(join(out_dir, 'images', 'training'))
    mmcv.mkdir_or_exist(join(out_dir, 'images', 'validation'))
    mmcv.mkdir_or_exist(join(out_dir, 'annotations'))
    mmcv.mkdir_or_exist(join(out_dir, 'annotations', 'training'))
    mmcv.mkdir_or_exist(join(out_dir, 'annotations', 'validation'))

    print('Generating training dataset...')
    train_instances = read_records(args.train_path)
    process_instances('training', train_instances, args.images_path, args.masks_path, out_dir)

    print('Generating validation dataset...')
    val_instances = read_records(args.test_path)
    process_instances('validation', val_instances, args.images_path, args.masks_path, out_dir)

    print('Done!')


if __name__ == '__main__':
    main()
