import os
import argparse
from main.wordclouds.utils import wc_classes_infreq, wc_classes_freq, \
    readoffendingimages_csv, create_class_text, \
    imagenet_images, openimages_images, \
    class_annotations_openimages, class_annotation_imagenet

parser = argparse.ArgumentParser(description='')

parser.add_argument('-dp', '--dataset_path', type=str,
                    default='/workspace/datasets/imagenet1k/train')
parser.add_argument('-cp', '--csv_path', type=str,
                    default='data/ViT-B-16/imagenet1k_train/inapp_images.csv')
parser.add_argument('-d', '--dataset', type=str,
                    default='imagenet', choices=['imagenet', 'openimages'])


args = parser.parse_args()

dataset_path = args.dataset_path
dir_path_csv = os.path.dirname(args.csv_path)
image_paths = dataset_path

files = readoffendingimages_csv(args.csv_path, threshold=.5)
if args.dataset == 'openimages':
    images = openimages_images(dataset_path, files)
    get_class_text = class_annotations_openimages
elif args.dataset == 'imagenet':
    images = imagenet_images(dataset_path, files)
    get_class_text = class_annotation_imagenet
else:
    images = None  # list of image ids
    get_class_text = None  # given image path return list of classes

    raise ValueError('add your dataset here')


class_images, class_text = create_class_text(images, get_class_text, dir_path_csv)

# print(class_text)
regex_classes = '|'.join(list(class_images.keys()))

## clases wordclouds
wordcloud_classes_freq = wc_classes_freq(class_text, regex_classes, dir_path_csv)
wc_classes_infreq(wordcloud_classes_freq, class_text, dir_path_csv)
