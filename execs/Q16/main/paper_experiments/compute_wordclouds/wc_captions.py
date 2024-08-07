import os
import argparse
from main.wordclouds.utils import wc_caption, load_offensive_captions

parser = argparse.ArgumentParser(description='')

parser.add_argument('-cp', '--csv_path', type=str,
                    default='data/ViT-B-16/imagenet1k_train/inapp_images.csv',
                    choices=['data/ViT-B-16/imagenet1k_train/inapp_images.csv',
                             'data/ViT-B-16/openimages_train/inapp_images.csv'])
parser.add_argument('-cgp', '--caption_paths', type=str,
                    default='data/captions/all/imagenet1k_train/magma_generated_captions',
                    choices=['data/captions/all/imagenet1k_train/magma_generated_captions',
                             'data/captions/all/openimages/magma_generated_captions'])

args = parser.parse_args()

dir_path_csv = os.path.dirname(args.csv_path)

_, text_captions = load_offensive_captions(args.caption_paths, args.csv_path)

wc_caption(text_captions, [0.1, 0.4], dir_path_csv)
