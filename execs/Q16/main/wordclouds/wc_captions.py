import os
import argparse
from main.wordclouds.utils import load_captions_from_csv, wc_caption

parser = argparse.ArgumentParser(description='')

parser.add_argument('--load_path', type=str,
                    default='data/captions/all/openimages_train/ViT-B-16/text_inapp.csv')

args = parser.parse_args()

dir_path_csv = os.path.dirname(args.csv_path)

text_captions = load_captions_from_csv(args.inapp_captions)

wc_caption(text_captions, None, dir_path_csv)
