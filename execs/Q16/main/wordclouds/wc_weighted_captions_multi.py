import os
import argparse
from main.wordclouds.utils import weighted_wc, load_captions_from_csv
import glob

parser = argparse.ArgumentParser(description='')

parser.add_argument('--load_path', type=str,
                    default='data/captions/all/imagenet1k_train/ViT-B-16')

args = parser.parse_args()

files_noninapp = glob.glob(os.path.join(args.load_path, 'text_noninapp*.csv'))
files_inapp = glob.glob(os.path.join(args.load_path, 'text_inapp*.csv'))

caption_text_noninapp = ''
for file_name in files_noninapp:
    caption_text_noninapp += load_captions_from_csv(file_name)

caption_text_inapp = ''
for file_name in files_inapp:
    caption_text_inapp += load_captions_from_csv(file_name)

caption_text_other = caption_text_noninapp

print('Splitted off and nonoff')
collocation_threshold=30
use_bigrams = True

weighted_wc(caption_text_inapp, caption_text_other, args.load_path)