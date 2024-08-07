import os
import argparse
from main.wordclouds.utils import weighted_wc
import pickle


parser = argparse.ArgumentParser(description='')

parser.add_argument('-cp', '--load_path', type=str,
                    default='data/captions/all/imagenet1k_train/magma_generated_captions/text_inapp_and_noninapp.p')

args = parser.parse_args()

path_offandnonoffText = args.load_path

data_text = pickle.load(open(path_offandnonoffText, "rb"))
caption_text_noninapp = data_text['nonoff']
caption_text_inapp = data_text['off']

caption_text_other = caption_text_inapp

print('Splitted off and nonoff')
collocation_threshold=30
use_bigrams = True

weighted_wc(caption_text_inapp, caption_text_other, os.path.dirname(args.load_path))