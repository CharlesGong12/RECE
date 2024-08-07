import numpy as np
import os
import argparse
from utils import load_captions, readoffendingimages_csv
from tqdm import tqdm

parser = argparse.ArgumentParser(description='')

parser.add_argument('-cgp', '--caption_paths', type=str,
                    default='./data/captions/all/imagenet1k_train/magma_generated_captions')
parser.add_argument('-cp', '--csv_path', type=str,
                    default='./data/ViT-B-16/imagenet1k_train/inapp_images.csv')

args = parser.parse_args()

## load all caption
img_text = load_captions(args.caption_paths)
img_path_offending = readoffendingimages_csv(args.csv_path, threshold=.5)
temps = [0.1, 0.4]
cnt_captions = 0
tmp_paths = []
img_caption_dict = {}


def replace_characters(text):
    new_text = text.replace('<PERSON>', 'PERSON').lower()#.replace('.', '')
    new_text = new_text.replace(',', '').replace('…', '')
    new_text = new_text.replace('"', '').replace(')', '').replace('(', '')
    #new_text
    return new_text


for temp in temps:
    for img_path in list(img_text[temp].keys()):
        img_id = img_path.split('.')[0]
        if img_id not in img_caption_dict:
            img_caption_dict[img_id] = []
        for img_captions in img_text[temp][img_path]:
            for caption in img_captions:
                tmp_paths += [img_path]
                cnt_captions += 1
                img_caption_dict[img_id] += [replace_characters(caption)]# + '\n'
        # print(t[0])
print('#Captions:', cnt_captions)
print('#Images:', len(np.unique(tmp_paths)))

## split captions into offending and non-offending images
caption_text_inapp = []
caption_text_noninapp = []

cnt = 0
for img_id in tqdm(list(img_caption_dict.keys())):
    for e in img_caption_dict[img_id]:
        if img_id in img_path_offending:
            caption_text_inapp += [[img_id, e]]
        else:
            caption_text_noninapp += [[img_id, e]]
    if cnt == 3:
        break
    cnt += 1
res_noninapp = np.asarray(caption_text_noninapp, dtype=str)
res_inapp = np.asarray(caption_text_inapp, dtype=str)

path_inappText = os.path.join(os.path.dirname(args.csv_path), "text_inapp.csv")
path_noninappText = os.path.join(os.path.dirname(args.csv_path), "text_noninapp.csv")
np.savetxt(path_noninappText, res_noninapp, delimiter="\t", fmt='%s')
np.savetxt(path_inappText, res_inapp, delimiter="\t", fmt='%s')
