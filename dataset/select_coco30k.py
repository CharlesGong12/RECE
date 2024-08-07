import os
import shutil
import csv
from tqdm import tqdm
from itertools import islice

prompt_csv_file = "/vhome/gongchao/UCE_acceleration/adv-train/dataset/coco_30k.csv"

coco_images_folder = "/share/common/ImageDatasets/coco2017/"

filtered_images_folder = "/share/ckpt/gongchao/SD_adv_train/coco30k_real"

if not os.path.exists(filtered_images_folder):
    os.makedirs(filtered_images_folder)

with open(prompt_csv_file, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    # csv_reader = islice(csv_reader, 13171, None)
    for row in tqdm(csv_reader):
        case_number = row[0]
        coco_id = row[-1]
        for root, dirs, files in os.walk(coco_images_folder):
            for file in files:
                if coco_id in file:
                    src_path = os.path.join(root, file)
                    dst_filename = f"{case_number}_0.png"
                    dst_path = os.path.join(filtered_images_folder, dst_filename)
                    shutil.copyfile(src_path, dst_path)
                    break

print("Done!")
