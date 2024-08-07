import pandas as pd
import random
file_path = '/vhome/gongchao/UCE_acceleration/adv-train/dataset/coco_30k.csv'
df = pd.read_csv(file_path)
# random select 3000 images, seed 42
random.seed(42)
df = df.sample(n=3000, random_state=42)
# save to csv
df.to_csv('/vhome/gongchao/UCE_acceleration/adv-train/dataset/coco_30k_val.csv', index=False)