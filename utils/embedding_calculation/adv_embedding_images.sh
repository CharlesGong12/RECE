#!/bin/bash
python /vhome/gongchao/UCE_acceleration/adv-train/execs/adv_embedding_images.py --num_adv_prompts 0 --concepts nudity --concept_type nudity
python /vhome/gongchao/UCE_acceleration/adv-train/execs/adv_embedding_images.py --num_adv_prompts 0 --concepts "female and male nudity" --concept_type nudity --seed 111