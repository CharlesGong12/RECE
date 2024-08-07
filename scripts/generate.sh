python /vhome/gongchao/UCE_acceleration/adv-train/execs/generate_images.py \
    --prompts_path dataset/coco_30k.csv \
    --concept coco_erasednudity \
    --save_path /ckpt2/RECE \
    --ckpt xxx.pt \
    --df_length 10000 \
    --df_start 10000

    
python /vhome/gongchao/UCE_acceleration/adv-train/attack_methods/RingABell.py \
    --csv_file dataset/vangogh-ring-a-bell.csv\
    --img_path /path_to_save_images \
    --ckpt xxxx.pt \
    --prompt_file dataset/big_artist_prompts.csv \
    --eta 0.9 \
    --concept vangogh \
    --from_case 20 \
    --to_case 40
