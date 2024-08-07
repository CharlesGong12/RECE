
python attack_methods/RingABell.py \
    --csv_file dataset/vangogh-ring-a-bell.csv \
    --img_path SD_adv_train/attack_robust/vangogh/ring-a-bell/RECE \
    --ckpt target_ckpt.pt \
    --prompt_file dataset/big_artist_prompts.csv \
    --eta 0.9 \
    --concept vangogh \
    --from_case 20 \
    --to_case 40
