
python train_artists.py \
    --concepts 'Kelly McKernan' \
    --guided_concepts art \
    --concept_type 'art'\
    --target_ckpt 'ckpt2/UCE/erased-kelly mckernan-towards_art-preserve_true-sd_1_4-method_replace-1-0.1.pt' \
    --emb_computing close_regzero \
    --regular_scale 1e-3 \
    --epochs 5 \
    --test_csv_path dataset/short_niche_art_prompts.csv
