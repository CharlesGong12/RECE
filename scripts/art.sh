
python train_artists.py \
    --concepts 'Kelly McKernan' \
    --guided_concepts art \
    --concept_type 'art'\
    --target_ckpt 'UCE/erased-kelly mckernan-towards_art-preserve_true-sd_1_4-method_replace-1-0.1.pt' \
    --emb_computing close_regzero \
    --regular_scale 1e-3 \
    --epochs 1 \
    --test_csv_path dataset/validation_niche_artists.csv
