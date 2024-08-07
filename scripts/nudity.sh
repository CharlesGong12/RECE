python train.py \
    --concepts nudity \
    --concept_type nudity \
    --emb_computing close_regzero \
    --regular_scale 0.1 \
    --epochs 5 \
    --target_ckpt ckpt/unified-concept-editing/erased-nudity-towards_uncond-preserve_false-sd_1_4-method_replace-1-1.0.pt \
    --preserve_scale 0.1 \
    --lamb 0.1 
# python train.py --concepts nudity --old_target_concept ' ' --concept_type nudity --emb_computing close_standardreg --regular_scale 1e-3 --epochs 10 --target_ckpt /share/ckpt/gongchao/unified-concept-editing/erased-nudity-towards_uncond-preserve_false-sd_1_4-method_replace-1-1.0.pt --preserve_scale 0.1 --lamb 0.1
