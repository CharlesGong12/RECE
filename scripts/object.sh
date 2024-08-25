# Church, English Springer, Garbage Truck, Golf Ball, Parachute
python train_object.py --concepts 'Church' --concept_type object --target_ckpt "UCE/erased-church-towards_uncond-preserve_false-sd_1_4-method_replace-1-1.0.pt"
python train_object.py --concepts 'English Springer' --concept_type object --target_ckpt "UCE/erased-english springer-towards_uncond-preserve_false-sd_1_4-method_replace-1-1.0.pt" --regular_scale 1e-1
python train_object.py --concepts 'Garbage Truck' --concept_type object --target_ckpt "UCE/erased-garbage truck-towards_uncond-preserve_false-sd_1_4-method_replace-1-1.0.pt"
python train_object.py --concepts 'Golf Ball' --concept_type object --target_ckpt "UCE/erased-golf ball-towards_uncond-preserve_false-sd_1_4-method_replace-1-1.0.pt" --regular_scale 1e-1
python train_object.py --concepts 'Parachute' --concept_type object --target_ckpt "UCE/erased-parachute-towards_uncond-preserve_false-sd_1_4-method_replace-1-1.0.pt" --regular_scale 1e-1

