
cd /vhome/gongchao/UCE_acceleration/Diffusion-MU-Attack

for i in {0..142}
do
    echo "Epoch: $i"
    python src/execs/attack.py --config-file configs/style/hard_prompt_sld-med_vangogh_P4D.json --attacker.attack_idx $i --logger.name attack_idx_$i
done
