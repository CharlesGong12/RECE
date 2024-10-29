# Reliable and Efficient Concept Erasure of Text-to-Image Diffusion Models

:star2: :star2: _ECCV 2024_ | [Arxiv](https://arxiv.org/abs/2407.12383) | :hugs:[Models](https://huggingface.co/ChaoGong/RECE) :star2: :star2:

**Authors**

[Chao Gong](https://scholar.google.com/citations?user=XYjTyOgAAAAJ&hl=zh-CN)\*, [Kai Chen](https://github.com/kay-ck)\*, Zhipeng Wei, Jingjing Chen, Yu-Gang Jiang

_Fudan University_

## Run

The edited models of RECE can be found :hugs:[here](https://huggingface.co/ChaoGong/RECE). 

* Run `pip install -r requirements.txt` to install the required packages.

* You can check `scripts/` for running scripts. For example, run the following command to erase "nudity":
  
    ```
    python train.py \
        --concepts nudity \
        --concept_type nudity \
        --emb_computing close_regzero \
        --regular_scale 0.1 \
        --epochs 3 \
        --target_ckpt unified-concept-editing/erased-nudity-towards_uncond-preserve_false-sd_1_4-method_replace-1-1.0.pt \
        --preserve_scale 0.1 \
        --lamb 0.1 
    ```

  Then, generate images of I2P:
    
    ```
    python execs/generate_images.py \
        --prompts_path dataset/i2p.csv \
        --concept nudity \
        --save_path /ckpt2/RECE \
        --ckpt results_above.pt \
    ```
  Finally, evaluate the erasure performance:

    ```
    python compute_nudity_rate.py \
        --root save_path_above
    ```
    


## Notes
* Configuration. We have released a new Arxiv version to state the experiment settings. For all concepts, the coefficients of Eq.3 are: $\lambda_1=0.1$ and $\lambda_2=0.1$. The regularization coefficients $\lambda$ and epochs are set as follows:

  1. Nudity and unsafe concepts(I2P concepts), $\lambda=1e-1$, with nudity for 3 epochs and unsafe concepts for 2 epochs.
  2. Artistic styles, $\lambda=1e-3$, 1 epoch.
  3. Difficult objects for UCE(e.g., church and garbage truck), $\lambda=1e-3$, 1 epoch.
  4. Easy objects for UCE(e.g., English Springer, golf ball and parachute), $\lambda=1e-1$, 1 epoch.
  5. For other objects where erasing accuracies reach 0 using UCE, RECE's further erasure is not applied.

* Red-teaming tools. Due to the open-source timeline, we used our reproduced Ring-A-Bell attack method for all baselines, available in `attack_methods/`. And we used the P4D attack method reproduced by [UnlearnDiff](https://github.com/OPTML-Group/Diffusion-MU-Attack).



## Citation
If you find our work helpful, please leave us a star and cite our paper.
  
  ```
  @article{gong2024reliable,
    title={Reliable and Efficient Concept Erasure of Text-to-Image Diffusion Models},
    author={Gong, Chao and Chen, Kai and Wei, Zhipeng and Chen, Jingjing and Jiang, Yu-Gang},
    journal={arXiv preprint arXiv:2407.12383},
    year={2024}
  }
  ```

## Acknowledgement
Some code is borrowed from [UCE](https://github.com/rohitgandikota/unified-concept-editing).