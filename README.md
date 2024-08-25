# Reliable and Efficient Concept Erasure of Text-to-Image Diffusion Models

:star2: :star2: _ECCV 2024_ | [Arxiv](https://arxiv.org/abs/2407.12383) | :hugs:[HuggingFace](https://huggingface.co/ChaoGong/RECE) :star2: :star2:

**Authors**

[Chao Gong](https://scholar.google.com/citations?user=XYjTyOgAAAAJ&hl=zh-CN)\*, [Kai Chen](https://github.com/kay-ck)\*, Zhipeng Wei, Jingjing Chen, Yu-Gang Jiang

_Fudan University_

## Code

The code that has been preliminarily organized has been released. 

1. Run `pip install -r requirements.txt` to install the required packages.

2. You can check `scripts/` for running scripts.

The edited models of RECE can be found :hugs:[here](https://huggingface.co/ChaoGong/RECE). 

## Erasure Details

For all concepts, the coefficients of Eq.3 are: $\lambda_1=0.1$ and $\lambda_2=0.1$.

The regularization coefficients $\lambda$ are:

1. Nudity and unsafe concepts(I2P concepts), $\lambda=1e-1$.
2. Artistic styles, $\lambda=1e-3$.
3. Difficult objects (e.g., church and garbage truck), $\lambda=1e-3$.
4. Easy objects (e.g., English Springer, golf ball and parachute), $\lambda=1e-1$.
5. For other objects where erasing accuracies reach 0 using UCE, RECE's further erasure is not applied.

**We will update the Arxiv version to correct/align the experiment settings.**

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