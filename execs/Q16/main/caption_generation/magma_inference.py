# (copied from https://github.com/Aleph-Alpha/magma/blob/master/example_inference.py):

from magma import Magma
from magma.image_input import ImageInput

temperature = 0.4

model = Magma.from_checkpoint(
    config_path = "configs/MAGMA_v1.yml",
    checkpoint_path = "./mp_rank_00_model_states.pt",
    device = 'cuda:0'
)

inputs =[
    ## supports urls and path/to/image
    ImageInput('path/to/image'),
    'A picture of'
]

## returns a tensor of shape: (1, 149, 4096)
embeddings = model.preprocess_inputs(inputs)

## returns a list of length embeddings.shape[0] (batch size)
output = model.generate(
    embeddings = embeddings,
    temperature = temperature
)

print(output[0])
