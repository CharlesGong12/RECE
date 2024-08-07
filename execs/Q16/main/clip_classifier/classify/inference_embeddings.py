from embedding_reader import EmbeddingReader
import fire
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import fsspec
import math
from main.clip_classifier.classify.utils import load_prompts


def load_safety_prompts(prompt_path):
    device = 'cpu'
    trained_prompts = load_prompts(prompt_path, device)
    return trained_prompts


clip_model_name = 'ViT-L/14'
prompt_path = f'data/{clip_model_name.replace("/", "-")}/prompts.p'
safety_prompts = load_safety_prompts(prompt_path)


def get_unsafe_results(embeddings):
    safety_predictions = np.einsum("ij,kj->ik", embeddings, safety_prompts)
    safety_results = np.argmax(safety_predictions, axis=1)
    return np.where(safety_results == 1)[0]


def main_embedding(input_folder, output_folder, batch_size=10**6, end=None):
    """main function"""
    reader = EmbeddingReader(input_folder)
    fs, relative_output_path = fsspec.core.url_to_fs(output_folder)
    fs.mkdirs(relative_output_path, exist_ok=True)

    total = reader.count
    batch_count = math.ceil(total // batch_size)
    padding = int(math.log10(batch_count)) + 1

    for i, (embeddings, ids) in enumerate(reader(batch_size=batch_size, start=0, end=end)):
        predictions = get_unsafe_results(embeddings)
        batch = np.hstack(ids.iloc[predictions].to_numpy())
        #print(batch.shape)
        padded_id = str(i).zfill(padding)
        output_file_path = os.path.join(relative_output_path, padded_id + ".npy")
        with fs.open(output_file_path, "wb") as f:
            np.save(f, batch)


if __name__ == '__main__':
    fire.Fire(main_embedding)
