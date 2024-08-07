import fire
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import fsspec
from main.paper_experiments.experiments import run_model_imagefolder
from argparse import Namespace


clip_model_name = 'ViT-L/14'
prompt_path = f'data/{clip_model_name.replace("/", "-")}/prompts.p'


def main_imagedataset(input_folder, output_folder):
    """main function"""
    args = Namespace(
        language_model='Clip_'+clip_model_name,
        model_type='sim',
        prompt_path=prompt_path,
        only_inappropriate=True,
        input_type='img',
    )
    run_model_imagefolder(args, input_folder, output_folder)


if __name__ == '__main__':
    fire.Fire(main_imagedataset)
