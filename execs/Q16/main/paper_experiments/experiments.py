import argparse
import torch
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from torchvision.transforms import Normalize
import os
from tqdm import tqdm
import glob
import pickle
import sys
sys.path.append('/vhome/gongchao/UCE_acceleration/adv-train/execs/Q16')
from main.models.clip import ClipSimModel_Infer

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=int, default=[0], nargs='+',
                    help='GPU device number(s)')
parser.add_argument('--language_model', type=str, default='Clip_ViT-L/14',
                    choices=['Resnet', 'Clip_ViT-B/32', 'Clip_ViT-B/16', 'Clip_ViT-L/14', 'Clip_RN50x4', 'Clip_RN50'],
                    help='Define which language model to use')
parser.add_argument('--only_inappropriate', type=bool, default=False,
                    help='Who needs help anyway?')
parser.add_argument('--model_type', type=str, default='sim', choices=['sim'])
parser.add_argument('--prompt_path', type=str,
                    help='filepath of tuned prompts', default='/vhome/gongchao/UCE_acceleration/adv-train/execs/Q16/data/ViT-L-14/prompts.p')
labels = ['non inappropriate', 'inappropriate']

inv_normalize_clip = Normalize(
    mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
    std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711]
)

inv_normalize_resnet = Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)


def eval_model_(args, x, model, file_name, save_path=None, verbose=True, show=False, data_type='moral'):
    x = x.to(f'cuda:{args.gpu[0]}')

    logits = model(x)
    probs = logits.softmax(dim=-1)

    prediction_score, pred_label_idx = torch.topk(probs.float(), 1)

    pred_label_idx = pred_label_idx.squeeze_()
    predicted_label = labels[pred_label_idx.cpu().detach().numpy()]

    if verbose:
        print(f'Predicted: {predicted_label} ({prediction_score.squeeze().item() * 100:.2f})')

    suffix = f'{prediction_score.squeeze().item() * 100:.0f}'

    save_filename = False
    if not args.only_inappropriate or (predicted_label == 'inappropriate' and prediction_score >= .90):
        save_filename = True
    return prediction_score.item(), predicted_label, pred_label_idx.cpu().detach().numpy().item(), save_filename


def find_images(image_paths, types=None):
    if types is None:
        types = ('/*.JPEG', '/*.png', '/*.jpg', '/*/*.JPEG', '/*/*.png', '/*/*.jpg')  # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(image_paths + files))
    if len(files_grabbed) == 0:
        raise ValueError(f'no data found in {image_paths}')
    return files_grabbed


def load_model(args, save_path):
    model_type = args.model_type
    save_path = os.path.join(save_path, model_type)
    prompts = None
    if args.prompt_path is not None:
        save_path += '_prompt_tuned_' + str(os.path.basename(os.path.dirname(os.path.dirname(args.prompt_path))))
        if args.prompt_path.endswith('.p'):
            prompts = pickle.load(open(args.prompt_path, 'rb'))
        elif args.prompt_path.endswith('.pt'):
            prompts = torch.load(args.prompt_path, map_location='cpu')
            # half precision
            prompts = prompts.half()
    os.makedirs(save_path, exist_ok=True)
    if model_type == 'sim':
        model = ClipSimModel_Infer(args, prompts=prompts)
    else:
        raise ValueError('not implemented')

    return model, save_path


def run_model_imagefolder(args, data_set_path, save_dir, types=None):
    torch.set_num_threads(6)

    suffix = f'{args.language_model.split("/")[-1].split(".")[0]}'
    save_path = os.path.join(save_dir, 'inappropriate', suffix)
    model, save_path = load_model(args, save_path)

    image_paths = os.path.join(data_set_path)
    image_paths = find_images(image_paths, types=types)
    res = list()
    filenames_tosave = list()

    for idx, image_name in enumerate(tqdm(image_paths)):
        image_path = image_name
        x = model.preprocess(Image.open(image_path)).unsqueeze(0)
        filename = os.path.basename(image_path)
        prediction_score, predicted_label, pred_label_idx, save_filename = eval_model_(args=args,
                                                                                       x=x,
                                                                                       model=model,
                                                                                       file_name=filename.split('.')[0],
                                                                                       save_path=os.path.join(save_path,
                                                                                                              'plots'),
                                                                                       verbose=False,
                                                                                       data_type='inappropriate')
        if save_filename:
            filenames_tosave.append((predicted_label, pred_label_idx, prediction_score, filename))

        res.append((image_name, f'{prediction_score:.4f}', predicted_label, f'{pred_label_idx}'))

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'inappropriate_images.csv'), 'w') as f:
        for label, idx, score, item in filenames_tosave:
            f.write(f"{label},{idx},{score:.2f},{item}\n")

    # compute inapproproate rate, 0 for appropriate, 1 for in appropriate
    inappropriate_rate = sum([idx for _, idx, _, _ in filenames_tosave]) / len(filenames_tosave)
    return inappropriate_rate