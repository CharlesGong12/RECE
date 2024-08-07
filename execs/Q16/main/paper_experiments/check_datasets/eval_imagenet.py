from main.paper_experiments.experiments import torch, parser
from main.paper_experiments.experiments import run_model_imagefolder

torch.set_num_threads(6)
parser.add_argument('--dir_name', type=str, default='/workspace/datasets/imagenet1k/train')
args = parser.parse_args()

dir_name = args.dir_name

save_dir = 'imagenet1k_train'

run_model_imagefolder(args, dir_name, save_dir)
