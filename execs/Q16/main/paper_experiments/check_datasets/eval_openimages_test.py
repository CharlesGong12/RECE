from main.paper_experiments.experiments import torch, parser
from main.paper_experiments.experiments import run_model_imagefolder

torch.set_num_threads(6)
parser.add_argument('--dir_name', type=str, default='/workspace/datasets/openimagesv6/test')
args = parser.parse_args()

dir_name = args.dir_name

save_dir = 'openimages_test'

run_model_imagefolder(args, dir_name, save_dir, types=['/*.jpg'])
