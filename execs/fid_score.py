import argparse
from cleanfid import fid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f1', type=str, default=None)
    parser.add_argument('--f2', type=str, default=None)
    args = parser.parse_args()
    file_path1 = args.f1
    file_path2 = args.f2
    # custom_model = 
    score = fid.compute_fid(file_path1, file_path2)
    print(f"FID score between {file_path1} and {file_path2} is {score}")
