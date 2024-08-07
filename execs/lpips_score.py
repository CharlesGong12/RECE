import argparse
import os
import lpips
from tqdm import tqdm
import cv2
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d0','--dir0', type=str, default='/share/ckpt/gongchao/SD_adv_train/coco30k_ori_sd_1_4/coco', help='direction of the target images')
    parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1', help='direction of the newly generated images')
    parser.add_argument('--filter', type=str, default=None, help='dir filter for the images')
    parser.add_argument('-v','--version', type=str, default='0.1')

    opt = parser.parse_args()

    ## Initializing the model
    loss_fn = lpips.LPIPS(net='alex',version=opt.version)
    loss_fn.cuda()

    # crawl directories
    if 'sd' in opt.dir0 or 'before' in opt.dir0:
        out_file_path = os.path.join(opt.dir1, "lpips_against_sd14.txt")
    if 'real' in opt.dir0:
        out_file_path = os.path.join(opt.dir1, "lpips_against_real.txt")
    f = open(out_file_path,'w')
    if os.path.exists(os.path.join(opt.dir1, "imgs")):
        opt.dir1 = os.path.join(opt.dir1, "imgs")
    if os.path.exists(os.path.join(opt.dir0, "imgs")):
        opt.dir0 = os.path.join(opt.dir0, "imgs")
    files = os.listdir(opt.dir0)
    files = [file for file in files if '.png' in file]
    if opt.filter is not None:
        if os.path.exists(os.path.join(opt.filter, "imgs")):
            opt.filter = os.path.join(opt.filter, "imgs")
        filter = os.listdir(opt.filter)
        files = [file for file in files if file in filter]
        
    dists = []

    for file in tqdm(files):
        if os.path.exists(os.path.join(opt.dir1,file)):
            # Load images
            img0 = lpips.load_image(os.path.join(opt.dir0,file))
            img0 = cv2.resize(img0, (64, 64))
            img1 = lpips.load_image(os.path.join(opt.dir1,file))
            img1 = cv2.resize(img1, (64, 64))
            img0 = lpips.im2tensor(img0) # RGB image from [-1,1]
            img1 = lpips.im2tensor(img1)

            img0 = img0.cuda()
            img1 = img1.cuda()

            # Compute distance
            dist01 = loss_fn.forward(img0,img1)
            # print('%s: %.3f'%(file,dist01))
            f.writelines('%s: %.6f\n'%(file,dist01))
            dists.append(dist01.item())
        else:
            continue
            # print(f"File {file} not found in {opt.dir1}")
    f.writelines(f'avg: {sum(dists) / len(dists)}')
    f.close()

    print(f"Average LPIPS of {opt.dir1} against {opt.dir0}: {sum(dists) / len(dists)}")

