import sys
# sys.path.append('core')

import argparse
import os
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import time
import shutil

DEVICE = 'cuda'

# CV to PIL conversion
def load_image_from_file(imfile):
    img_cv = cv2.imread(imfile)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img =  np.array(Image.fromarray(img_cv)).astype(np.uint8)
    # img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

# CV to PIL conversion
def load_image_from_cv(img_cv):
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img =  np.array(Image.fromarray(img_cv)).astype(np.uint8)
    # img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

# def viz(img, flo, name1, name2):
def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # cv2.imwrite('demo-out/result.png', flo)
    return flo

def load_model(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    return model

def demo(args, model, image1, image2):
    with torch.no_grad():
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        start = time.perf_counter()
        flow_low, flow_up = model(image1, image2, iters=15, test_mode=True)           
        stop = time.perf_counter()
        print("prediction:", (stop - start) * 1000, "ms")
        # viz(image1, flow_up, imfile1[12:-4], imfile2[12:-4])
        return viz(image1, flow_up)


if __name__ == '__main__':
    args = argparse.Namespace(model='models/raft-things.pth', path='demo-frames', small=False, mixed_precision=True, alternate_corr=False)#small=False, mixed_precision=False, alternate_corr=False)

    image1 = load_image_from_file('demo-frames/out_0067.jpg')
    image2 = load_image_from_file('demo-frames/out_0068.jpg')
    model = load_model(args)
    for i in range(10):
        print(demo(args, model, image1, image2))
