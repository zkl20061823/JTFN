import cv2
import os
import argparse
import torch
import torch.nn.functional as F

def generate_BD(mask_folder, save_folder):
    imgs = os.listdir(mask_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for item in imgs:
        img_name = os.path.basename(item)
        if not '.png' in img_name:
            continue
        img_path = os.path.join(mask_folder, item)
        mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = mask / 255.0

        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        img = torch.abs(img - F.max_pool2d(img, 3, 1, 1))
        img = img.detach().squeeze(0).squeeze(0).numpy()
        img[img >= 0.5] = 1
        img[img < 0.5] = 0
        boundary_path = os.path.join(save_folder, img_name)
        cv2.imwrite(boundary_path, img * 255)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate boundary labels from annotation')
    parser.add_argument('--mask', type=str, help='Folder path saves mask annotation')
    parser.add_argument('--save', type=str, help='Folder path saves generated boundary labels')
    args = parser.parse_args()
    generate_BD(args.mask, args.save)