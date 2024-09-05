import argparse
from collections import OrderedDict
from torch.utils.data import DataLoader
from data.select_dataset import define_Dataset
from utils.utils_logger import get_logger
import json5
from utils import tensor2uint, find_best_checkpoint, imsave, rgb2ycbcr, calculate_psnr, calculate_ssim, calculate_niqe, \
    calculate_lpips,mkdir
import torch
import os
import random
import numpy as np
from models.select_network import define_G
import lpips


def test(opt: OrderedDict, args):
    opt['datasets']['test']['scale'] = opt['scale']
    opt['datasets']['test']['n_channels'] = opt['n_channels']
    opt['datasets']['test']['img_size'] = opt['netG']['img_size']
    opt['datasets']['test']['phase'] = 'test'
    opt['datasets']['test']['H_size'] = None
    opt['datasets']['test']['degradation_type'] = None
    opt['is_train'] = False

    model = define_G(opt)
    epoch, opt['path']['pretrained_netG'] = find_best_checkpoint(opt['path']['models'], net_type=opt['net_type'])

    pretrained_model = torch.load(opt['path']['pretrained_netG'], map_location=args.device)
    if 'params' in pretrained_model.keys():
        pretrained_model = pretrained_model['params']
    logger = get_logger('test', os.path.join(os.path.dirname(opt['path']['pretrained_netG']), '../..', 'test.log'))
    logger.info(opt['file_name'])
    logger.info(args.idx)
    logger.info(args.tile)
    model.load_state_dict(pretrained_model, strict=True)
    model = model.to(args.device)

    seed = 3407

    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    test_set = define_Dataset(opt['datasets']['test'])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
    lpips_ = lpips.LPIPS(net='alex').eval()
    avg_psnr = []
    avg_ssim = []
    avg_ssim_y = []
    avg_msssim_y = []
    avg_niqe_y = []
    avg_psnr_y = []
    avg_lpips = []
    window_size = 8
    model.eval()
    for idx, data in enumerate(test_loader):
        image_name_ext = os.path.basename(data['L_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)
        img_dir = os.path.join(os.path.dirname(opt['path']['pretrained_netG']),
                               '../../images/{}/{}'.format(str(opt['scale']), str(args.tile)), img_name)
        mkdir(img_dir)
        with torch.no_grad():
            img_lq = [i.float().to(args.device) for i in data['L']]
            _, _, h_old, w_old = img_lq[0].shape

            E = ttest(img_lq, model, opt['scale'], tile=args.tile)
            while type(E) is list:
                E = E[args.idx]
            H = data['H'][0].to(args.device)
            E_img = tensor2uint(E[:, :, :h_old * opt['scale'], :w_old * opt['scale']])
            H_img = tensor2uint(H)
            imsave(E_img, img_path=os.path.join(img_dir, img_name + '_{}_pre.png'.format(epoch)))
            imsave(H_img, img_path=os.path.join(img_dir, img_name + '_{}_gt.png'.format(epoch)))
            # -----------------------
            # calculate PSNRY
            # -----------------------
            current_psnr = calculate_psnr(E_img, H_img, border=opt['scale'], maxn=255)
            current_ssim = calculate_ssim(E_img, H_img, border=opt['scale'], maxn=255)
            current_lpips = calculate_lpips(E_img / 255, H_img / 255, lpips_, border=opt['scale'], maxn=1)
            E_img0 = rgb2ycbcr(E_img.astype(np.float32) / 255.) * 255.
            H_img0 = rgb2ycbcr(H_img.astype(np.float32) / 255.) * 255.
            current_ssim_y = calculate_ssim(E_img0, H_img0, border=opt['scale'], maxn=255)
            current_msssim_y = 0
            current_niqe_y = calculate_niqe(E_img, border=opt['scale'])
            current_psnr_y = calculate_psnr(E_img0, H_img0, border=opt['scale'], maxn=255)
        if not args.quite:
            logger.info(
                '{:->4d}--> {:>10s} | {:<4.2f} | {:<4.3f} | {:<4.3f} | {:<4.3f} | {:<4.3f} | {:<4.3f}'.format(idx,
                                                                                                              image_name_ext,
                                                                                                              current_psnr_y,
                                                                                                              current_ssim_y,
                                                                                                              current_ssim,
                                                                                                              current_msssim_y,
                                                                                                              current_niqe_y,
                                                                                                              current_lpips))
        avg_psnr.append(current_psnr)
        avg_ssim.append(current_ssim)
        avg_ssim_y.append(current_ssim_y)
        avg_msssim_y.append(current_msssim_y)
        avg_psnr_y.append(current_psnr_y)
        avg_niqe_y.append(current_niqe_y)
        avg_lpips.append(current_lpips)
    logger.info(
        '-- Average PSNRY|SSIMY|SSIM|MSSSIM|NIQE|LPIPS: {:<4.2f} | {:<4.3f} | {:<4.3f} | {:<4.3f} | {:<4.3f} | {:<4.3f}'.format(
            np.mean(avg_psnr_y),
            np.mean(avg_ssim_y),
            np.mean(avg_ssim),
            np.mean(avg_msssim_y),
            np.mean(avg_niqe_y),
            np.mean(avg_lpips)))


def ttest(img_lq: list, model, scale, tile=None):
    if tile is None:
        # test the image as a whole
        output = model(img_lq)

    else:
        # test the image tile by tile
        b1, c1, h1, w1 = img_lq[0].size()
        tile = min(tile, h1, w1)
        tile_overlap = 8
        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h1 - tile, stride)) + [h1 - tile]
        w_idx_list = list(range(0, w1 - tile, stride)) + [w1 - tile]
        E = []
        W = []

        flag = False
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patchs = []
                for img_ in img_lq:
                    b, c, h, w = img_.shape
                    s = int(h / h1)
                    in_patchs.append(
                        img_[:, :, h_idx * s:(h_idx + tile) * s, w_idx * s:(w_idx + tile) * s])
                out_patchs = model(in_patchs)
                for i, out_patch in enumerate(out_patchs):
                    if type(out_patch) is list:
                        if flag is False:
                            E.append([])
                            W.append([])

                        for j, out in enumerate(out_patch):
                            b, c, h, w = out.shape
                            if flag is False:
                                if h != tile * sf:
                                    E[i].append(torch.zeros(b, c, h1, w1).type_as(out))
                                    W[i].append(torch.zeros(b, c, h1, w1).type_as(out))
                                else:
                                    E[i].append(torch.zeros(b, c, h1 * sf, w1 * sf).type_as(out))
                                    W[i].append(torch.zeros(b, c, h1 * sf, w1 * sf).type_as(out))
                            if h != tile * sf:
                                out_mask = torch.ones_like(out)
                                E[i][j][:, :, h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out)
                                W[i][j][:, :, h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_mask)
                            else:
                                out_mask = torch.ones_like(out)
                                E[i][j][:, :, h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out)
                                W[i][j][:, :, h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(
                                    out_mask)

                    else:
                        if flag is False:
                            E.append(torch.zeros(b1, c1, h1 * sf, w1 * sf).type_as(out_patch))
                            W.append(torch.zeros(b1, c1, h1 * sf, w1 * sf).type_as(out_patch))

                        out_patch_mask = torch.ones_like(out_patch)
                        E[i][:, :, h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
                        W[i][:, :, h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)
                flag = True
        output = []
        for e, w in zip(E, W):
            if type(e) is list:
                output.append([])
                for i, j in zip(e, w):
                    output[-1].append(i.div(j))
            else:
                output.append(e.div(w))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str,
                        default='options/dat/x4/train_lora_datv2_3_0_0_d180_b64_s4_fdb_t96_wlfreezehb01234_r4.json')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-q', '--quite', type=bool, default=False)
    parser.add_argument('-n', '--net_type', type=str, default='G')
    parser.add_argument('-t', '--tile', type=int, default=0, help='0 is no tile')
    parser.add_argument('-i', '--idx', type=int, default=-1)
    parser.add_argument('-da', '--dataset', type=str, default='realsr')

    args = parser.parse_args()
    if args.tile == 0:
        args.tile = None

    with open(args.opt, 'r', encoding='utf-8') as f:
        json_str = f.read()
    opts = parse(args.opt, is_train=False)
    opts['net_type'] = args.net_type
    opts['file_name'] = os.path.basename(__file__)
    if args.dataset == 'drealsr':
        opts['datasets']['test']['dataroot_H'] = '../../Drealsr/x{}/Test_x{}/test_HR'.format(opts['scale'],
                                                                                             opts['scale'])
        opts['datasets']['test']['dataroot_L'] = '../../Drealsr/x{}/Test_x{}/test_LR'.format(opts['scale'],
                                                                                             opts['scale'])
    elif args.dataset == 'd2crealsr':
        opts['datasets']['test']['dataroot_H'] = '../../D2CRealSR/test_HR'
        opts['datasets']['test']['dataroot_L'] = '../../D2CRealSR/test_LR'

    test(opt=opts, args=args)
