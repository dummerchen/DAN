import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from tqdm import tqdm
from utils import *
from data.select_dataset import define_Dataset
from models.select_model import define_Model
from main_test_realsr import ttest
import lpips
def main(json_path='options/swinir/x4/train_lora_swinirv6_1_0_0_d180_b64_f2_s4_fdb_t96_wlfreezehb01234_r4.json'):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', default=0, help='local rank')
    parser.add_argument('--dist', default=False, help='dist')
    parser.add_argument('--idx', default=0, type=int, help='compare idx')
    args = parser.parse_args()
    opt = parse(args.opt, is_train=True)
    if parser.parse_args().dist is False:
        opt['dist'] = False
        print('not use dist')
    else:
        print('use dist')
    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))
        lpips_ = lpips.LPIPS(net='alex')

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    init_iter_G, init_path_G = find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = find_last_checkpoint(opt['path']['models'], net_type='E')
    if init_path_G != None:
        opt['path']['pretrained_netG'] = init_path_G
        opt['train']['G_param_strict'] = True
    if init_path_E != None:
        opt['path']['pretrained_netE'] = init_path_E
        opt['train']['E_param_strict'] = True

    init_iter_optimizerG, init_path_optimizerG = find_last_checkpoint(opt['path']['models'],
                                                                      net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        save(opt)
    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        logger = utils_logger.get_logger(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
        logger.info(os.path.basename(__file__))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = 3407
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':

            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            opt['train']['train_size'] = train_size
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True,
                                                   seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'] // opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers'] // opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        elif phase == 'eval':
            eval_set = define_Dataset(dataset_opt)
            eval_loader = DataLoader(eval_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    # if opt['rank'] == 0:
    # logger.info(model.info_network())
    # logger.info(model.info_params())

    if opt['iter'] is None:
        opt['iter'] = 1500000
        opt['repeat'] = 1

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    best_save_psnr = 0
    repeat = 1
    if opt['tile'] is None:
        tile = opt['netG']['img_size']
    elif opt['tile'] is False:
        # full test
        tile = None
    else:
        tile = opt['tile']

    print(tile)
    if opt['Epoch'] is None:
        Epoch = 100000
    else:
        Epoch = opt['Epoch']
    for epoch in range(Epoch):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        bar = tqdm(train_loader)
        for i, train_data in enumerate(bar):
            current_step += 1

            model.update_learning_rate(current_step)
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, int(current_step),
                                                                          model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)
                # wandb.log(logs)
            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % (opt['train']['checkpoint_save'] // repeat) == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(int(current_step))

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % (opt['train']['checkpoint_test'] // repeat) == 0 and opt['rank'] == 0:

                avg_psnr = 0.0
                avg_ssim = 0.0
                avg_ssim_y = 0.0
                avg_lpips = 0.0
                avg_psnr_y = 0.0
                avg_niqe_y = 0.0
                idx = 0
                model.netG.eval()

                ori_len = len(model.L)
                if current_step < 0:
                    try:
                        loader = eval_loader
                    except:
                        loader = test_loader
                else:
                    loader = test_loader
                for test_data in loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    model.feed_data(test_data)

                    _, _, h_old, w_old = model.L[0].size()
                    window_size = opt['netG']['window_size']
                    if window_size is not None:
                        lr = [img_pad_window_size(i, window_size) for i in model.L] * ori_len
                    else:
                        lr = model.L * ori_len
                    with torch.no_grad():
                        E_img = ttest(lr, model.netG, scale=opt['scale'], tile=tile)
                        while type(E_img) is list:
                            E_img = E_img[args.idx]
                    H_img = tensor2uint(model.H[0].detach().float().cpu())
                    E_img = tensor2uint(E_img[:, :, :h_old * opt['scale'], :w_old * opt['scale']])

                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    current_psnr = calculate_psnr(E_img, H_img, border=border, maxn=255)
                    current_lpips = calculate_lpips(E_img / 255, H_img / 255, lpips_=lpips_, border=border, maxn=1)
                    current_ssim = calculate_ssim(E_img, H_img, border=opt['scale'], maxn=255)
                    E_img0 = rgb2ycbcr(E_img.astype(np.float32) / 255.) * 255.
                    H_img0 = rgb2ycbcr(H_img.astype(np.float32) / 255.) * 255.
                    current_ssim_y = calculate_ssim(E_img0, H_img0, border=opt['scale'], maxn=255)
                    current_psnr_y = calculate_psnr(E_img0, H_img0, border=border, maxn=255)
                    current_niqe_y = calculate_niqe(E_img0, border=opt['scale'])
                    logger.info(
                        '{:->4d}--> {:>10s} | {:<4.2f} | {:<4.3f} | {:<4.3f} | {:<4.3f}| {:<4.3f}'.format(idx, image_name_ext,
                                                                                               current_psnr_y,
                                                                                               current_ssim,
                                                                                               current_ssim_y,
                                                                                               current_lpips,
                                                                                               current_niqe_y
                                                                                               ))
                    avg_psnr += current_psnr
                    avg_ssim_y += current_ssim_y
                    avg_ssim += current_ssim
                    avg_lpips += current_lpips
                    avg_psnr_y += current_psnr_y
                    avg_niqe_y += current_niqe_y
                avg_psnr = avg_psnr / idx
                avg_ssim_y = avg_ssim_y / idx
                avg_ssim = avg_ssim / idx
                avg_lpips = avg_lpips / idx
                avg_psnr_y = avg_psnr_y / idx
                avg_niqe_y = avg_niqe_y / idx

                # testing log
                logger.info(
                    '<epoch:{:3d}, iter:{:8,d}, Average PSNR|SSIMY|SSIM|LPIPS|NIQE: {:<.2f} | {:<.3f} | {:<.3f}| {:<.3f}| {:<.3f}\n'
                        .format(epoch, int(current_step), avg_psnr_y, avg_ssim_y,avg_ssim, avg_lpips, avg_niqe_y))
                model.netG.train()
                if np.mean(avg_psnr) > best_save_psnr:
                    best_save_psnr = np.mean(avg_psnr)
                    model.save(current_step, save_best=True)



if __name__ == '__main__':
    main()
