import random
import torch.utils.data as data
from utils import uint2single, modcrop, get_image_paths, imresize_np, imread_uint, augment_img, single2tensor3


class DatasetSR(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetSR, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = get_image_paths(opt['dataroot_H'])
        self.paths_L = get_image_paths(opt['dataroot_L'])
        if self.opt['H_size'] is not None:
            self.patch_size = self.opt['H_size']
        else:
            self.patch_size = 384

        self.L_size = self.patch_size // self.sf

        if self.opt['degradation_type'] is not None:
            self.degradation_type = self.opt['degradation_type']
        else:
            self.degradation_type = 'sr'
        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L),
                                                                                           len(self.paths_H))

    def __getitem__(self, index):

        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = imread_uint(H_path, self.n_channels)
        img_H = uint2single(img_H)
        # ------------------------------------
        # modcrop
        # ------------------------------------
        img_H = modcrop(img_H, self.sf)

        # ------------------------------------
        # get L image
        # ------------------------------------
        if self.paths_L:
            # --------------------------------
            # directly load L image
            # --------------------------------
            L_path = self.paths_L[index]
            img_L = imread_uint(L_path, self.n_channels)
            img_L = uint2single(img_L)

        else:
            # --------------------------------
            # sythesize L image via matlab's bicubic
            # --------------------------------
            H, W = img_H.shape[:2]
            img_L = imresize_np(img_H, 1 / self.sf, True)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':
            H, W, C = img_L.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H = augment_img(img_L, mode=mode), augment_img(img_H, mode=mode)
        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = single2tensor3(img_H), single2tensor3(img_L)

        if L_path is None:
            L_path = H_path

        return {'L': [img_L], 'H': [img_H], 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
