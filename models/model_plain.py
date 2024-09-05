from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G
from models.model_base import ModelBase
from models.loss import *

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip
from utils.utils_image import mkdir
from thop import clever_format, profile


class ModelPlain(ModelBase):
    """Train with pixel loss"""

    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']  # training option
        self.logger = get_logger('train')
        self.netG = define_G(opt)
        self.G_lossfn = []
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    def describe_gflops(self, network):
        network = self.get_bare_model(network)
        flops, params = profile(network, inputs=(self.L,))
        flops, params = clever_format([flops, params], '%.3f')
        msg = 'Thop Calculate Network GFLOPs:{}, Params:{}'.format(flops, params)
        self.logger.info(msg)
        return msg

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()  # load model
        self.netG.train()  # set training mode,for BN
        self.define_loss()  # define loss
        self.define_optimizer()  # define optimizer
        self.load_optimizers()  # load optimizer
        self.define_scheduler()  # define scheduler
        self.log_dict = OrderedDict()  # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            self.logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')


    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            self.logger.info('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label, save_best=False):
        temp_save_dir = self.save_dir
        if save_best is True:
            self.save_dir = os.path.join(self.save_dir, 'best')
            mkdir(self.save_dir)
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)
        self.save_dir = temp_save_dir

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        self.G_lossfn_type = self.opt_train['G_lossfn_type']
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

        def get_loss_fn(lossfn_type: str):
            if lossfn_type == 'l1':
                return nn.L1Loss().to(self.device)
            elif lossfn_type == 'l2sum':
                return nn.MSELoss(reduction='sum').to(self.device)
            else:
                try:
                    return (eval(lossfn_type)(self.opt).to(self.device))
                except Exception as e:
                    raise NotImplementedError('Loss type [{:s}] is not found and {}'.format(lossfn_type, e))

        for lossfn_type in self.G_lossfn_type:
            if type(lossfn_type) is list:
                ts = []
                for lt in lossfn_type:
                    ts.append(get_loss_fn(lt))
                self.G_lossfn.append(ts)
            else:
                self.G_lossfn.append(get_loss_fn(lossfn_type))
                # ----------------------------------------

    # define optimizer
    # ----------------------------------------
    def define_optimizer(self, print_optimize=False):
        G_optim_params = []
        optimize_params = []
        trainable_params = 0
        all_param = 0
        task = self.opt['task']
        if 'lora' in task:
            init_bool = False
        else:
            init_bool = True
        for k, v in self.netG.named_parameters():
            all_param += v.numel()
            v.requires_grad = init_bool
            if 'lora_' in k:
                v.requires_grad = True
            if v.requires_grad:
                G_optim_params.append(v)
                trainable_params += v.numel()
                optimize_params.append(k)
            else:
                pass
        if print_optimize:
            for k in optimize_params:
                self.logger.info('Params [{:s}] will optimize.'.format(k))
        self.logger.info(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'])
        else:
            raise NotImplementedError

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'StepLR':
            self.schedulers.append(lr_scheduler.StepLR(self.G_optimizer,
                                                       self.opt_train['G_scheduler_milestones'],
                                                       self.opt_train['G_scheduler_gamma']
                                                       ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer,
                                                                            self.opt_train['G_scheduler_periods'],
                                                                            self.opt_train[
                                                                                'G_scheduler_restart_weights'],
                                                                            self.opt_train['G_scheduler_eta_min']
                                                                            ))
        else:
            raise NotImplementedError

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = []
        for d in data['L']:
            if type(d) is list:
                self.L.append([dd.float().to(self.device) for dd in d])
            else:
                self.L.append(d.float().to(self.device))

        if need_H:
            self.H = []
            for d in data['H']:
                if type(d) is list:
                    self.H.append([dd.float().to(self.device) for dd in d])
                else:
                    self.H.append(d.float().to(self.device))

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self, current_step=None):
        self.E = self.netG(self.L)
        if type(self.E) is list:
            if type(self.H) is list:
                # 自动用HR填充不足
                if len(self.H) < len(self.E):
                    for _ in range(len(self.E) - len(self.H)):
                        self.H.insert(-1, self.H[0])
                else:
                    self.H = self.H[:len(self.E)]
                if len(self.L) < len(self.E):
                    self.L = self.L + self.L * (len(self.E) - len(self.L))
                else:
                    self.L = self.L[:len(self.E)]
            else:
                self.H = [self.H] * len(self.E)
        else:
            self.E = [self.E]
            self.H = [self.H]

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        G_loss = []
        for i, func in enumerate(self.G_lossfn):
            if type(func) is list:
                for j, f in enumerate(func):
                    G_loss.append(self.G_lossfn_weight[i][j] * f(self.E[i], self.H[i]))
                    self.log_dict['G' + self.G_lossfn_type[i][j]] = G_loss[-1].item()
            else:
                G_loss.append(self.G_lossfn_weight[i] * func(self.E[i], self.H[i]))
                self.log_dict['G' + self.G_lossfn_type[i]] = G_loss[-1].item()
        G_loss = sum(G_loss)
        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'],
                                           norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train[
            'G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train[
            'G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()
        self.log_dict['epoch'] = current_step
        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self, tile=None):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        self.logger.info(msg)
        return msg

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        self.logger.info(msg)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        self.logger.info(msg)
        return msg
