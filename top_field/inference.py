import os
import warnings

warnings.filterwarnings('ignore')

import torch
from top_field.utils import utils, warp, image_utils
from top_field.models import end_2_end_optimization
from top_field.options import fake_options
import cv2 as cv
from PIL import Image
import numpy as np
from matplotlib.pylab import plt


class CONFIG:
    opt = fake_options.FakeOptions()
    opt.batch_size = 1
    opt.coord_conv_template = True
    opt.error_model = 'loss_surface'
    opt.error_target = 'iou_whole'
    opt.guess_model = 'init_guess'
    opt.homo_param_method = 'deep_homography'
    opt.load_weights_error_model = 'pretrained_loss_surface'
    opt.load_weights_upstream = 'pretrained_init_guess'
    opt.lr_optim = 1e-5
    opt.need_single_image_normalization = True
    opt.need_spectral_norm_error_model = True
    opt.need_spectral_norm_upstream = False
    opt.optim_criterion = 'l1loss'
    opt.optim_iters = 5
    opt.optim_method = 'stn'
    opt.optim_type = 'adam'
    opt.out_dir = './out'
    opt.prevent_neg = 'sigmoid'
    opt.template_path = './top_field/data/world_cup_template.png'
    opt.warp_dim = 8
    opt.warp_type = 'homography'

    result_folder = './result'


class Perspective2D:

    def __init__(self, test: bool = False):
        self.test = test
        self.result_folder = CONFIG.result_folder

        self.opt = CONFIG.opt

        self.template_image = self._convert_template_img()

        self.e2e = end_2_end_optimization.End2EndOptimFactory.get_end_2_end_optimization_model(self.opt)

    def _convert_template_img(self):
        # print(os.listdir('./'))
        template_img = cv.cvtColor(cv.imread(self.opt.template_path), cv.COLOR_BGR2RGB)
        cv.imwrite(os.path.join(self.result_folder, f'template.png'), template_img)
        template_img = template_img / 255.0

        if self.opt.coord_conv_template:
            template_img = image_utils.rgb_template_to_coord_conv_template(template_img)

        template_img = utils.np_img_to_torch_img(template_img)
        if self.opt.need_single_image_normalization:
            template_img = image_utils.normalize_single_image(template_img)

        return template_img

    def _convert_goal_img(self, frame):
        cv.imwrite(os.path.join(self.result_folder, f'frame.png'), frame)
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        img = Image.fromarray(np.uint8(img))
        img = img.resize([256, 256], resample=Image.NEAREST)
        img = np.array(img)

        if self.test:
            print('Original image:')
            plt.imshow(img)
            plt.show()
            print('Shape:', img.shape)

        img = utils.np_img_to_torch_img(img)
        if self.opt.need_single_image_normalization:
            img = image_utils.normalize_single_image(img)

        return img

    def _image_draw(self, goal_img):
        goal_image_draw = cv.cvtColor(goal_img, cv.COLOR_BGR2RGB)
        goal_image_draw = goal_image_draw / 255.0

        template_image_draw = cv.cvtColor(cv.imread(self.opt.template_path), cv.COLOR_BGR2RGB)
        template_image_draw = template_image_draw / 255.0
        template_image_draw = image_utils.rgb_template_to_coord_conv_template(template_image_draw)
        template_image_draw = utils.np_img_to_torch_img(template_image_draw)

        return goal_image_draw, template_image_draw

    def warmaped(self, frame, idx: int = 1) -> tuple:
        goal_image = self._convert_goal_img(frame)
        _, optim_homography = self.e2e.optim(goal_image[None], self.template_image)
        # print('HERE',optim_homography.cpu().shape)
        # print('HERE', optim_homography.shape)
        H_inv = torch.inverse(optim_homography)
        outshape = self.template_image.shape[1:3]

        goal_image_draw, template_image_draw = self._image_draw(frame)
        warped_frm = warp.warp_image(utils.np_img_to_torch_img(goal_image_draw)[None], H_inv, out_shape=outshape)[0]

        warped_frm_np = utils.torch_img_to_np_img(warped_frm)
        warped_frm_white = np.where(warped_frm_np!=0, 255, warped_frm_np)
        # print(warped_frm_n+p)
        warmap_image =  utils.torch_img_to_np_img(template_image_draw)
        warmap_image_main = warped_frm_white * 0.5 + utils.torch_img_to_np_img(
            template_image_draw) * 0.5
        # warmap_image = utils.torch_img_to_np_img(warped_frm) * 0.5 + utils.torch_img_to_np_img(
        #     template_image_draw) * 0.7

        if self.test:
            print('Warmap image:')
            plt.imshow(warmap_image_main)
            plt.show()
            print('shape:', warmap_image_main.shape)
            print('***************************')
        # print('HERE2', optim_homography.cpu().shape)
        return warmap_image_main, warmap_image, optim_homography.cpu()
