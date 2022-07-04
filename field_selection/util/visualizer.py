import os
import ntpath
from . import util
import cv2


class Visualizer():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False

    # save image to the disk
    def save_images(self, results_dir, visuals, image_path, aspect_ratio=1.0):
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        for label, im in visuals.items():

            if label == 'fake_C':

                if not os.path.exists(os.path.join(results_dir, label)):
                    os.mkdir(os.path.join(results_dir, label))

                image_name = f'{name}_{label}.png'
                save_path = os.path.join(results_dir, label, image_name)
                print(save_path)
                h, w, _ = im.shape
                if aspect_ratio > 1.0:
                    im = cv2.resize(im, (h, int(w * aspect_ratio)), interpolation=cv2.INTER_CUBIC)
                if aspect_ratio < 1.0:
                    im = cv2.resize(im, (int(h / aspect_ratio), w), interpolation=cv2.INTER_CUBIC)

                util.save_image(im, save_path)