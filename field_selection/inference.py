import os
import yaml
from tqdm import tqdm

from field_selection.data.data_loader import CreateDataLoader
from field_selection.models.models import create_model
from field_selection.util.visualizer import Visualizer

from warnings import filterwarnings

filterwarnings('ignore')


class Args:
    def __init__(self, image_folder, save_folder, show_network):
        self.nThreads = 1
        self.batchSize = 1  # test code only supports batchSize = 1
        self.serial_batches = True  # no shuffle
        self.no_flip = True  # no flip
        self.continue_train = False
        self.dataroot = image_folder
        self.which_direction = 'AtoB'
        self.model = 'two_pix2pix'
        self.name = 'soccer_seg_detection_pix2pix'
        self.output_nc = 1
        self.dataset_mode = 'unaligned'
        self.which_model_netG = 'unet_256'
        self.phase = ''
        self.which_epoch = 'latest'
        self.results_dir = save_folder
        self.aspect_ratio = 1.0
        self.checkpoints_dir = './checkpoints'
        self.display_id = 1
        self.display_port = 8097
        self.display_winsize = 256
        self.fineSize = 256
        self.gpu_ids = []
        self.how_many = len(os.listdir(image_folder))
        self.init_type = 'normal'
        self.input_nc = 3
        self.isTrain = False
        self.loadSize = 256
        self.max_dataset_size = 10000000
        self.n_layers_D = 3
        self.ndf = 64
        self.ngf = 64
        self.no_dropout = False
        self.norm = 'batch'
        self.ntest = 'inf'
        self.resize_or_crop = 'resize_and_crop'
        self.which_model_netD = 'basic'
        self.show_network = show_network


def field_selection():
    with open('options.yaml') as file:
        parameters = yaml.safe_load(file)['filed_selection']

    image_folder = parameters['image_folder']
    save_folder = parameters['save_folder']
    show_network = parameters['show_network']

    opt = Args(image_folder, save_folder, show_network)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)

    for i, data in tqdm(enumerate(dataset)):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        visualizer.save_images(save_folder, visuals, img_path, aspect_ratio=opt.aspect_ratio)
