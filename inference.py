import glob
from IPython.display import clear_output
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import yaml

from field_selection.inference import field_selection
from top_field.inference import Perspective2D
from yolo_inference.inference import detect_obj


class Analytics:
    """
    Complete Pipeline for Football Event Analysis.
    Args:

    Attributes:
        DIR_ORIG_IMAGES: path to folder with original frames.
        DIR_FS_IMAGE: path to folder with frames of the selected field.
        DIR_WARMAP: path to image directory after warmup conversion.
        DIR_WARMAP_WITH_POS_CAMERA: path to image folder after warmup conversion.
        PATH_TO_PHYSICAL_STATE_PLAYERS: path to image folder after warmup conversion with selected camera area.
        video_path: path to video file.
        fs_directory: path to temporary folder with original frames for 'field selection'.
        clear_dir: flag for clearing images after yolo inference.
    """

    DIR_ORIG_IMAGES = './results/frame'
    DIR_FS_IMAGE = './field_selection/results/fake_C'
    DIR_WARMAP = './results/warmap/'
    DIR_WARMAP_WITH_POS_CAMERA = 'results/warmap_with_pos_camera/'
    PATH_TO_PHYSICAL_STATE_PLAYERS = './results/coords.csv'

    def __init__(self):
        with open('options.yaml') as file:
            parameters = yaml.safe_load(file)['main']

        self.video_path = parameters['video_path']
        self.fs_directory = parameters['fs_directory']
        self.txt_directory = parameters['txt_directory']
        self.clear_dir = parameters['clear_dir']

    def _convert_box(self, box: list, image_shape: tuple) -> list:
        """
        Conversion of bonding box coordinates from yolo coordinates to image coordinates.
        Args:
            box (list): list of yolo coordinates.
            image_shape (tuple): image size.
        Returns:
            list: list of image coordinates.
        """

        # box[0] - center box by X
        # box[1] - center box by Y
        # box[2] - width
        # box[3] - height

        x_min = box[0] * image_shape[0] - box[2] * image_shape[0] / 2
        y_min = box[1] * image_shape[1] - box[3] * image_shape[1] / 2
        width = box[2] * image_shape[0]
        height = box[3] * image_shape[1]

        return [x_min, y_min, width, height]

    def _parsing_file(self, file, image_shape: tuple = (1280, 592)) -> tuple:
        """
        Parsing a txt file after yolo inference to get object labels and their coordinates.
        Attr:
            file (str) : path to txt file.
            image_shape (tuple) : frame size.
        Returns:
            tuple: list of labels and list of bbox.
        """
        with open(file, 'r') as f:
            coord = [values.split() for values in f.readlines()]

        labels = [int(value[0]) for value in coord]
        boxs = [np.array(value[1:], dtype=float) for value in coord]

        boxs = [self._convert_box(box, image_shape) for box in boxs]

        return labels, boxs

    def _clean_dir(self) -> None:
        """
        Clean up image directory after output.
        """

        file_format = ['txt', 'jpg', 'png']
        for j in file_format:
            files = glob.glob(f'results/detect_object/yolo_result/**/**/*.{j}', recursive=True)
            for i in files:
                try:
                    os.remove(i)
                except OSError as e:
                    print("Error: %s : %s" % (i, e.strerror))

    def _coordinate_to_scheme_field(self, optim_homography, path_txt: str) -> tuple:
        """
        Transferring the coordinates of the players to the scheme of the playing surface.
        Attr:
            optim_homography: homography matrix.
            path_txt (str): path to txt file after yolo inference.
        Returns:
            tuple: coordinates in the game surface diagram objects and their labels.
        """

        labels, boxs = self._parsing_file(path_txt)
        x_final_person, y_final_person = list(), list()
        x_final_ball, y_final_ball = list(), list()

        for idx, label in enumerate(labels):
            r = 10

            if boxs[idx][0] + boxs[idx][2] >= 1280:
                x = 1280 - r
            else:
                x = boxs[idx][0] + boxs[idx][2] / 2

            if boxs[idx][1] + boxs[idx][3] >= 720:
                y = 720 - r
            else:
                y = boxs[idx][1] + boxs[idx][3]
            x = torch.tensor(x / 1280 - 0.5).float()
            y = torch.tensor(y / 720 - 0.5).float()
            xy = torch.stack([x, y, torch.ones_like(x)])

            xy_warped = torch.matmul(torch.tensor(optim_homography), xy)

            xy_warped, z_warped = xy_warped.split(2, dim=1)
            xy_warped = 2.0 * xy_warped / (z_warped + 1e-8)
            x_warped, y_warped = torch.unbind(xy_warped, dim=1)

            x_warped = (x_warped.item() * 0.5 + 0.5) * 1050
            y_warped = (y_warped.item() * 0.5 + 0.5) * 680

            if x_warped > 1050:
                x_warped = 1045
            elif x_warped < 0:
                x_warped = 5
            if y_warped > 680:
                y_warped = 675
            elif y_warped < 0:
                y_warped = 5

            if label == 0:
                x_final_person.append(x_warped)
                y_final_person.append(y_warped)
            elif label == 32:
                x_final_ball.append(x_warped)
                y_final_ball.append(y_warped)

        return x_final_person, y_final_person, x_final_ball, y_final_ball, labels

    def _save_object(self, dataset: pd.DataFrame, frame_id: int, x_coords: list, y_coords: list,
                     cls: list) -> pd.DataFrame:
        """
        Formation of a dataset with the physical state of objects on frames.
        Attr:
            dataset (pd.DataFrame): original DataFrame.
            frame_id (int): id of frame.
            x_coords (list): x-coordinates of object.
            y_coords (list): y-coordinates of object.
            cls (list): object class.
        Returns:
            pd.DataFrame: with the physical state of objects.
        """

        curr_dataset = pd.DataFrame(columns=['frame_id', 'obj_id', 'x_coord', 'y_coord', 'class'])
        curr_dataset['frame_id'] = [frame_id for _ in range(len(x_coords))]
        curr_dataset['obj_id'] = [idx + 1 for idx in range(len(x_coords))]
        curr_dataset['x_coord'] = np.array(x_coords) / 10
        curr_dataset['y_coord'] = np.array(y_coords) / 10
        curr_dataset['class'] = cls
        dataset = pd.concat([dataset, curr_dataset], ignore_index=True)

        return dataset

    def _preprocess_filed_and_detectoin(self) -> None:
        """
        Stages of selecting the playing surface and detecting the necessary objects.
        """
        field_selection()  # highlighting the playing surface
        detect_obj()  # detecting objects

    def _calculate_physical_state(self) -> None:
        """
        Calculation of the physical state of objects.
        """
        videocap = cv2.VideoCapture(self.video_path)
        counter = 0
        df = pd.DataFrame(columns=['frame_id', 'obj_id', 'x_coord', 'y_coord', 'class'])

        perspective = Perspective2D(test=False)  # init class for warmap_transform

        while True:

            success, image = videocap.read()
            if not success:
                break

            path2frame = os.path.join(self.DIR_ORIG_IMAGES, f'frame{counter}.jpg')
            cv2.imwrite(path2frame, image)  # write original frame

            path2frame_fs = os.path.join(self.fs_directory, f'frame{counter}.jpg')
            cv2.imwrite(path2frame_fs, image)  # for field selection

            self._preprocess_filed_and_detectoin()

            pil_image = Image.fromarray(np.uint8(image[..., ::-1]))
            pil_image = pil_image.resize([1280, 720], resample=Image.NEAREST)
            image_ = np.array(pil_image)
            warmap_main, warmap, optim_homography = perspective.warmaped(image_)

            path_txt = os.path.join(os.path.join(self.txt_directory, f'frame{counter}_fake_C.txt'))

            x_person, y_person, x_ball, y_ball, labels = self._coordinate_to_scheme_field(optim_homography, path_txt)

            os.remove(path2frame_fs)
            os.remove(os.path.join(self.DIR_FS_IMAGE, f'frame{counter}_fake_C.png'))

            plt.imshow(warmap)
            plt.scatter(x_person, y_person, c='blue')
            plt.scatter(x_ball, y_ball, c='red')
            plt.savefig(os.path.join(self.DIR_WARMAP, f'warmap_{counter}.png'))
            plt.close()

            plt.imshow(warmap_main)
            plt.scatter(x_person, y_person, c='blue')
            plt.scatter(x_ball, y_ball, c='red')
            plt.savefig(os.path.join(self.DIR_WARMAP_WITH_POS_CAMERA, f'warmap_{counter}.png'))
            plt.close()

            x_final = x_person + x_ball
            y_final = y_person + y_ball

            df = self._save_object(df, counter, x_final, y_final, labels)

            counter += 1
            clear_output()

        df.to_csv(self.PATH_TO_PHYSICAL_STATE_PLAYERS, index=False)

    def __call__(self):
        """
        Pipeline launch.
        """
        self._calculate_physical_state()

        if self.clear_dir:
            self._clean_dir()


if __name__ == '__main__':
    analyzer = Analytics()
    analyzer()
