from boto3.session import Session
import time
import os
import numpy as np
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

load_dotenv()


class S3_client:
    """
    Класс для подключения к S3
        aws_access_key: пользовательский ключ доступа
        aws_secret_key: секретный ключ пользователя
        bucket_name: названия бакета для взаимодействия
        session: сессия
        s3_client: клиент подключения к хранилищу S3
    """

    def __init__(self, bucket_name):
        self.aws_access_key = os.environ.get(f'ACCESS_KEY')
        self.aws_secret_key = os.environ.get(f'SECRET_KEY')
        self.bucket_name = bucket_name

        self.session = Session(aws_access_key_id=self.aws_access_key, aws_secret_access_key=self.aws_secret_key)
        self.s3_client = self.session.client('s3')

    def __delete_local(self, local_path) -> None:
        """
        Метод для удаления файлы из локальной среды
            :param local_path: удаляемый файл
        """
        os.remove(local_path)

    def __delete_s3(self, name_file_s3) -> None:
        """
        Метод для удаления файла из хранилища S3
            :param name_file_s3: удаляемый объект
            :return:
        """
        self.s3_client.delete_object(Bucket=self.bucket_name, Key=name_file_s3)

    def download_file(self, s3_file, local_path, delete_file_s3: bool = False) -> dict:
        """
        Метод загружает объект из хранилища S3
            :param s3_file: название объекта в хранилище s3
            :param local_path: путь куда загружается файл
            :param delete_file_s3: флаг, обозначающий удалять объект в S3 после загрузки или нет (True - удалить (рекомендуется для тестирования))
            :return: словарь, состоящий из метрик (время выполнения, размер объекта, скорость загрузки)
        """
        try:
            start_time = time.time()
            self.s3_client.download_file(self.bucket_name, s3_file, local_path)
            end_time = time.time() - start_time
            size_file = os.path.getsize(local_path) * 1e-6
            IO = size_file / end_time
            if delete_file_s3:
                self.__delete_s3(s3_file)

        except Exception as e:
            end_time = np.nan
            size_file = np.nan
            IO = np.nan

        return {'time': end_time, 'size_file': size_file, 'IO': IO}

    def upload_file(self, local_path: str, s3_file: str, delete_local_file: bool = False) -> dict:
        """
        Метод загружает обект в хранилище S3
            :param local_path: путь откуда загружается файл
            :param s3_file: название объекта в хранилище s3
            :param delete_local_file: флаг, обозначающий удалять загружаемый объект нет (True - удалить (рекомендуется для тестирования))
            :return: словарь, состоящий из метрик (время выполнения, размер объекта, скорость загрузки)
        """
        try:
            start_time = time.time()
            self.s3_client.upload_file(local_path, self.bucket_name, s3_file)
            end_time = time.time() - start_time
            size_file = os.path.getsize(local_path) * 1e-6
            IO = size_file / end_time
            if delete_local_file:
                self.__delete_local(local_path)

        except Exception as e:
            end_time = np.nan
            size_file = np.nan
            IO = np.nan

        return {'time': end_time, 'size_file': size_file, 'IO': IO}

    def get_files(self):
        """
        Метод находит все файлы в бакете
        :return: возвращает список файллов, хранящихся в бакете
        """
        S3 = self.session.resource('s3')
        bucket = S3.Bucket(self.bucket_name)

        return [file.key for file in bucket.objects.all()]


if __name__ == '__main__':
    number_iteration = 10
    s3 = S3_client(bucket_name='socker.games.coursjob')
    df_write = pd.DataFrame(columns=['file_name', 'time', 'size_file', 'IO'])
    df_read = pd.DataFrame(columns=['file_name', 'time', 'size_file', 'IO'])
    for file in tqdm(s3.get_files()):
        for _ in tqdm(range(number_iteration)):
            metrics_write = s3.download_file(s3_file=file, local_path=f'./files_example/{file}',
                                             delete_file_s3=True)
            metrics_write['file_name'] = file
            df_write = df_write.append(metrics_write, ignore_index=True)

            metrics_read = s3.upload_file(local_path=f'./files_example/{file}', s3_file=file,
                                          delete_local_file=True)
            metrics_read['file_name'] = file
            df_read = df_read.append(metrics_read, ignore_index=True)

    df_write.to_csv('./Test_results/S3_wrtie.csv', index=False)
    df_read.to_csv('./Test_results/S3_read.csv', index=False)

    print(df_write)
    print(df_read)
