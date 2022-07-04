# Название
«Система видеоаналитики спортивных состязаний с использованием распределенной системы хранения информации»
Авторы: Семенов Андрей Максимович, Зараев Роберт Эдуардович

# Цель
Снизить издержки на анализ трансляций спортивных мероприятий за счет использования нейросетевых технологий и распределенной файловой системы.

# Целевая группа проекта
Футбольные клубы, спортивные федерации, спортивные телевизионные каналы, букмекерские компании.

# Pipeline работы нейросетевого комплекса
![Pipeline работы нейросетевого комплекса](https://github.com/In48semenov/football_analytics/blob/main/example_pipeline_neural_networks/pipeline.png)

# Руководство
1. git clone https://github.com/In48semenov/football_analytics.git
2. cd football_analytics
3. активация виртуального окружения
4. pip install requirements.txt
5. cкачивание и добавление весов для моделеей
   4.1. Перейти по ссылке: [Google Drive](https://drive.google.com/drive/folders/1FSpRM3VPV-BglIMqAo4Zu5dlPXXcURs8?usp=sharing)
   4.2. Скачать
      4.2.1. из папки field_selection скачать все файлы и поместить в field_selection/models_weights
      4.2.2. из папки yolo скачать все файлы и поместить в yolo_inference/models_weights
      4.2.3. из папки top_field перейти во внутренние папки, скачать из них все файлы и поместить в top_field/out/pretrained_init_guess и top_field/out/pretrained_loss_surface соответсвенно
6. создание директории results/detect_object, results/frame, results/warmap, results/warmap_with_pos_camera
7. по необходимости изменить параметры запуска в config-файле 'options.yaml'
8. запуск в терминале:
> python3 inference.py