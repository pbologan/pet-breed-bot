# Руководство по работе с датасетами

## Доступные датасеты

### 1. Oxford-IIIT Pet Dataset
- **Размер**: ~800MB
- **Изображений**: ~7,400
- **Классов**: 37 (25 пород собак, 12 пород кошек)
- **Источник**: https://www.robots.ox.ac.uk/~vgg/data/pets/
- **Описание**: Высококачественный датасет с различными позами и освещением

### 2. Stanford Dogs Dataset
- **Размер**: ~750MB
- **Изображений**: ~20,000
- **Классов**: 120 пород собак
- **Источник**: http://vision.stanford.edu/aditya86/ImageNetDogs/
- **Описание**: Подмножество ImageNet, специализированное на породах собак

### 3. CUB-200-2011 Birds Dataset
- **Размер**: ~1.1GB
- **Изображений**: 11,788
- **Классов**: 200 видов птиц
- **Источник**: https://data.caltech.edu/records/65de6-vp158
- **Описание**: Детализированный датасет североамериканских птиц

### 4. Kaggle Cat Breeds Datasets
- **Различные датасеты** доступны на Kaggle
- Требуют API ключ Kaggle для загрузки

## Автоматическая загрузка

### Загрузка всех публичных датасетов

```bash
python scripts/download_datasets.py --dataset all
```

### Загрузка конкретного датасета

```bash
# Oxford-IIIT Pet Dataset (собаки и кошки)
python scripts/download_datasets.py --dataset oxford-pets

# Stanford Dogs Dataset
python scripts/download_datasets.py --dataset stanford-dogs

# CUB-200-2011 Birds Dataset
python scripts/download_datasets.py --dataset birds
```

### Указание директории вывода

```bash
python scripts/download_datasets.py --dataset oxford-pets --output-dir /path/to/data
```

## Загрузка с Kaggle

### Настройка Kaggle API

1. Установите Kaggle CLI:
```bash
pip install kaggle
```

2. Получите API токен:
   - Зайдите на https://www.kaggle.com/account
   - Прокрутите до раздела "API"
   - Нажмите "Create New API Token"
   - Скачается файл `kaggle.json`

3. Настройте credentials:
```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Список популярных датасетов

```bash
python scripts/download_kaggle_datasets.py --list-popular
```

### Загрузка датасета с Kaggle

```bash
# Cat Breeds Dataset
python scripts/download_kaggle_datasets.py --dataset-id ma7555/cat-breeds-dataset

# Dog Breed Identification
python scripts/download_kaggle_datasets.py --dataset-id stanford-dogs/dog-breed-identification

# 70 Dog Breeds Image Data Set
python scripts/download_kaggle_datasets.py --dataset-id gpiosenka/70-dog-breedsimage-data-set
```

## Ручная загрузка

Если автоматическая загрузка не работает, скачайте датасеты вручную:

### Oxford-IIIT Pet Dataset
1. Перейдите на https://www.robots.ox.ac.uk/~vgg/data/pets/
2. Скачайте:
   - `images.tar.gz` (изображения)
   - `annotations.tar.gz` (аннотации)
3. Распакуйте в `data/raw/oxford-pets/`

### Stanford Dogs Dataset
1. Перейдите на http://vision.stanford.edu/aditya86/ImageNetDogs/
2. Скачайте:
   - `images.tar` (изображения)
   - `annotation.tar` (аннотации)
   - `lists.tar` (train/test splits)
3. Распакуйте в `data/raw/stanford-dogs/`

### CUB-200-2011 Birds Dataset
1. Перейдите на https://data.caltech.edu/records/65de6-vp158
2. Скачайте `CUB_200_2011.tgz`
3. Распакуйте в `data/raw/cub-200-2011/`

## Структура после загрузки

```
data/raw/
├── oxford-pets/
│   ├── images/
│   │   ├── Abyssinian_1.jpg
│   │   ├── Abyssinian_2.jpg
│   │   └── ...
│   └── annotations/
│       ├── trimaps/
│       └── xmls/
├── stanford-dogs/
│   ├── Images/
│   │   ├── n02085620-Chihuahua/
│   │   ├── n02085782-Japanese_spaniel/
│   │   └── ...
│   ├── Annotation/
│   └── lists/
├── cub-200-2011/
│   └── CUB_200_2011/
│       ├── images/
│       │   ├── 001.Black_footed_Albatross/
│       │   └── ...
│       └── images.txt
└── cats/
    └── (Kaggle datasets)
```

## Следующие шаги

После загрузки датасетов:

1. **Подготовьте данные**:
```bash
python scripts/prepare_data.py --input-dir data/raw --output-dir data/processed
```

2. **Создайте карту меток**:
```bash
python scripts/prepare_data.py --create-label-map
```

3. **Обучите модель**:
```bash
python -m app.ml.train --data-dir data/processed --epochs 10
```

## Рекомендации

### Для начала работы
Рекомендуем начать с **Oxford-IIIT Pet Dataset**:
- Небольшой размер (~800MB)
- Хорошее качество изображений
- Включает собак и кошек
- Легко загружается автоматически

```bash
python scripts/download_datasets.py --dataset oxford-pets
```

### Для продвинутой модели
Используйте комбинацию датасетов:
1. Oxford-IIIT Pet Dataset (базовый)
2. Stanford Dogs Dataset (больше пород собак)
3. Kaggle Cat Breeds (больше пород кошек)
4. CUB-200-2011 (птицы)

### Оптимизация хранилища
Если не хватает места:
- Загружайте датасеты по одному
- Удаляйте архивы после распаковки
- Используйте только нужные классы

## Устранение проблем

### Ошибка загрузки
```bash
# Проверьте интернет-соединение
ping www.robots.ox.ac.uk

# Попробуйте повторно
python scripts/download_datasets.py --dataset oxford-pets
```

### Ошибка распаковки
```bash
# Проверьте свободное место
df -h

# Удалите поврежденные архивы и загрузите заново
rm data/raw/oxford-pets/*.tar.gz
python scripts/download_datasets.py --dataset oxford-pets
```

### Kaggle API не работает
```bash
# Проверьте установку
kaggle --version

# Проверьте credentials
cat ~/.kaggle/kaggle.json

# Переустановите
pip uninstall kaggle
pip install kaggle
```

## Дополнительные ресурсы

- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Papers with Code - Animal Recognition](https://paperswithcode.com/task/animal-recognition)
- [Roboflow Universe - Pet Datasets](https://universe.roboflow.com/)
