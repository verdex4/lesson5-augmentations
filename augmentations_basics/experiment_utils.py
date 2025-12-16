import os
import torch
from torchvision import transforms
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import random
from PIL import Image, ImageFile
from custom_augs import RandomBlur, RandomPerspective, RandomBrightnessContrast
from augmentations_basics.extra_augs import get_extra_augs, AddGaussianNoise
from custom_augs import RandomBlur
from augmentations_basics.datasets import CustomImageDataset
from augmentations_basics.visualization_utils import (save_plot, show_single_augmentation,
                                                      show_multiple_augmentations,
                                                      show_images, plot_size_stats,
                                                      plot_dataset_sizes)
import time
import tracemalloc
import psutil
import gc

PROJECT_ROOT = "C:/Users/verdex/lesson5-augmentations"

def split_train_val(train_path=f'{PROJECT_ROOT}/data/train', 
                    val_path=f'{PROJECT_ROOT}/data/val', 
                    val_ratio=0.2):
    """
    Разделяет тренировочные данные на train/val
    """
    if os.path.exists(val_path):
        print("Уже есть валидационная выборка")
        return
    
    # Создаём папку val если её нет
    os.makedirs(val_path)
    
    # Для каждого класса
    for class_name in os.listdir(train_path):
        class_train_path = os.path.join(train_path, class_name)
        class_val_path = os.path.join(val_path, class_name)
        
        # Пропускаем если не папка
        if not os.path.isdir(class_train_path):
            continue
            
        # Создаём папку класса в val
        os.makedirs(class_val_path, exist_ok=True)
        
        # Получаем все файлы в классе
        all_images = [f for f in os.listdir(class_train_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(all_images) == 0:
            continue
            
        # Разделяем на train/val
        train_images, val_images = train_test_split(
            all_images, 
            test_size=val_ratio,
            random_state=42  # для воспроизводимости
        )
        
        # Переносим val изображения
        for img in val_images:
            src = os.path.join(class_train_path, img)
            dst = os.path.join(class_val_path, img)
            shutil.move(src, dst)
            
        print(f"Класс {class_name}: {len(train_images)} train, {len(val_images)} val")
    
    print("Разделение завершено")

def get_random_samples(num_classes, samples_per_class, target_size=(224, 224), 
                       root_path=f'{PROJECT_ROOT}/data/train'):
    # 1. Получаем все доступные классы (папки)
    all_classes = [d for d in os.listdir(root_path) 
                  if os.path.isdir(os.path.join(root_path, d))]
    
    if len(all_classes) < num_classes:
        print(f"Классов меньше чем нужно: {len(all_classes)} < {num_classes}, взято {len(all_classes)} классов")
        num_classes = len(all_classes)
    
    # 2. Выбираем случайные классы
    selected_classes = random.sample(all_classes, num_classes)
    
    # 3. Для каждого класса берём случайные изображения
    result = []  # [(image, class_name), ...]
    
    for class_name in selected_classes:
        class_path = os.path.join(root_path, class_name)
        
        # Все изображения в папке класса
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(images) < samples_per_class:
            print(f"В классе '{class_name}' меньше {samples_per_class} изображений")
            continue
            
        # Выбираем случайные изображения
        selected_images = random.sample(images, samples_per_class)
        
        for img_name in selected_images:
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            result.append((img, class_name))
    
    return result

def pipe_standard_augmentations():
    standard_augs = [
        ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=1.0)), # поворот
        ("RandomCrop", transforms.RandomCrop(200, padding=20)), # обрезка
        # изменение цветовых характеристик
        ("ColorJitter", transforms.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.6, hue=0.1)),
        ("RandomRotation", transforms.RandomRotation(degrees=60)), # поворот
        ("RandomGrayscale", transforms.RandomGrayscale(p=1.0)) # серое изображение
    ]

    images = get_random_samples(5, 1)
    all_augs = [] # список для аугментаций без названий

    for img_tup, aug_tup in zip(images, standard_augs):
        original_img, class_name = img_tup
        aug_name, aug = aug_tup
        aug_transform = transforms.Compose([
            aug,
            transforms.ToTensor()
        ])
        aug_img = aug_transform(original_img)
        all_augs.append(aug)
        # отдельные аугментации
        show_single_augmentation(original_img, aug_img, aug_name, 
                                 folder=f'{PROJECT_ROOT}/plots/standard_augs',
                                 filename=f'{aug_name}.png') 
    
    # все аугментации на одном изображении
    original_img, class_name = random.sample(images, 1)[0]
    aug_transform = transforms.Compose([
        *all_augs,
        transforms.ToTensor()
    ])
    aug_img = aug_transform(original_img)
    show_single_augmentation(original_img, aug_img, title="Все аугментации вместе",
                             folder=f'{PROJECT_ROOT}/plots/standard_augs',
                             filename="all_augs.png")

def test_custom_augmentations():
    dataset = CustomImageDataset(f"{PROJECT_ROOT}/data/train", transform=None, target_size=(224, 224))

    blur_aug = transforms.Compose([
        transforms.ToTensor(),
        RandomBlur(kernel_size_range=(5, 9),
                   sigma_range=(0.8, 2))
    ])
    perspective_aug = transforms.Compose([
        transforms.ToTensor(),
        RandomPerspective(distortion_scale=0.6, p=1)
    ])
    brightness_contrast_aug = transforms.Compose([
        transforms.ToTensor(),
        RandomBrightnessContrast(brightness_range=(0.6, 1.5),
                                 contrast_range=(0.6, 1.5))
    ])

    custom_augs = [
        ("Random Blur", blur_aug), 
        ("Random Perspective", perspective_aug), 
        ("Random Brightness-Contrast", brightness_contrast_aug)
    ]

    original_img, label = random.choice(dataset)
    original_title = "Оригинал"
    aug_images = []
    aug_titles = []

    for name, aug in custom_augs:
        aug_img = aug(original_img)
        aug_images.append(aug_img)
        aug_titles.append(name)
        show_single_augmentation(original_img, aug_img, title=name,
                                 folder=f'{PROJECT_ROOT}/plots/custom_augs',
                                 filename=f"{name}.png")
    
    extra_augs = get_extra_augs()

    for name, aug in extra_augs:
        aug_img = aug(original_img)
        aug_images.append(aug_img)
        aug_titles.append(name)

    all_images = [original_img] + aug_images
    all_titles = [original_title] + aug_titles
    show_images(all_images, all_titles, nrow=5, ncol=2, 
                title="Сравнение кастомных аугментаций и готовых",
                folder=f"{PROJECT_ROOT}/plots/custom_augs",
                filename="augs_compare.png")

def dataset_analysis():
    train_folder = f"{PROJECT_ROOT}/data/train"
    classes = []
    images_count = []
    all_widths, all_heights = [], []

    for class_name in os.listdir(train_folder):
        img_count = 0
        min_size = [float('inf'), float('inf')]
        max_size = [0, 0]
        widths, heights = [], []

        class_path = os.path.join(train_folder, class_name)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            with Image.open(img_path) as img:
                w, h = img.size
                if w * h < min_size[0] * min_size[1]:
                    min_size[0], min_size[1] = w, h
                if w * h > max_size[0] * max_size[1]:
                    max_size[0], max_size[1] = w, h
                widths.append(w)
                heights.append(h)
                all_widths.append(w)
                all_heights.append(h)
                img_count += 1

        min_size = tuple(min_size)
        max_size = tuple(max_size)

        plot_size_stats(min_size, max_size, widths, heights, class_name,
                        folder=f"{PROJECT_ROOT}/plots/dataset_analysis",
                        filename=f"sizes_class_{class_name}.png")
        classes.append(class_name)
        images_count.append(img_count)
    
    min_width = min(all_widths)
    max_width = max(all_widths)
    min_height = min(all_heights)
    max_height = max(all_heights)
    min_size = min(zip(all_widths, all_heights), key=lambda size: size[0]*size[1])
    max_size = max(zip(all_widths, all_heights), key=lambda size: size[0]*size[1])
    avg_size = (
        round(sum(all_widths)/len(all_widths)),
        round(sum(all_heights)/len(all_heights))
    )

    print("========== Общая статистика по датасету ==========")
    print(f"Минимальная ширина: {min_width}")
    print(f"Максимальная ширина: {max_width}")
    print(f"Минимальная высота: {min_height}")
    print(f"Максимальная высота: {max_height}")
    print(f"Минимальный размер: {min_size}")
    print(f"Максимальный размер: {max_size}")
    print(f"Средний размер: {avg_size}")

    plot_dataset_sizes(all_widths, all_heights, classes, images_count,
                       folder=f"{PROJECT_ROOT}/plots/dataset_analysis",
                       filename="all_dataset.png")

class AugmentationPipeline:
    def __init__(self):
        self.augmentations = {}  # {name: transform}
    
    def add_augmentation(self, name, aug):
        self.augmentations[name] = aug
    
    def remove_augmentation(self, name):
        self.augmentations.pop(name, None)
    
    def apply(self, image):
        result = image
        for aug in self.augmentations.values():
            result = aug(result)
        return result
    
    def get_augmentations(self):
        return list(self.augmentations.keys())

def create_configurations():
    light = AugmentationPipeline()
    light.add_augmentation('light_rotation', transforms.RandomRotation(10))
    light.add_augmentation('light_brightness_contrast', 
                           RandomBrightnessContrast(brightness_range=(0.9, 1.1), 
                                                    contrast_range=(0.9, 1.1)))
    
    medium = AugmentationPipeline()
    medium.add_augmentation('medium_rotation', transforms.RandomRotation(20))
    medium.add_augmentation('medium_brightness_contrast', 
                            RandomBrightnessContrast(brightness_range=(0.8, 1.2), 
                                                     contrast_range=(0.8, 1.2)))
    
    heavy = AugmentationPipeline()
    heavy.add_augmentation('heavy_rotation', transforms.RandomRotation(30))
    heavy.add_augmentation('heavy_brightness_contrast', 
                           RandomBrightnessContrast(brightness_range=(0.7, 1.3), 
                                                    contrast_range=(0.7, 1.3)))
    
    return {'light config': light, 'medium config': medium, 'heavy config': heavy}

def test_augmentation_pipelines():
    configs = create_configurations() # {'light config': light, 'medium config': medium, 'heavy config': heavy}
    sample = get_random_samples(3, 1) # [(image, class_name)]
    images = []
    labels = []
    for original_img, class_name in sample:
        images.append(original_img)
        labels.append(f"Оригинал, класс {class_name}")

        for name, config in configs.items():
            aug_img = config.apply(original_img)
            images.append(aug_img)
            labels.append(name)

    show_images(images, labels, 4, 3, title="Сравнение конфигураций",
                folder=f"{PROJECT_ROOT}/plots/aug_configs_comparison",
                filename="comparison.png")

def get_sample(samples_cnt, target_size=(224, 224), only_images=False, root_path=f"{PROJECT_ROOT}/data/train"):
    cnt = 0
    images = []
    all_classes = [d for d in os.listdir(root_path) 
                  if os.path.isdir(os.path.join(root_path, d))]
    result = []
    is_enough = False

    while cnt < samples_cnt:
        for class_name in all_classes:
            class_path = os.path.join(root_path, class_name)
        
            images = [f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
            for img_name in images:
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path).convert('RGB')
                img = img.resize(target_size, Image.Resampling.LANCZOS)

                if only_images:
                    result.append(img)
                else:
                    result.append((img, class_name))
                
                cnt += 1
                if cnt >= samples_cnt:
                    is_enough = True
                    break
            if is_enough:
                break

    return result

def test_different_sizes(samples_per_size=100):
    sizes = [(64, 64), (128, 128), (224, 224), (512, 512)]
    root_path = f"{PROJECT_ROOT}/data/train"
    samples_cnt = 100

    cnt = 0
    images_paths = []
    all_classes = [d for d in os.listdir(root_path) 
                if os.path.isdir(os.path.join(root_path, d))]
    is_enough = False

    while cnt < samples_cnt:
        for class_name in all_classes:
            class_path = os.path.join(root_path, class_name)

            for f in os.listdir(class_path):
                images_paths.append(os.path.join(class_path, f))
                cnt += 1
                if cnt >= samples_cnt:
                    is_enough = True
                    break
            if is_enough:
                break
    
    load_times, aug_times = [], []
    load_memories, aug_memories = [], []
    
    for size in sizes:
        #print(f"========== SIZE: {size} ==========")
        
        gc.collect()
        tracemalloc.start()
        load_start = time.perf_counter()
        
        images = []

        for img_path in images_paths:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(size, Image.Resampling.LANCZOS)
            arr = np.array(img)
            images.append(arr)

        load_end = time.perf_counter()
        load_time = load_end - load_start
        load_times.append(f"{load_time:.2f}")

        load_current, load_peak = tracemalloc.get_traced_memory()
        load_memories.append(f"{(load_peak/1024/1024):.2f}")

        #print(f"Load time: {load_time} s")
        #print(f"Load current memory: {load_current/1024/1024} mb")
        #print(f"Load peak memory: {load_peak/1024/1024} mb")
        tracemalloc.stop()

        aug_pipeline = AugmentationPipeline()
        augmentations = {
            "random blur": RandomBlur(kernel_size_range=(3, 7), sigma_range=(0.5, 1.0)),
            "random perspective": RandomPerspective(distortion_scale=0.3, p=1),
            "gaussian noise": AddGaussianNoise(mean=0, std=0.1)
        }

        aug_pipeline.add_augmentation("to tensor", transforms.ToTensor())
        for name, aug in augmentations.items():
            aug_pipeline.add_augmentation(name, aug)

        aug_start = time.perf_counter()
        aug_images = []

        for img in images:
            augmented = aug_pipeline.apply(img)
            aug_images.append(augmented)

        aug_end = time.perf_counter()
        aug_time = aug_end - aug_start
        aug_times.append(f"{aug_time:.2f}")
        #print(f"Augment time: {aug_time} s")

        tensor_memory = 0
        for aug in aug_images:
            tensor_memory += aug.element_size() * aug.nelement()
        
        aug_memories.append(f"{(tensor_memory/1024/1024):.2f}")
        #print(f"Tensor memory: {(tensor_memory/1024/1024):.2f} MB")

        del aug_images
        del images
    
    from augmentations_basics.visualization_utils import plot_sizes_experiment
    int_sizes = [s for s, s in sizes]
    plot_sizes_experiment(int_sizes, load_times, aug_times, load_memories, aug_memories,
                          folder=f"{PROJECT_ROOT}/plots/sizes_experiment",
                          filename=f"comparison.png")

def train_model(using_augs: bool = False):
    from torchvision import models
    from torch.utils.data import DataLoader
    from augmentations_basics.datasets import CustomImageDataset

    # Подготовка датасета
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_path = f'{PROJECT_ROOT}/data/train'
    train_augs_path = f'{PROJECT_ROOT}/data/train_with_augmentations'
    if using_augs:
        if not os.path.exists(train_augs_path):
            augment_images(train_path, train_augs_path)
        train_dataset = CustomImageDataset(train_augs_path, 
                                           transform=transform)
        batch_size = 108
    else:
        train_dataset = CustomImageDataset(f'{PROJECT_ROOT}/data/train', transform=transform)
        batch_size = 36

    val_dataset = CustomImageDataset(f'{PROJECT_ROOT}/data/val', transform=transform)
    test_dataset = CustomImageDataset(f'{PROJECT_ROOT}/data/test', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=9)
    test_loader = DataLoader(test_dataset, batch_size=25)

    # Загрузка предобученной модели
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(model.classifier[1].in_features, len(train_dataset.get_class_names()))
    )

    # Обучение
    for lr in [0.001]: # 0.0001, 0.0005
        print(f"Learning rate: {lr}")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        train_losses, val_losses, test_losses = [], [], []
        train_accuracies, val_accuracies, test_accuracies = [], [], []

        for epoch in range(7):
            # Тренировка
            model.train()

            correct, total = 0, 0
            epoch_losses = []
            for x, y in train_loader:
                optimizer.zero_grad()
                out = model(x)

                _, predicted = torch.max(out.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

                loss = loss_fn(out, y)
                epoch_losses.append(loss.item())

                loss.backward()
                optimizer.step()
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            acc = correct / total
            train_losses.append(avg_loss)
            train_accuracies.append(acc)

            # Валидация
            model.eval()
            epoch_losses = []
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    out = model(x)
                    _, predicted = torch.max(out.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                    loss = loss_fn(out, y)
                    epoch_losses.append(loss.item())
            
            avg_loss = sum(epoch_losses)/len(epoch_losses)
            acc = correct / total
            val_losses.append(avg_loss)
            val_accuracies.append(acc)

            # Тест
            model.eval()
            epoch_losses = []
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    out = model(x)
                    _, predicted = torch.max(out.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                    loss = loss_fn(out, y)
                    epoch_losses.append(loss.item())
            
            avg_loss = sum(epoch_losses)/len(epoch_losses)
            acc = correct / total
            test_losses.append(avg_loss)
            test_accuracies.append(acc)

            print(f'Epoch {epoch+1} done!')
        
        from augmentations_basics.visualization_utils import plot_learning_curves
        if using_augs:
            folder = f"{PROJECT_ROOT}/plots/fit_pretrain_with_augs"
        else:
            folder = f"{PROJECT_ROOT}/plots/fit_pretrain"
        plot_learning_curves(train_losses, val_losses, test_losses, 
                             train_accuracies, val_accuracies, test_accuracies,
                             lr=lr,
                             using_augs=using_augs,
                             folder=folder,
                             filename=f"learing_curves_lr_{lr}.png")
        
def augment_images(source_folder, target_folder, augmentations_per_image=2):
    from augmentations_basics.extra_augs import Posterize, AddGaussianNoise
    augmentations = [
        transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.5, 0.8)),
        transforms.ColorJitter(brightness=0.3, saturation=0.1, hue=0.1),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomCrop(150),
        Posterize(bits=2),
        AddGaussianNoise(mean=0, std=0.01)
    ]
    from pathlib import Path
    target_path = Path(target_folder)
    target_path.mkdir(parents=True, exist_ok=True)

    source_path = Path(source_folder)

    for class_folder in source_path.iterdir():
        if class_folder.is_dir():
            # Создаем соответствующую папку класса в целевой директории
            target_class_folder = target_path / class_folder.name
            target_class_folder.mkdir(exist_ok=True)
            
            # Проходим по всем изображениям в папке класса
            for img_file in class_folder.glob('*.*'):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    continue
                
                image = Image.open(img_file).convert('RGB')
                image = image.resize((224, 224), Image.Resampling.LANCZOS)
                
                # 1. Копируем оригинальное изображение
                original_filename = f"{img_file.stem}_original{img_file.suffix}"
                original_target = target_class_folder / original_filename
                shutil.copy2(img_file, original_target)
                
                # 2. Создаем аугментированные версии
                for i in range(augmentations_per_image):
                    augs = random.sample(augmentations, k=2) # Выбираем 2 случайные аугментации
                    
                    # создаем пайплайн
                    transform = transforms.Compose(
                        [transforms.ToTensor(), *augs, transforms.ToPILImage('RGB')]) 

                    augmented = transform(image)

                    if augmented.mode != 'RGB':
                        augmented = augmented.convert('RGB')
                    
                    # Сохраняем аугментированное изображение
                    augmented_filename = f"{img_file.stem}_aug_{i+1}{img_file.suffix}"
                    augmented_path = target_class_folder / augmented_filename
                    augmented.save(augmented_path)


