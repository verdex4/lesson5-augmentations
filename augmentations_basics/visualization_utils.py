import os
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from augmentations_basics.image_utils import image_to_numpy, normalize_image

def save_plot(folder, filename):
    """Сохраняет текущую активированную фигуру matplotlib"""
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')

def show_images(images, labels=None, nrow=8, ncol=1, title=None, size=128,
                folder=None, filename=None):
    """Визуализирует батч изображений."""
    # Увеличиваем изображения до 128x128 для лучшей видимости
    resize_transform = transforms.Resize((size, size), antialias=True)
    images_resized = [resize_transform(img) for img in images]
    
    # Создаем сетку изображений
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow*2, ncol*2))
    if nrow == 1 and ncol == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()

    total_images = min(len(images_resized), len(axes_flat))
    
    for idx in range(total_images):
        img_np = image_to_numpy(images_resized[idx])
        # Нормализуем для отображения
        img_np = normalize_image(img_np)
        img_np = np.clip(img_np, 0, 1)
        axes_flat[idx].imshow(img_np)
        axes_flat[idx].axis('off')
        if labels is not None and idx < len(labels):
            axes_flat[idx].set_title(f'{labels[idx]}', fontsize=10)
    
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if folder and filename:
        save_plot(folder, filename)
    plt.show()

def show_single_augmentation(original_img, augmented_img, title="Аугментация",
                             folder=None, filename=None):
    """Визуализирует оригинальное и аугментированное изображение рядом."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_img)
    aug_resized = resize_transform(augmented_img)
    
    # Оригинальное изображение
    orig_np = image_to_numpy(orig_resized)
    #print(f"Оригинал перед нормализацией: форма {orig_np.shape}, диапазон [{orig_np.min():.6f}, {orig_np.max():.6f}]")
    orig_np = normalize_image(orig_np)
    #print(f"Оригинал после нормализации: форма {orig_np.shape}, диапазон [{orig_np.min():.6f}, {orig_np.max():.6f}]")
    orig_np = np.clip(orig_np, 0, 1)

    ax1.imshow(orig_np)
    ax1.set_title("Оригинал")
    ax1.axis('off')
    
    # Аугментированное изображение
    aug_np = image_to_numpy(aug_resized)
    #print(f"Аугмент перед нормализацией:  форма {aug_np.shape}, диапазон [{aug_np.min():.6f}, {aug_np.max():.6f}]")
    aug_np = normalize_image(aug_np)
    #print(f"Аугмент после нормализации:  форма {aug_np.shape}, диапазон [{aug_np.min():.6f}, {aug_np.max():.6f}]")
    aug_np = np.clip(aug_np, 0, 1)

    ax2.imshow(aug_np)
    ax2.set_title(title)
    ax2.axis('off')
    
    plt.tight_layout()
    if folder and filename:
        save_plot(folder, filename)
    plt.show()

def show_multiple_augmentations(original_img, augmented_imgs, titles,
                                folder=None, filename=None):
    """Визуализирует оригинальное изображение и несколько аугментаций."""
    n_augs = len(augmented_imgs)
    fig, axes = plt.subplots(1, n_augs + 1, figsize=((n_augs + 1) * 2, 2))
    
    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_img)
    
    # Оригинальное изображение
    orig_np = image_to_numpy(orig_resized)
    orig_np = normalize_image(orig_np)
    orig_np = np.clip(orig_np, 0, 1)
    axes[0].imshow(orig_np)
    axes[0].set_title("Оригинал")
    axes[0].axis('off')
    
    # Аугментированные изображения
    for i, (aug_img, title) in enumerate(zip(augmented_imgs, titles)):
        aug_resized = resize_transform(aug_img)
        aug_np = image_to_numpy(aug_resized)
        aug_np = normalize_image(aug_np)
        aug_np = np.clip(aug_np, 0, 1)
        axes[i + 1].imshow(aug_np)
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    if folder and filename:
        save_plot(folder, filename)
    plt.show()

def plot_size_stats(min_size, max_size, widths, heights, class_name,
                    folder=None, filename=None):
    """Выводит статистику по размерам изображений и визуализирует распределение"""
    avg_size = (round(sum(widths)/len(widths)), round(sum(heights)/len(heights)))

    print(f"========== Класс {class_name} ==========")
    print(f"Минимальная ширина: {min(widths)}")
    print(f"Максимальная ширина: {max(widths)}")
    print(f"Минимальная высота: {min(heights)}")
    print(f"Максимальная высота: {max(heights)}")
    print(f"Минимальный размер: {min_size}")
    print(f"Максимальный размер: {max_size}")
    print(f"Средний размер: {avg_size}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].hist(widths, bins=30)
    axes[0, 0].set_xlabel("Ширина")
    axes[0, 0].set_ylabel("Частота")
    axes[0, 0].set_title(f"Класс {class_name}: Гистограмма ширин")

    axes[0, 1].hist(heights, bins=30)
    axes[0, 1].set_xlabel("Высота")
    axes[0, 1].set_ylabel("Частота")
    axes[0, 1].set_title(f"Класс {class_name}: Гистограмма высот")

    axes[1, 0].scatter(widths, heights)
    axes[1, 0].set_xlabel("Ширина")
    axes[1, 0].set_ylabel("Высота")
    axes[1, 0].set_title(f"Класс {class_name}: Распределение размеров")

    axes[1, 1].boxplot([widths, heights], labels=['Ширины', 'Высоты'], showmeans=True)
    axes[1, 1].set_title(f"Класс {class_name}: Boxplot")

    plt.tight_layout()
    if folder and filename:
        save_plot(folder, filename)
    plt.show()

def plot_dataset_sizes(all_widths, all_heights, classes, images_count,
                       folder=None, filename=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(classes, images_count, 'o', linestyle="--")
    axes[0].set_xlabel("Герои")
    axes[0].set_ylabel("Количество")
    axes[0].set_title("Количество изображений в классах")

    axes[1].boxplot([all_widths, all_heights], labels=['Ширины', 'Высоты'], showmeans=True)
    axes[1].set_title("Датасет: Boxplot")
    
    plt.tight_layout()
    if folder and filename:
        save_plot(folder, filename)
    plt.show()

def plot_sizes_experiment(sizes, load_times, aug_times, load_memories, aug_memories,
                          folder=None, filename=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(sizes, load_times, 'o', linestyle='-')
    axes[0, 0].set_xlabel("Размер")
    axes[0, 0].set_ylabel("Время, с")
    axes[0, 0].set_xticks(sizes)
    axes[0, 0].set_yticks(load_times)
    axes[0, 0].set_title("Время загрузки")

    axes[0, 1].plot(sizes, aug_times, 'o', linestyle='-')
    axes[0, 1].set_xlabel("Размер")
    axes[0, 1].set_ylabel("Время, с")
    axes[0, 1].set_xticks(sizes)
    axes[0, 1].set_yticks(aug_times)
    axes[0, 1].set_title("Время аугментаций")

    axes[1, 0].plot(sizes, load_memories, 'o', linestyle='-')
    axes[1, 0].set_xlabel("Размер")
    axes[1, 0].set_ylabel("Память, Мб")
    axes[1, 0].set_xticks(sizes)
    axes[1, 0].set_yticks(load_memories)
    axes[1, 0].set_title("Потребление памяти при загрузке")

    axes[1, 1].plot(sizes, aug_memories, 'o', linestyle='-')
    axes[1, 1].set_xlabel("Размер")
    axes[1, 1].set_ylabel("Память, Мб")
    axes[1, 1].set_xticks(sizes)
    axes[1, 1].set_yticks(aug_memories)
    axes[1, 1].set_title("Потребление памяти при аугментациях")

    fig.suptitle("Сравнение времени и памяти в зависимости от размера изображений")

    plt.tight_layout()
    if folder and filename:
        save_plot(folder, filename)
    plt.show()

def plot_learning_curves(train_losses, val_losses, test_losses, 
                         train_accuracies, val_accuracies, test_accuracies,
                         lr, using_augs,
                         folder=None, filename=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = [i for i in range(1, len(train_losses)+1)]

    axes[0].plot(epochs, train_losses, 'o', linestyle='-', label='Train Loss', color='blue')
    axes[0].plot(epochs, val_losses, 'o', linestyle='-', label='Validation Loss', color='orange')
    axes[0].plot(epochs, test_losses, 'o', linestyle='-', label='Test Loss', color='red')
    axes[0].set_xlabel("Эпохи")
    axes[0].set_ylabel("Потери")
    axes[0].set_xticks(epochs)
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_accuracies, 'o', linestyle='-', label='Train Acc', color='blue')
    axes[1].plot(epochs, val_accuracies, 'o', linestyle='-', label='Validation Acc', color='orange')
    axes[1].plot(epochs, test_accuracies, 'o', linestyle='-', label='Test Acc', color='red')
    axes[1].set_xlabel("Эпохи")
    axes[1].set_ylabel("Точность")
    axes[1].set_xticks(epochs)
    axes[1].set_title("Accuracy")
    axes[1].legend()

    if using_augs:
        fig.suptitle(f"Кривые обучения, dataset: aug, lr: {lr}")
    else:
        fig.suptitle(f"Кривые обучения, dataset: simple, lr: {lr}")

    plt.tight_layout()
    if folder and filename:
        save_plot(folder, filename)
    plt.show()
