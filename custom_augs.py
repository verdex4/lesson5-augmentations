import torch
from torchvision.transforms import functional as F
import random

class RandomBlur:
    """Случайное размытие гауссовым фильтром"""
    def __init__(self, kernel_size_range=(3, 7), sigma_range=(0.1, 2.0)):
        self.kernel_size_range = kernel_size_range
        self.sigma_range = sigma_range
    
    def __call__(self, img_tensor):
        """
        img_tensor: Tensor формы [C, H, W]
        """
        # Выбираем случайный размер ядра (только нечётные числа)
        kernel_size = random.choice(
            [k for k in range(self.kernel_size_range[0], 
                            self.kernel_size_range[1] + 1) 
             if k % 2 == 1]
        )
        
        # Случайное sigma
        sigma = random.uniform(*self.sigma_range)
        
        # Применяем гауссово размытие
        blurred = F.gaussian_blur(
            img_tensor, 
            kernel_size=[kernel_size, kernel_size],
            sigma=[sigma, sigma]
        )
        
        return blurred

class RandomPerspective:
    """Случайное перспективное искажение"""
    def __init__(self, distortion_scale=0.5, p=0.5):
        self.distortion_scale = distortion_scale
        self.p = p
    
    def __call__(self, img_tensor):
        if random.random() > self.p:
            return img_tensor
        
        # Генерируем случайные точки для перспективного преобразования
        _, height, width = img_tensor.shape
        
        # Начальные точки (углы изображения)
        startpoints = [
            [0, 0],                     # Верхний левый
            [width - 1, 0],             # Верхний правый
            [width - 1, height - 1],    # Нижний правый
            [0, height - 1]             # Нижний левый
        ]
        
        # Случайное смещение для каждой точки
        max_distortion = int(min(height, width) * self.distortion_scale)
        
        endpoints = []
        for x, y in startpoints:
            dx = random.randint(-max_distortion, max_distortion)
            dy = random.randint(-max_distortion, max_distortion)
            endpoints.append([x + dx, y + dy])
        
        # Применяем перспективное преобразование
        perspective_img = F.perspective(
            img_tensor,
            startpoints,
            endpoints
        )
        
        return perspective_img

class RandomBrightnessContrast:
    """Случайное изменение яркости и контрастности"""
    def __init__(self, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def __call__(self, img_tensor):
        # Случайные коэффициенты
        brightness_factor = random.uniform(*self.brightness_range)
        contrast_factor = random.uniform(*self.contrast_range)
        
        # Применяем яркость
        if brightness_factor != 1.0:
            img_tensor = F.adjust_brightness(
                img_tensor, brightness_factor
            )
        
        # Применяем контраст
        if contrast_factor != 1.0:
            img_tensor = F.adjust_contrast(
                img_tensor, contrast_factor
            )
        
        return img_tensor