import numpy as np

def image_to_numpy(img):
    """Преобразует изображение в numpy-массив с размерностью (Высота, Ширина, Каналы)"""
    img = np.array(img)
    shape = img.shape

    if len(shape) == 2: # 2D изображение
        return img[:, :, np.newaxis]

    if len(shape) == 3: # 3D изображение
        # Определяем позицию канала по минимальной размерности
        # (обычно каналов меньше, чем высоты/ширины)
        min_dim = np.argmin(shape)
        
        # Если каналы на первом месте (0) -> (H, W, C)
        if min_dim == 0 and shape[0] in [1, 3, 4]:  # 1, 3 или 4 канала
            return img.transpose(1, 2, 0)
        
        # Если каналы на втором месте (1) -> (H, W, C)
        elif min_dim == 1 and shape[1] in [1, 3, 4]:
            return img.transpose(0, 2, 1)
        
        # Если каналы уже на последнем месте (2) -> оставляем как есть
        elif min_dim == 2 or shape[2] in [1, 3, 4]:
            return img
    
    return img

def normalize_image(img):
    """Нормализует numpy изображение с проверкой уже нормализованного изображения"""
    if img.max() > 1.0:
        img = (img - img.min()) / (img.max() - img.min())
    else:
        img = img.astype(np.float32)
    return img