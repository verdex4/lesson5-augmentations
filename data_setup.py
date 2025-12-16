import shutil
import os
from augmentations_basics.experiment_utils import split_train_val, PROJECT_ROOT

# Пути к папкам для удаления
paths_to_remove = [
    'data/train/labels',
    'data/train/images', 
    'data/val/labels',
    'data/val/images',
    'data/test/labels',
    'data/test/images'
]

for path in paths_to_remove:
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Удалено: {path}")
    else:
        print(f"Папка не существует: {path}")

# разделение на train/val
split_train_val(train_path=f"{PROJECT_ROOT}/data/train",
                val_path=f"{PROJECT_ROOT}/data/val",
                val_ratio=0.2)  # 20% в валидацию