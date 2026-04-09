#!/usr/bin/env python3
"""
SLM Mask Viewer - CLI версия для проверки функциональности без GUI
"""

import numpy as np
from PIL import Image
import cv2
import os


def convert_to_8bit(image):
    """Конвертация изображения в 8-битный формат"""
    if image is None:
        return None
    
    min_val = np.min(image)
    max_val = np.max(image)
    
    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8)
    
    # Нормализация к диапазону 0-255
    normalized = (image - min_val) / (max_val - min_val) * 255.0
    return normalized.astype(np.uint8)


def compute_fft(image, log_scale=True):
    """Вычисление FFT изображения"""
    # Вычисление FFT
    fft_result = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_result)
    
    # Вычисление амплитудного спектра
    magnitude = np.abs(fft_shifted)
    
    # Применение масштаба (логарифмический или линейный)
    if log_scale:
        magnitude_display = np.log(magnitude + 1)
    else:
        magnitude_display = magnitude
    
    return magnitude, magnitude_display


def load_mask(filepath):
    """Загрузка маски из файла"""
    if filepath.endswith(('.npy', '.npz')):
        data = np.load(filepath)
        if isinstance(data, np.lib.npyio.NpzFile):
            data = data[list(data.keys())[0]]
        return data.astype(np.float64)
    else:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Не удалось прочитать изображение")
        
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return img.astype(np.float64)


def main():
    print("=" * 60)
    print("SLM Mask Viewer - Тестирование функциональности")
    print("=" * 60)
    
    # Загрузка тестовой маски
    test_file = 'test_mask.png'
    if not os.path.exists(test_file):
        print(f"\n✗ Тестовый файл {test_file} не найден!")
        print("Создайте тестовую маску сначала.")
        return
    
    print(f"\n✓ Загрузка маски: {test_file}")
    mask = load_mask(test_file)
    print(f"  Размер: {mask.shape[1]}x{mask.shape[0]}")
    print(f"  Тип данных: {mask.dtype}")
    print(f"  Диапазон: [{np.min(mask):.2f}, {np.max(mask):.2f}]")
    
    # Конвертация в 8-бит
    print("\n✓ Конвертация в 8-битный формат")
    mask_8bit = convert_to_8bit(mask)
    print(f"  Результат: {mask_8bit.shape}, dtype={mask_8bit.dtype}")
    print(f"  Диапазон: [{np.min(mask_8bit)}, {np.max(mask_8bit)}]")
    
    # Вычисление FFT
    print("\n✓ Вычисление FFT")
    magnitude, magnitude_log = compute_fft(mask, log_scale=True)
    print(f"  FFT размер: {magnitude.shape}")
    print(f"  Max magnitude: {np.max(magnitude):.2e}")
    print(f"  Min magnitude: {np.min(magnitude):.2e}")
    print(f"  Mean magnitude: {np.mean(magnitude):.2e}")
    
    # Сохранение результатов
    print("\n✓ Сохранение результатов")
    
    # Сохранение 8-битной версии
    img_8bit_pil = Image.fromarray(mask_8bit)
    img_8bit_pil.save('output_8bit.png')
    print(f"  8-bit маска сохранена: output_8bit.png")
    
    # Сохранение FFT
    fft_8bit = convert_to_8bit(magnitude_log)
    fft_pil = Image.fromarray(fft_8bit)
    fft_pil.save('output_fft.png')
    print(f"  FFT изображение сохранено: output_fft.png")
    
    # Сохранение FFT данных
    np.save('output_fft_data.npy', magnitude)
    print(f"  FFT данные сохранены: output_fft_data.npy")
    
    print("\n" + "=" * 60)
    print("Все тесты успешно пройдены!")
    print("=" * 60)
    print("\nДля запуска GUI версии выполните:")
    print("  python mask_viewer.py")
    print("\nПримечание: Для работы GUI требуется установленная библиотека tkinter")


if __name__ == "__main__":
    main()
