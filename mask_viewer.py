#!/usr/bin/env python3
"""
SLM Mask Viewer - Приложение для просмотра 8-битных масок и их FFT
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import cv2
import os


class SLMMaskViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("SLM Mask Viewer - 8-bit & FFT")
        self.root.geometry("1200x800")
        
        # Переменные
        self.current_image = None
        self.current_image_path = None
        self.fft_window = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Верхняя панель с кнопками
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        btn_load = ttk.Button(top_frame, text="Загрузить маску", command=self.load_mask)
        btn_load.pack(side=tk.LEFT, padx=5)
        
        btn_fft = ttk.Button(top_frame, text="Показать FFT", command=self.show_fft)
        btn_fft.pack(side=tk.LEFT, padx=5)
        
        btn_save_8bit = ttk.Button(top_frame, text="Сохранить 8-bit", command=self.save_8bit)
        btn_save_8bit.pack(side=tk.LEFT, padx=5)
        
        btn_quit = ttk.Button(top_frame, text="Выход", command=self.root.quit)
        btn_quit.pack(side=tk.RIGHT, padx=5)
        
        # Информация о файле
        self.info_label = ttk.Label(top_frame, text="Нет загруженного файла")
        self.info_label.pack(side=tk.LEFT, padx=20)
        
        # Основная область для изображения
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - оригинальное изображение
        left_frame = ttk.LabelFrame(main_frame, text="8-bit Маска")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.image_label = ttk.Label(left_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Правая панель - гистограмма
        right_frame = ttk.LabelFrame(main_frame, text="Гистограмма интенсивности")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.hist_canvas = tk.Canvas(right_frame, bg='white', width=400, height=300)
        self.hist_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Статус бар
        self.status_var = tk.StringVar()
        self.status_var.set("Готов к работе")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
    def load_mask(self):
        """Загрузка маски из файла"""
        filetypes = [
            ("Изображения", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
            ("NumPy файлы", "*.npy *.npz"),
            ("Все файлы", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Выберите файл маски",
            filetypes=filetypes
        )
        
        if not filepath:
            return
            
        try:
            # Загрузка в зависимости от типа файла
            if filepath.endswith(('.npy', '.npz')):
                data = np.load(filepath)
                if isinstance(data, np.lib.npyio.NpzFile):
                    # Если это .npz файл, берем первый массив
                    data = data[list(data.keys())[0]]
                self.current_image = data.astype(np.float64)
            else:
                # Загрузка изображения
                img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError("Не удалось прочитать изображение")
                
                # Конвертация в оттенки серого если необходимо
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                self.current_image = img.astype(np.float64)
            
            self.current_image_path = filepath
            
            # Обновление информации
            filename = os.path.basename(filepath)
            shape = self.current_image.shape
            dtype = self.current_image.dtype
            min_val = np.min(self.current_image)
            max_val = np.max(self.current_image)
            
            self.info_label.config(
                text=f"{filename} | {shape[1]}x{shape[0]} | {dtype} | [{min_val:.2f}, {max_val:.2f}]"
            )
            
            # Отображение изображения
            self.display_image()
            
            # Отображение гистограммы
            self.display_histogram()
            
            self.status_var.set(f"Загружено: {filepath}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{str(e)}")
            self.status_var.set("Ошибка загрузки")
    
    def convert_to_8bit(self, image):
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
    
    def display_image(self):
        """Отображение текущего изображения в 8-битном формате"""
        if self.current_image is None:
            return
        
        # Конвертация в 8-бит
        img_8bit = self.convert_to_8bit(self.current_image)
        
        # Получение размеров области отображения
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()
        
        if label_width < 10 or label_height < 10:
            # Если размеры еще не определены, используем значения по умолчанию
            label_width = 500
            label_height = 400
        
        # Масштабирование изображения для отображения
        h, w = img_8bit.shape
        scale = min(label_width / w, label_height / h) * 0.95
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_resized = cv2.resize(img_8bit, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img_8bit
        
        # Конвертация для отображения в PIL
        img_pil = Image.fromarray(img_resized, mode='L')
        
        # Сохранение ссылки на изображение (чтобы не было собрано garbage collector)
        self.tk_image = ImageTk.PhotoImage(img_pil)
        
        self.image_label.config(image=self.tk_image)
    
    def display_histogram(self):
        """Отображение гистограммы интенсивности"""
        if self.current_image is None:
            return
        
        self.hist_canvas.delete("all")
        
        # Вычисление гистограммы
        img_8bit = self.convert_to_8bit(self.current_image)
        hist = cv2.calcHist([img_8bit], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # Нормализация для отображения
        hist_max = np.max(hist)
        if hist_max == 0:
            return
        
        canvas_width = 380
        canvas_height = 280
        
        # Отрисовка гистограммы
        points = []
        for i in range(256):
            x = i * canvas_width / 256
            y = canvas_height - (hist[i] / hist_max) * canvas_height
            points.append((x, y))
        
        # Рисуем линии гистограммы
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            self.hist_canvas.create_line(x1, y1, x2, y2, fill='blue', width=1)
        
        # Подписи
        self.hist_canvas.create_text(10, 10, anchor=tk.NW, text="0", fill='black')
        self.hist_canvas.create_text(canvas_width - 10, 10, anchor=tk.NE, text="255", fill='black')
        self.hist_canvas.create_text(canvas_width / 2, canvas_height + 20, text="Интенсивность", fill='black')
    
    def show_fft(self):
        """Отображение FFT изображения в отдельном окне"""
        if self.current_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение!")
            return
        
        # Создание или активация окна FFT
        if self.fft_window is None or not self.fft_window.winfo_exists():
            self.fft_window = tk.Toplevel(self.root)
            self.fft_window.title("FFT Visualization")
            self.fft_window.geometry("800x600")
            
            # Настройка окна FFT
            fft_frame = ttk.Frame(self.fft_window)
            fft_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Кнопки управления
            btn_frame = ttk.Frame(fft_frame)
            btn_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Button(btn_frame, text="Обновить FFT", command=self.update_fft_display).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Сохранить FFT", command=self.save_fft).pack(side=tk.LEFT, padx=5)
            
            # Выбор масштаба логарифмический/линейный
            self.fft_scale_var = tk.StringVar(value="log")
            ttk.Radiobutton(btn_frame, text="Логарифмический", variable=self.fft_scale_var, 
                          value="log", command=self.update_fft_display).pack(side=tk.LEFT, padx=5)
            ttk.Radiobutton(btn_frame, text="Линейный", variable=self.fft_scale_var, 
                          value="linear", command=self.update_fft_display).pack(side=tk.LEFT, padx=5)
            
            # Область для FFT изображения
            self.fft_label = ttk.Label(fft_frame)
            self.fft_label.pack(fill=tk.BOTH, expand=True)
            
            # Информация о FFT
            self.fft_info_label = ttk.Label(fft_frame, text="")
            self.fft_info_label.pack(pady=5)
        
        self.update_fft_display()
    
    def update_fft_display(self):
        """Обновление отображения FFT"""
        if self.current_image is None or not hasattr(self, 'fft_label'):
            return
        
        # Вычисление FFT
        fft_result = np.fft.fft2(self.current_image)
        fft_shifted = np.fft.fftshift(fft_result)
        
        # Вычисление амплитудного спектра
        magnitude = np.abs(fft_shifted)
        
        # Применение масштаба (логарифмический или линейный)
        if self.fft_scale_var.get() == "log":
            magnitude_display = np.log(magnitude + 1)
        else:
            magnitude_display = magnitude
        
        # Нормализация к 8-бит
        magnitude_8bit = self.convert_to_8bit(magnitude_display)
        
        # Получение размеров
        label_width = self.fft_label.winfo_width()
        label_height = self.fft_label.winfo_height()
        
        if label_width < 10 or label_height < 10:
            label_width = 600
            label_height = 450
        
        # Масштабирование
        h, w = magnitude_8bit.shape
        scale = min(label_width / w, label_height / h) * 0.95
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_resized = cv2.resize(magnitude_8bit, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_resized = magnitude_8bit
        
        # Конвертация для отображения
        img_pil = Image.fromarray(img_resized, mode='L')
        self.tk_fft_image = ImageTk.PhotoImage(img_pil)
        
        self.fft_label.config(image=self.tk_fft_image)
        
        # Обновление информации
        max_mag = np.max(magnitude)
        min_mag = np.min(magnitude)
        mean_mag = np.mean(magnitude)
        
        info_text = f"FFT: {w}x{h} | Max: {max_mag:.2e} | Min: {min_mag:.2e} | Mean: {mean_mag:.2e}"
        self.fft_info_label.config(text=info_text)
    
    def save_8bit(self):
        """Сохранение текущей маски в 8-битном формате"""
        if self.current_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение!")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Сохранить 8-битную маску",
            defaultextension=".png",
            filetypes=[("PNG файлы", "*.png"), ("JPEG файлы", "*.jpg"), ("BMP файлы", "*.bmp")]
        )
        
        if not filepath:
            return
        
        try:
            img_8bit = self.convert_to_8bit(self.current_image)
            img_pil = Image.fromarray(img_8bit, mode='L')
            img_pil.save(filepath)
            
            self.status_var.set(f"Сохранено: {filepath}")
            messagebox.showinfo("Успех", f"Маска сохранена в:\n{filepath}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")
    
    def save_fft(self):
        """Сохранение FFT изображения"""
        if self.current_image is None:
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Сохранить FFT изображение",
            defaultextension=".png",
            filetypes=[("PNG файлы", "*.png"), ("JPEG файлы", "*.jpg"), ("NumPy файлы", "*.npy")]
        )
        
        if not filepath:
            return
        
        try:
            # Вычисление FFT
            fft_result = np.fft.fft2(self.current_image)
            fft_shifted = np.fft.fftshift(fft_result)
            magnitude = np.abs(fft_shifted)
            
            if filepath.endswith('.npy'):
                np.save(filepath, magnitude)
            else:
                if self.fft_scale_var.get() == "log":
                    magnitude_display = np.log(magnitude + 1)
                else:
                    magnitude_display = magnitude
                
                img_8bit = self.convert_to_8bit(magnitude_display)
                img_pil = Image.fromarray(img_8bit, mode='L')
                img_pil.save(filepath)
            
            self.status_var.set(f"FFT сохранен: {filepath}")
            messagebox.showinfo("Успех", f"FFT сохранен в:\n{filepath}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить FFT:\n{str(e)}")


def main():
    root = tk.Tk()
    app = SLMMaskViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
