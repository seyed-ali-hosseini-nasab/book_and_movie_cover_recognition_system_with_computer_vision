import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os

class ScrollableFrame(ttk.Frame):
    """فریم قابل اسکرول عمودی"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # ایجاد Canvas و Scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        # پیکربندی اسکرول
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # چیدمان
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # پشتیبانی از اسکرول موس
        self.bind_mousewheel()

    def bind_mousewheel(self):
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def bind_to_mousewheel(event):
            self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def unbind_from_mousewheel(event):
            self.canvas.unbind_all("<MouseWheel>")

        self.canvas.bind('<Enter>', bind_to_mousewheel)
        self.canvas.bind('<Leave>', unbind_from_mousewheel)


class ImageSelector(ttk.Frame):
    """کادری برای انتخاب فایل یا پوشه تصاویر"""

    def __init__(self, parent, title, default_path="", is_folder=False):
        super().__init__(parent)
        self.title = title
        self.is_folder = is_folder
        self.default_path = default_path
        self.selected_paths = []
        self.create_widgets()

        # بارگذاری پیش‌فرض اگر موجود باشد
        if default_path and os.path.exists(default_path):
            self.load_default_path()

    def create_widgets(self):
        # عنوان
        title_frame = ttk.Frame(self)
        title_frame.pack(fill=tk.X, pady=5)
        ttk.Label(title_frame, text=self.title, font=("Arial", 12, "bold")).pack(side=tk.RIGHT)

        # دکمه‌ها
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="انتخاب از فایل‌ها", command=self.select_files).pack(side=tk.RIGHT, padx=5)
        if self.default_path:
            ttk.Button(button_frame, text="استفاده از پیش‌فرض", command=self.load_default_path).pack(side=tk.RIGHT, padx=5)

        # لیست فایل‌های انتخاب شده
        self.listbox = tk.Listbox(self, width=50, height=3, font=("Tahoma", 9))
        self.listbox.pack(pady=5, fill=tk.BOTH, expand=True)

    def load_default_path(self):
        """بارگذاری تمام تصاویر از مسیر پیش‌فرض"""
        if self.is_folder and os.path.isdir(self.default_path):
            self.selected_paths = [
                os.path.join(self.default_path, f)
                for f in os.listdir(self.default_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
        elif not self.is_folder and os.path.isfile(self.default_path):
            self.selected_paths = [self.default_path]
        self.update_listbox()

    def select_files(self):
        """انتخاب دستی فایل‌ها یا پوشه"""
        if self.is_folder:
            folder = filedialog.askdirectory(initialdir=self.default_path or ".")
            if folder:
                self.selected_paths = [
                    os.path.join(folder, f)
                    for f in os.listdir(folder)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
        else:
            files = filedialog.askopenfilenames(
                initialdir=self.default_path or ".",
                filetypes=[("Image files", "*.jpg *.jpeg *.png")]
            )
            self.selected_paths = list(files)
        self.update_listbox()

    def update_listbox(self):
        """به‌روزرسانی لیست‌باکس با نام فایل‌های انتخاب شده"""
        self.listbox.delete(0, tk.END)
        for path in self.selected_paths:
            self.listbox.insert(tk.END, os.path.basename(path))


class ProgressDialog:
    """دیالوگ نمایش پیشرفت پردازش"""

    def __init__(self, parent, title="در حال پردازش..."):
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("400x150")
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.grab_set()

        # مرکز کردن پنجره
        self.window.geometry("+{}+{}".format(
            parent.winfo_rootx() + 50,
            parent.winfo_rooty() + 50
        ))
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.status_label = ttk.Label(main_frame, text="آماده‌سازی...", font=("Tahoma", 10))
        self.status_label.pack(pady=10)

        self.progress = ttk.Progressbar(main_frame, mode='determinate', length=300)
        self.progress.pack(pady=10)

        self.percent_label = ttk.Label(main_frame, text="0%", font=("Tahoma", 9))
        self.percent_label.pack(pady=5)

    def update_progress(self, current, total, status="در حال پردازش..."):
        """به‌روزرسانی نوار پیشرفت و وضعیت"""
        if total > 0:
            percent = (current / total) * 100
            self.progress['value'] = percent
            self.percent_label.config(text=f"{percent:.1f}%")
            self.status_label.config(text=f"{status} ({current}/{total})")
        self.window.update()

    def close(self):
        """بستن دیالوگ"""
        self.window.destroy()


class ResultsDisplay(ttk.Frame):
    """نمایش نتایج تشخیص با امکان اسکرول عمودی"""

    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):
        header_frame = ttk.Frame(self)
        header_frame.pack(fill=tk.X, pady=10)
        ttk.Label(header_frame, text="نتایج تشخیص", font=("Arial", 14, "bold")).pack(side=tk.RIGHT)

        self.scroll_frame = ScrollableFrame(self)
        self.scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.results_container = self.scroll_frame.scrollable_frame

    def clear_results(self):
        """پاک کردن نتایج قبلی"""
        for widget in self.results_container.winfo_children():
            widget.destroy()

    def add_result(self, input_path, result_path, confidence, matches, rank, error_message=None):
        """افزودن یک نتیجه جدید به لیست نتایج"""
        result_frame = ttk.LabelFrame(
            self.results_container,
            text=f"رتبه {rank} – امتیاز: {confidence:.1f}",
            padding=10
        )
        result_frame.pack(fill=tk.X, pady=5, padx=5)

        # بخش تصاویر
        images_frame = ttk.Frame(result_frame)
        images_frame.pack(fill=tk.X, pady=5)

        # تصویر ورودی
        input_frame = ttk.Frame(images_frame)
        input_frame.pack(side=tk.RIGHT, padx=10)
        ttk.Label(input_frame, text="تصویر ورودی", font=("Tahoma", 10, "bold")).pack()
        self.add_image(input_frame, input_path)

        # تصویر تطبیق یافته
        result_img_frame = ttk.Frame(images_frame)
        result_img_frame.pack(side=tk.RIGHT, padx=10)
        ttk.Label(result_img_frame, text="تطبیق یافته", font=("Tahoma", 10, "bold")).pack()
        self.add_image(result_img_frame, result_path)

        # بخش اطلاعات
        info_frame = ttk.Frame(result_frame)
        info_frame.pack(fill=tk.X, pady=5)
        ttk.Label(info_frame, text=f"نام فایل: {os.path.basename(result_path)}", font=("Tahoma", 9)).pack(side=tk.RIGHT, padx=10)
        ttk.Label(info_frame, text=f"تعداد تطبیق‌ها: {matches}", font=("Tahoma", 9)).pack(side=tk.RIGHT, padx=10)

        # نمایش پیام خطا در صورت وجود
        if error_message:
            ttk.Label(
                result_frame,
                text=f"خطا: {error_message}",
                foreground="red",
                font=("Tahoma", 9, "italic")
            ).pack(pady=5)

    def add_image(self, parent, image_path, size=(150, 150)):
        """بارگذاری و نمایش تصویر در اندازه‌ی کوچک‌شده"""
        try:
            image = Image.open(image_path)
            image.thumbnail(size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            label = ttk.Label(parent, image=photo)
            label.image = photo  # نگه‌داشتن ارجاع
            label.pack()
        except Exception:
            ttk.Label(parent, text="خطا در بارگذاری تصویر").pack()
