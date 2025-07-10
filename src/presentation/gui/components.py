import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)

        # Frame inside the canvas
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Put the inner frame into the canvas
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack scrollbar first to take its own width
        self.scrollbar.pack(side="right", fill="y")
        # Then pack canvas to fill the rest
        self.canvas.pack(side="left", fill="both", expand=True)

        # Bind mouse wheel scrolling
        self._bind_mousewheel()

    def _bind_mousewheel(self):
        def _on_mousewheel(event):
            # Windows and MacOS: delta is event.delta
            # Linux: bind to Button-4 and Button-5 instead
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Bind enter to capture wheel events
        self.canvas.bind('<Enter>', lambda e: self.canvas.bind_all("<MouseWheel>", _on_mousewheel))


class ImageSelector(ttk.Frame):
    def __init__(self, parent, title, default_path="", is_folder=False):
        super().__init__(parent)
        self.title = title
        self.is_folder = is_folder
        self.default_path = default_path
        self.selected_paths = []
        self.create_widgets()

        # Default loading if available
        if default_path and os.path.exists(default_path):
            self.load_default_path()

    def create_widgets(self):
        # Title
        title_frame = ttk.Frame(self)
        title_frame.pack(fill=tk.X, pady=5)
        ttk.Label(title_frame, text=self.title, font=("Arial", 12, "bold")).pack(side=tk.RIGHT)

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="انتخاب از فایل‌ها", command=self.select_files).pack(side=tk.RIGHT, padx=5)
        if self.default_path:
            ttk.Button(button_frame, text="استفاده از پیش‌فرض", command=self.load_default_path).pack(side=tk.RIGHT,
                                                                                                     padx=5)

        # List of selected files
        self.listbox = tk.Listbox(self, width=50, height=3, font=("Tahoma", 9))
        self.listbox.pack(pady=5, fill=tk.BOTH, expand=True)

    def load_default_path(self):
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
        self.listbox.delete(0, tk.END)
        for path in self.selected_paths:
            self.listbox.insert(tk.END, os.path.basename(path))


class ProgressDialog:
    def __init__(self, parent, title="در حال پردازش..."):
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("400x150")
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.grab_set()

        # Center the window
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
        if total > 0:
            percent = (current / total) * 100
            self.progress['value'] = percent
            self.percent_label.config(text=f"{percent:.1f}%")
            self.status_label.config(text=f"{status} ({current}/{total})")
        self.window.update()

    def close(self):
        self.window.destroy()


class ResultsDisplay(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):
        header_frame = ttk.Frame(self)
        header_frame.pack(fill=tk.X, pady=10)
        ttk.Label(header_frame, text="نتایج تشخیص", font=("Arial", 14, "bold")).pack(side=tk.RIGHT)

        self.results_container = ttk.Frame(self)
        self.results_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def clear_results(self):
        for widget in self.results_container.winfo_children():
            widget.destroy()

    def add_result(
            self,
            input_path,
            result_path,
            confidence,
            matches,
            rank,
            error_message=None,
            source_frame_path=None,
            overlay_image_path=None
    ):
        """
        Add a result entry; show input image, matched cover,
        and the overlay image if available.
        """
        result_frame = ttk.LabelFrame(
            self.results_container,
            text=f"رتبه {rank} – امتیاز: {confidence:.1f}",
            padding=10
        )
        result_frame.pack(fill=tk.X, pady=5, padx=5)

        images_frame = ttk.Frame(result_frame)
        images_frame.pack(fill=tk.X, pady=5)

        # 1. Original input image or video frame
        input_frame = ttk.Frame(images_frame)
        input_frame.pack(side=tk.RIGHT, padx=10)
        if source_frame_path:
            # for video: show the matched frame
            ttk.Label(input_frame, text="فریم منطبق", font=("Tahoma", 10, "bold")).pack()
            self.add_image(input_frame, source_frame_path)
        else:
            # for image: show the input image
            ttk.Label(input_frame, text="تصویر ورودی", font=("Tahoma", 10, "bold")).pack()
            self.add_image(input_frame, input_path)

        # 2. Matched cover image
        cover_frame = ttk.Frame(images_frame)
        cover_frame.pack(side=tk.RIGHT, padx=10)
        ttk.Label(cover_frame, text="جلد منطبق", font=("Tahoma", 10, "bold")).pack()
        self.add_image(cover_frame, result_path)

        # 3. Overlay image if exists
        if overlay_image_path and os.path.exists(overlay_image_path):
            overlay_frame = ttk.Frame(images_frame)
            overlay_frame.pack(side=tk.RIGHT, padx=10)
            ttk.Label(overlay_frame, text="نتیجه همپوشانی", font=("Tahoma", 10, "bold")).pack()
            self.add_image(overlay_frame, overlay_image_path)

        # Info section
        info_frame = ttk.Frame(result_frame)
        info_frame.pack(fill=tk.X, pady=5)
        ttk.Label(info_frame, text=f"تعداد تطبیق‌ها: {matches}", font=("Tahoma", 9)).pack(side=tk.RIGHT, padx=5)
        if error_message:
            ttk.Label(result_frame, text=f"خطا: {error_message}", foreground="red").pack(pady=5)

    def add_image(self, parent, image_path, size=(300, 300)):
        """Loading and displaying the image in reduced size"""
        try:
            image = Image.open(image_path)
            image.thumbnail(size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            label = ttk.Label(parent, image=photo)
            label.image = photo  # Keeping the referral
            label.pack()
        except Exception:
            ttk.Label(parent, text="خطا در بارگذاری تصویر").pack()


class VideoSelector(ImageSelector):
    def __init__(self, parent, title, default_path=""):
        super().__init__(parent, title, default_path, is_folder=False)

    def select_files(self):
        video, = filedialog.askopenfilename(
            initialdir=self.default_path or ".",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")],
        ),
        self.selected_paths = [video] if video else []
        self.update_listbox()
