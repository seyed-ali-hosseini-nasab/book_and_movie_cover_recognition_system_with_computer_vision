import tkinter as tk
from tkinter import ttk, messagebox
from .components import (
    ScrollableFrame,
    ImageSelector,
    VideoSelector,
    ProgressDialog,
    ResultsDisplay
)
from src.application.use_cases.find_matching_book_movie import FindMatchingBookMovieUseCase
from src.application.use_cases.process_video import ProcessVideoUseCase
from src.infrastructure.feature_extractors.sift_extractor import SIFTExtractor
from src.infrastructure.matchers.flann_matcher import FLANNMatcher
from src.infrastructure.repositories.file_image_repository import FileImageRepository
from src.infrastructure.repositories.file_video_repository import FileVideoRepository
from src.infrastructure.video_processors.frame_extractor import FrameExtractor
from ttkthemes import ThemedTk
import os
import threading


class BookCoverRecognitionApp(ThemedTk):
    def __init__(self):
        super().__init__(theme="arc")
        self.title("سیستم تشخیص جلد کتاب و ویدئو")
        self.minsize(1080, 720)
        self.geometry("1080x720")
        self.option_add("*Font", "Tahoma 10")

        self.setup_use_cases()
        self.create_widgets()

    def setup_use_cases(self):
        self.feature_extractor = SIFTExtractor()
        self.matcher = FLANNMatcher()
        self.image_repository = FileImageRepository()
        self.video_repository = FileVideoRepository()
        self.book_movie_use_case = FindMatchingBookMovieUseCase(
            feature_extractor=self.feature_extractor,
            matcher=self.matcher,
            image_repository=self.image_repository
        )
        self.video_use_case = None  # will initialize on demand

    def create_widgets(self):
        # root scrollable frame
        root_scroll = ScrollableFrame(self)
        root_scroll.pack(fill=tk.BOTH, expand=True)
        container = root_scroll.scrollable_frame
        container.columnconfigure(0, weight=1, minsize=1080)

        # settings panel
        settings = ttk.LabelFrame(container, text="تنظیمات ورودی", padding=10)
        settings.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        settings.columnconfigure(0, weight=1)

        # input image selector
        self.input_selector = ImageSelector(
            settings,
            "تصویر ورودی",
            default_path="data/input_images/Return.jpg"
        )
        self.input_selector.pack(fill=tk.X, pady=5)

        # book/movie images selector
        self.book_selector = ImageSelector(
            settings,
            "پوشه تصاویر کتاب/فیلم",
            default_path="data/book_movie_images",
            is_folder=True
        )
        self.book_selector.pack(fill=tk.X, pady=5)

        # video selector
        self.video_selector = VideoSelector(
            settings,
            "ویدئوی تریلر",
            default_path="data/trailers"
        )
        self.video_selector.pack(fill=tk.X, pady=5)

        # param controls
        self.frame_skip_var = tk.IntVar(value=30)
        self.phash_var = tk.IntVar(value=20)
        self.hist_var = tk.DoubleVar(value=0.3)

        self._build_spinbox(
            settings,
            "Frame Skip:",
            self.frame_skip_var,
            mn=1, mx=120, step=1,
            hint="هر چه عدد بیشتر باشد، فریم‌های کمتری پردازش می‌شوند (سرعت↑، دقت↓)"
        )

        self._build_spinbox(
            settings,
            "pHash Thresh:",
            self.phash_var,
            mn=1, mx=64, step=1,
            hint="ملاک فاصله هامینگ pHash (کمتر→فیلتر سخت‌تر)"
        )

        self._build_spinbox(
            settings,
            "Hist Thresh:",
            self.hist_var,
            mn=0.0, mx=1.0, step=0.05,
            hint="آستانه تغییر هیستوگرام HSV (کمتر→انتخاب فریم‌های بیشتر)"
        )

        # action buttons
        actions = ttk.Frame(container)
        actions.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        actions.columnconfigure(0, weight=1)

        self.process_img_btn = ttk.Button(
            actions, text="تشخیص تصویر",
            command=self.start_image_processing,
            style="Accent.TButton"
        )
        self.process_img_btn.pack(side=tk.RIGHT, padx=5)

        self.process_vid_btn = ttk.Button(
            actions, text="تشخیص ویدئو",
            command=self.start_video_processing,
            style="Accent.TButton"
        )
        self.process_vid_btn.pack(side=tk.RIGHT, padx=5)

        ttk.Button(actions, text="پاک کردن نتایج", command=self.clear_results).pack(side=tk.RIGHT, padx=5)

        ttk.Button(
            actions,
            text="پاک کردن کش",
            command=self.clear_frame_cache
        ).pack(side=tk.RIGHT, padx=5)

        # results display
        self.result_display = ResultsDisplay(container)
        self.result_display.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        container.rowconfigure(2, weight=1)

    def _build_spinbox(self, parent, text, var, mn, mx, step, hint):
        """
        Build a Spinbox that automatically clamps its value
        to the [mn, mx] interval and ignores invalid input.
        """
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.X, pady=5)
        ttk.Label(frm, text=text).pack(side=tk.RIGHT)

        # register validation callback
        def _validator(value_if_edit):
            # empty string → allow temporarily
            if value_if_edit == "":
                return True
            try:
                v = float(value_if_edit)
            except ValueError:
                return False
            return mn <= v <= mx

        vcmd = (self.register(_validator), "%P")

        spn = ttk.Spinbox(
            frm,
            from_=mn,
            to=mx,
            increment=step,
            width=5,
            textvariable=var,
            validate="key",
            validatecommand=vcmd,
        )
        spn.pack(side=tk.RIGHT, padx=5)

        # ensure clamping on focus-loss or <Return>
        def _clamp_event(_):
            try:
                v = float(var.get())
            except ValueError:
                var.set(mn)
                return
            if v < mn:
                var.set(mn)
            elif v > mx:
                var.set(mx)

        spn.bind("<FocusOut>", _clamp_event)
        spn.bind("<Return>", _clamp_event)

        ttk.Label(parent, text=hint, font=("Tahoma", 8), foreground="gray") \
            .pack(fill=tk.X)

    def start_image_processing(self):
        imgs = self.input_selector.selected_paths
        books = self.book_selector.selected_paths
        if not imgs or not books:
            messagebox.showwarning("خطا", "لطفاً تصویر و تصاویر کتاب/فیلم را انتخاب کنید")
            return
        self.process_img_btn.config(state='disabled')
        threading.Thread(target=self._process_images, args=(imgs[0], books), daemon=True).start()

    def _process_images(self, img_path, book_paths):
        self.after(0, self.show_progress_dialog)
        results = []
        total = len(book_paths)
        for idx, path in enumerate(book_paths, start=1):
            self.after(0, lambda c=idx, t=total, p=path:
            self.update_progress(c, t, f"پردازش {os.path.basename(p)}"))
            res = self.book_movie_use_case.execute_single_comparison(img_path, path)
            if res:
                results.append(res)
        results.sort(key=lambda x: x.confidence_score, reverse=True)
        self.after(0, lambda: self.show_results(img_path, results))
        self.after(0, self.close_progress_dialog)

    def start_video_processing(self):
        vids = self.video_selector.selected_paths
        books = self.book_selector.selected_paths
        if not vids or not books:
            messagebox.showwarning("خطا", "لطفاً ویدئو و تصاویر کتاب/فیلم را انتخاب کنید")
            return
        # reinitialize video use case with GUI parameters
        self.video_use_case = ProcessVideoUseCase(
            video_repo=self.video_repository,
            frame_extractor=FrameExtractor(frame_skip=self.frame_skip_var.get()),
            feature_extractor=self.feature_extractor,
            matcher=self.matcher,
            image_repo=self.image_repository,
            phash_thresh=self.phash_var.get(),
            hist_thresh=self.hist_var.get()
        )
        self.process_vid_btn.config(state='disabled')
        threading.Thread(target=self._process_video, args=(vids[0],), daemon=True).start()

    def _process_video(self, video_path):
        self.after(0, self.show_progress_dialog)
        self.after(0, lambda: self.progress_dialog.progress.config(mode='indeterminate'))
        self.after(0, lambda: self.progress_dialog.progress.start(10))

        results = self.video_use_case.execute(video_path)

        self.after(0, lambda: self.progress_dialog.progress.stop())
        self.after(0, self.close_progress_dialog)
        results.sort(key=lambda x: x.confidence_score, reverse=True)
        self.after(0, lambda: self.show_results(video_path, results))

    def show_progress_dialog(self):
        self.progress_dialog = ProgressDialog(self, "در حال پردازش")

    def update_progress(self, current, total, status):
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.update_progress(current, total, status)

    def close_progress_dialog(self):
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            del self.progress_dialog

    def show_results(self, input_path, results):
        self.result_display.clear_results()
        if not results:
            messagebox.showinfo("نتیجه", "هیچ تطبیقی یافت نشد")
        else:
            for rank, res in enumerate(results, start=1):
                self.result_display.add_result(
                    input_path=input_path,
                    result_path=res.target_image_path,
                    confidence=res.confidence_score,
                    matches=res.good_matches_count,
                    rank=rank,
                    error_message=res.error_message,
                    source_frame_path=res.source_frame_path  # pass frame path for video
                )
            messagebox.showinfo("تکمیل", f"یافت شد {len(results)} تطبیق.")
        self.process_img_btn.config(state='normal')
        self.process_vid_btn.config(state='normal')

    def clear_results(self):
        self.result_display.clear_results()
        self.process_img_btn.config(state='normal')
        self.process_vid_btn.config(state='normal')

    def clear_frame_cache(self):
        if self.video_use_case:
            self.video_use_case.clear_cache()
            messagebox.showinfo("تکمیل", "کش ویدئو پاک شد")
        else:
            messagebox.showwarning("خطا", "ابتدا یک ویدئو پردازش کنید")


if __name__ == "__main__":
    app = BookCoverRecognitionApp()
    app.mainloop()
