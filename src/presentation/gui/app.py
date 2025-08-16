import tkinter as tk
from tkinter import ttk, messagebox

from src.application.use_cases import AsyncFrameProcessor
from .components import (
    ScrollableFrame,
    ImageSelector,
    VideoSelector,
    ProgressDialog,
    ResultsDisplay
)
from src.application.use_cases.image_processing import FindMatchingBookMovieUseCase, OverlayBookCoverUseCase
from src.application.use_cases.video_processing import ProcessInputVideoUseCase
from src.infrastructure.feature_extractors.sift_extractor import SIFTExtractor
from src.infrastructure.matchers.flann_matcher import FLANNMatcher
from src.infrastructure.repositories.file_image_repository import FileImageRepository
from src.infrastructure.repositories.file_video_repository import FileVideoRepository
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

        # Image processing use cases
        self.book_movie_use_case = FindMatchingBookMovieUseCase(
            feature_extractor=self.feature_extractor,
            matcher=self.matcher,
            image_repository=self.image_repository
        )
        self.overlay_use_case = OverlayBookCoverUseCase(
            feature_extractor=self.feature_extractor,
            matcher=self.matcher
        )

        self.frame_processor_async = AsyncFrameProcessor(self.book_movie_use_case, max_workers=6)

        # Video processing use case (init on demand)
        self.video_use_case = None

    def create_widgets(self):
        root_scroll = ScrollableFrame(self)
        root_scroll.pack(fill=tk.BOTH, expand=True)
        container = root_scroll.scrollable_frame
        container.columnconfigure(0, weight=1, minsize=1080)

        # Settings panel
        settings = ttk.LabelFrame(container, text="تنظیمات ورودی", padding=10)
        settings.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        settings.columnconfigure(0, weight=1)

        # Image selector
        self.input_selector = ImageSelector(
            settings,
            "تصویر ورودی",
            default_path="data/input_images/Return.jpg"
        )
        self.input_selector.pack(fill=tk.X, pady=5)

        self.book_selector = ImageSelector(
            settings,
            "پوشه تصاویر کتاب/فیلم",
            default_path="data/book_movie_images",
            is_folder=True
        )
        self.book_selector.pack(fill=tk.X, pady=5)

        # Video selector
        self.video_selector = VideoSelector(
            settings,
            "ویدئوی ورودی",
            default_path="data/input_videos"
        )
        self.video_selector.pack(fill=tk.X, pady=5)

        # Parameter controls
        self.frame_skip_var = tk.IntVar(value=1)
        self.min_conf_var = tk.DoubleVar(value=5.0)

        self._build_spinbox(
            settings, "Frame Skip:", self.frame_skip_var,
            mn=1, mx=120, step=1,
            hint="هر چه عدد بیشتر باشد فریم‌های کمتری پردازش می‌شوند"
        )
        self._build_spinbox(
            settings, "Min Confidence:", self.min_conf_var,
            mn=0.0, mx=100.0, step=0.5,
            hint="حداقل مقدار confidence برای تشخیص کتاب"
        )

        # Action buttons
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
            actions, text="پردازش ویدئو",
            command=self.start_video_processing,
            style="Accent.TButton"
        )
        self.process_vid_btn.pack(side=tk.RIGHT, padx=5)

        ttk.Button(actions, text="پاک کردن نتایج", command=self.clear_results) \
            .pack(side=tk.RIGHT, padx=5)

        # Results display
        self.result_display = ResultsDisplay(container)
        self.result_display.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        container.rowconfigure(2, weight=1)

    def _build_spinbox(self, parent, text, var, mn, mx, step, hint):
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.X, pady=5)
        ttk.Label(frm, text=text).pack(side=tk.RIGHT)

        def _validator(value_if_edit):
            if value_if_edit == "":
                return True
            try:
                v = float(value_if_edit)
            except ValueError:
                return False
            return mn <= v <= mx

        vcmd = (self.register(_validator), "%P")
        spn = ttk.Spinbox(
            frm, from_=mn, to=mx, increment=step,
            width=5, textvariable=var,
            validate="key", validatecommand=vcmd
        )
        spn.pack(side=tk.RIGHT, padx=5)

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
        ttk.Label(parent, text=hint, font=("Tahoma", 8), foreground="gray") \
            .pack(fill=tk.X)

    def start_image_processing(self):
        imgs = self.input_selector.selected_paths
        books = self.book_selector.selected_paths
        if not imgs or not books:
            messagebox.showwarning("خطا", "لطفاً تصویر و تصاویر کتاب/فیلم را انتخاب کنید")
            return
        self.process_img_btn.config(state='disabled')
        threading.Thread(
            target=self._process_images,
            args=(imgs[0], books),
            daemon=True
        ).start()

    def _process_images(self, img_path, book_paths):
        self.after(0, self.show_progress_dialog)
        results = []
        total = len(book_paths)
        for idx, book_path in enumerate(book_paths, start=1):
            self.after(
                0,
                lambda c=idx,
                       t=total,
                       p=book_path: self.update_progress(c, t, f"Processing {os.path.basename(p)}")
            )
            match = self.book_movie_use_case.execute_single_comparison_with_overlay(
                input_image_path=img_path,
                book_image_path=book_path,
                enable_overlay=True
            )
            results.append(match)
        results.sort(key=lambda r: r.confidence_score, reverse=True)
        self.after(0, lambda: self.show_results(img_path, results))
        self.after(0, self.close_progress_dialog)

    def start_video_processing(self):
        vids = self.video_selector.selected_paths
        if not vids:
            messagebox.showwarning("خطا", "لطفاً ویدئو را انتخاب کنید")
            return

        # Initialize video processing use case
        self.video_use_case = ProcessInputVideoUseCase(
            feature_extractor=self.feature_extractor,
            matcher=self.matcher,
            image_repository=self.image_repository,
            video_repository=self.video_repository,
            frame_processor=self.frame_processor_async,
            min_conf=self.min_conf_var.get()
        )

        self.process_vid_btn.config(state='disabled')
        threading.Thread(
            target=self._process_video,
            args=(os.path.basename(vids[0]),),
            daemon=True
        ).start()

    def _process_video(self, video_name):
        self.after(0, self.show_progress_dialog)

        def progress(msg, pct):
            self.after(0, lambda: self.update_progress(0, 0, msg))

        result = self.video_use_case.execute(
            input_video_name=video_name,
            progress_callback=progress
        )
        self.after(0, self.close_progress_dialog)
        # Display video processing result
        if result.success:
            messagebox.showinfo(
                "تکمیل",
                f"📹 ویدئو: {result.source_video_name}\n"
                f"📖 کتاب: {result.target_book_name}\n"
                f"🎬 فریم‌های جایگزین: {result.replaced_frames_count}/{result.total_frames_processed}\n"
                f"⏱ زمان: {result.processing_time_seconds:.1f}s\n"
                f"💾 خروجی: {result.output_video_path}"
            )
        else:
            messagebox.showerror("خطا", result.error_message)

        self.process_vid_btn.config(state='normal')

    def show_progress_dialog(self):
        self.progress_dialog = ProgressDialog(self, "در حال پردازش")

    def update_progress(self, current, total, status):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.update_progress(current, total, status)

    def close_progress_dialog(self):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
            del self.progress_dialog

    def show_results(self, input_path, results, max_results=5):
        self.result_display.clear_results()
        if not results:
            messagebox.showinfo("نتیجه", "هیچ تطبیقی یافت نشد")
        else:
            min_res = results[:min(len(results), max_results)]
            for rank, res in enumerate(min_res, start=1):
                self.result_display.add_result(
                    input_path=input_path,
                    result_path=res.target_image_path,
                    confidence=res.confidence_score,
                    matches=res.good_matches_count,
                    rank=rank,
                    error_message=res.error_message,
                    source_frame_path=res.source_frame_path,
                    overlay_image_path=res.overlay_image_path
                )
            messagebox.showinfo("تکمیل", f"{len(results)} تطبیق یافت شد و {len(min_res)} تای آن در نتایج قابل مشاهده هستند")
        self.process_img_btn.config(state='normal')

    def clear_results(self):
        self.result_display.clear_results()
        self.process_img_btn.config(state='normal')
        self.process_vid_btn.config(state='normal')
        self.video_use_case = None


if __name__ == "__main__":
    app = BookCoverRecognitionApp()
    app.mainloop()
