import tkinter as tk
from tkinter import ttk, messagebox
from .components import ImageSelector, ProgressDialog, ResultsDisplay
from src.application.use_cases.find_matching_book_movie import FindMatchingBookMovieUseCase
from src.infrastructure.feature_extractors.sift_extractor import SIFTExtractor
from src.infrastructure.matchers.flann_matcher import FLANNMatcher
from src.infrastructure.repositories.file_image_repository import FileImageRepository
from ttkthemes import ThemedTk
import os
import threading


class BookCoverRecognitionApp(ThemedTk):
    def __init__(self):
        super().__init__(theme="arc")
        self.title("سیستم تشخیص جلد کتاب")
        self.minsize(800, 600)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.geometry("1000x800")

        # Persian font
        self.option_add("*Font", "Tahoma 10")

        self.create_widgets()
        self.setup_use_case()

    def setup_use_case(self):
        self.feature_extractor = SIFTExtractor()
        self.matcher = FLANNMatcher()
        self.image_repository = FileImageRepository()
        self.use_case = FindMatchingBookMovieUseCase(
            feature_extractor=self.feature_extractor,
            matcher=self.matcher,
            image_repository=self.image_repository
        )

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Settings section
        settings_frame = ttk.LabelFrame(main_frame, text="تنظیمات ورودی", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        self.input_selector = ImageSelector(
            settings_frame,
            "تصویر ورودی را انتخاب کنید",
            default_path="data/input_images/Return.jpg"
        )
        self.input_selector.pack(fill=tk.X, pady=5)

        self.book_selector = ImageSelector(
            settings_frame,
            "پوشه تصاویر کتاب فیلم را انتخاب کنید",
            default_path="data/book_movie_images",
            is_folder=True
        )
        self.book_selector.pack(fill=tk.X, pady=5)

        # Processing button
        process_frame = ttk.Frame(main_frame)
        process_frame.pack(fill=tk.X, pady=10)

        self.process_btn = ttk.Button(process_frame, text="شروع تشخیص",
                                      command=self.start_processing,
                                      style="Accent.TButton")
        self.process_btn.pack(side=tk.RIGHT, padx=5)

        ttk.Button(process_frame, text="پاک کردن نتایج",
                   command=self.clear_results).pack(side=tk.RIGHT, padx=5)

        self.result_display = ResultsDisplay(main_frame)
        self.result_display.pack(fill=tk.BOTH, expand=True, pady=10)

    def start_processing(self):
        input_paths = self.input_selector.selected_paths
        book_paths = self.book_selector.selected_paths

        if not input_paths:
            messagebox.showwarning("خطا", "لطفاً تصویر ورودی را انتخاب کنید")
            return

        if not book_paths:
            messagebox.showwarning("خطا", "لطفاً تصاویر کتاب/فیلم را انتخاب کنید")
            return

        self.process_btn.config(state='disabled')

        thread = threading.Thread(target=self.process_images, args=(input_paths[0], book_paths))
        thread.daemon = True
        thread.start()

    def process_images(self, input_path, book_paths):
        # Show progress dialog
        self.after(0, self.show_progress_dialog)

        results = []
        total_books = len(book_paths)

        for index, book_path in enumerate(book_paths, start=1):
            self.after(
                0,
                lambda current=index, total=total_books, path=book_path:
                self.update_progress(current, total, f"پردازش {os.path.basename(path)}")
            )

            # Calculating the match for each book
            result = self.use_case.execute_single_comparison(input_path, book_path)
            if result:
                results.append(result)

        results.sort(key=lambda x: x.confidence_score, reverse=True)

        # Show results
        self.after(0, lambda: self.show_results(input_path, results))

        # Close results display
        self.after(0, self.close_progress_dialog)

    def show_progress_dialog(self):
        self.progress_dialog = ProgressDialog(self, "در حال تشخیص جلد کتاب...")

    def update_progress(self, current, total, status):
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.update_progress(current, total, status)

    def close_progress_dialog(self):
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

    def show_results(self, input_path, results):
        self.result_display.clear_results()

        if not results:
            messagebox.showinfo("نتیجه", "هیچ تطبیقی یافت نشد")
            return

        # Show all results
        for rank, result in enumerate(results, 1):
            self.result_display.add_result(
                input_path=input_path,
                result_path=result.target_image_path,
                confidence=result.confidence_score,
                matches=result.good_matches_count,
                rank=rank,
                error_message=result.error_message
            )

        messagebox.showinfo("تکمیل", f"تشخیص تکمیل شد. {len(results)} تطبیق یافت شد.")

    def clear_results(self):
        self.result_display.clear_results()
        self.process_btn.config(state='normal')
