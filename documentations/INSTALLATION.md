# دستورالعمل نصب (INSTALLATION)

این راهنما نحوه آماده‌سازی محیط توسعه و نصب پیش‌نیازهای لازم برای اجرای پروژه "سیستم تشخیص جلد کتاب و فیلم" را تشریح می‌کند.

## پیش‌نیازها
- Python 3.8 یا بالاتر  
- pip (مدیریت بسته‌های Python)  
- Git  

## مراحل نصب

1. کلون کردن مخزن پروژه:
git clone https://github.com/seyed-ali-hosseini-nasab/computer_vision_book_cover_recognition_system.git
cd computer_vision_book_cover_recognition_system

2. ایجاد و فعال‌سازی محیط مجازی (Virtual Environment):
python -m venv venv

در ویندوز
venv\Scripts\activate

در لینوکس/مک
source venv/bin/activate

3. نصب وابستگی‌ها:
pip install -r requirements.txt


4. تنظیم فایل نگاشت کتاب به تریلر:
فایل `data/book_trailer_mapping.json` را بررسی و در صورت نیاز کلیدها (نام کتاب) را با نام فایل‌های جلد در پوشه `data/book_movie_images` یکسان کنید.

5. راه‌اندازی GUI یا خط فرمان:
- اجرای واسط گرافیکی:
  ```
  python src/presentation/gui/main.py
  ```
- استفاده از خط فرمان (در صورت وجود اسکریپت):
  ```
  python run_app.py
  ```

پروژه اکنون آماده اجرا است.
