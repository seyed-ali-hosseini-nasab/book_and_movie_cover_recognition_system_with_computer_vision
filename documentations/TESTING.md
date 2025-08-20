# آزمون‌ها (TESTING)

این سند دستورالعمل اجرای تست‌های واحد و انتها-به‌انتها برای پروژه "سیستم تشخیص جلد کتاب و فیلم" را ارائه می‌دهد.

## محیط تست
- از فریم‌ورک **pytest** استفاده می‌شود.  
- پوشش تست‌ها با **coverage** اندازه‌گیری می‌شود.

## اجرای تست‌ها

1. فعال‌سازی محیط مجازی:
source venv/bin/activate # یا venv\Scripts\activate در ویندوز
2. اجرای تمامی تست‌ها:
pytest --maxfail=1 --disable-warnings -q
3. تولید گزارش پوشش کد:
coverage run -m pytest
coverage report -m

## ساختار تست‌ها
```
tests/
├── test_input_video_replacement.py # تستهای ویدئوهای ورودی
├── test_overlay_generation.py # تستهای همپوشانی تصاویر ورودی
└── utils.py # کدهای کاربردی خارج از منطق تست
```
