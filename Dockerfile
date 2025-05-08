# نبدأ من صورة Python الرسمية
FROM python:3.9-slim

# تعيين مجلد العمل في الحاوية
WORKDIR /app

# نسخ ملفات المشروع إلى الحاوية
COPY . .

# تثبيت المكتبات المطلوبة من ملف requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# تعيين المنفذ الذي سيتم تشغيل التطبيق عليه
EXPOSE 8501

# تشغيل تطبيق Streamlit
CMD ["streamlit", "run", "app.py"]
