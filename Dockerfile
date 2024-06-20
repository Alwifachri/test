# Gunakan gambar dasar resmi Python
FROM python:3.9-slim

# Tentukan direktori kerja dalam container
WORKDIR /app

# Salin requirements.txt ke direktori kerja
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin sisa kode aplikasi
COPY . .

# Tentukan command untuk menjalankan aplikasi
CMD ["gunicorn", "-b", ":$PORT", "app:app"]
