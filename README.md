# Streamlit — Klasifikasi Keseluruhan Dokumen Pajak Daerah

Aplikasi menerima **PDF digital dan PDF hasil scan**. Isi dokumen diekstrak,
halaman scan diproses dengan Tesseract OCR, kemudian keseluruhan dokumen
diklasifikasikan ke salah satu dari tujuh kelas pajak menggunakan IndoBERT-p1.

## 1. Siapkan model dari Google Colab

Unduh `model_domain_p1.zip` dari bagian ekspor model pada notebook p1.
Ekstrak seluruh isinya langsung ke folder `model` di samping `app.py`.

Struktur:
```text
streamlit_indobert_pajak/
├── app.py
├── requirements.txt
├── README.md
└── model/
    ├── config.json
    ├── model.safetensors (atau pytorch_model.bin)
    ├── tokenizer.json / vocab.txt
    └── file tokenizer lainnya
```

Jangan gunakan checkpoint MLM/CPT_DIR. Aplikasi membutuhkan model klasifikasi
hasil fine-tuning yang diekspor dari FINAL_DIR.

## 2. Instal Tesseract OCR di Windows

PDF hasil scan membutuhkan aplikasi Tesseract OCR selain Python package.

1. Instal Tesseract OCR untuk Windows.
2. Pastikan `tesseract.exe` dapat ditemukan melalui PATH.
3. Pasang data bahasa Indonesia (`ind.traineddata`) dan English (`eng.traineddata`)
   pada folder `tessdata` instalasi Tesseract.
4. Jika Tesseract tidak terdeteksi otomatis, atur environment variable
   `TESSERACT_CMD` ke lokasi `tesseract.exe`, misalnya:
   ```powershell
   $env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

Aplikasi memilih `ind+eng` bila tersedia. Jika hanya `eng` yang tersedia,
OCR tetap dapat berjalan tetapi pembacaan teks Bahasa Indonesia mungkin kurang akurat.

## 3. Instalasi Python

Buka terminal pada folder aplikasi:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Jalankan

```powershell
streamlit run app.py
```

Buka alamat lokal yang ditampilkan Streamlit, biasanya http://localhost:8501.

## 5. Cara klasifikasi keseluruhan dokumen

- PDF diproses halaman demi halaman.
- Halaman dengan teks digital yang sangat sedikit dianggap berpotensi hasil scan
  dan diproses menggunakan OCR.
- Teks digabungkan dan dibersihkan.
- Dokumen dibagi menjadi potongan berurutan dengan maksimum 510 token isi
  (ditambah token khusus model sehingga total maksimum 512).
- Model memprediksi setiap potongan.
- Probabilitas setiap kelas dirata-ratakan dengan bobot jumlah token isi tiap
  potongan. Kelas dengan probabilitas agregat tertinggi ditampilkan sebagai
  kategori dominan dokumen.

## 6. Catatan metodologis penting

- Urutan `LABELS` pada app.py harus sama persis dengan `LABEL_MAP` saat
  fine-tuning IndoBERT-p1.
- Fungsi `clean_text()` harus disamakan dengan fungsi `remove_noise()` pada
  notebook sebelum pengujian akhir agar preprocessing konsisten.
- Agregasi probabilitas per potongan adalah metode untuk menghasilkan satu
  label dokumen. Metode ini perlu dijelaskan dan diuji sebagai bagian dari
  implementasi, karena model aslinya dilatih pada contoh teks, bukan label
  dokumen panjang secara langsung.
- PDF hasil scan memerlukan OCR; kualitas hasil bergantung pada resolusi,
  kemiringan, dan kejernihan scan.
- Skor probabilitas model bukan jaminan kebenaran prediksi dan bukan keputusan
  pajak resmi.
