# Feedforward Neural Network Implementation

Implementasi neural network sederhana menggunakan Python dan NumPy.

## Struktur Proyek

```
.
├── configs/
│   └── config.yaml         # Konfigurasi model dan training
├── src/
│   ├── models/
│   │   └── neural_network.py  # Implementasi model neural network
│   ├── config/
│   │   └── config.py      # Fungsi untuk memuat konfigurasi
│   └── main.py            # Script utama untuk menjalankan model
├── data/                  # Direktori untuk data
├── models/                # Direktori untuk menyimpan model
├── notebooks/            # Jupyter notebooks
├── tests/                # Unit tests
└── requirements.txt      # Dependensi proyek
```

## Instalasi

1. Clone repositori:

```bash
git clone https://github.com/caernations/feedforward-neural-network.git
cd feedforward-neural-network
```

2. Buat virtual environment (opsional tapi direkomendasikan):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependensi:

```bash
pip install -r requirements.txt
```

## Penggunaan

1. Sesuaikan konfigurasi di `configs/config.yaml`

2. Jalankan model:

```bash
python src/main.py
```

## Konfigurasi

File `config.yaml` berisi semua parameter yang dapat dikonfigurasi:

- Arsitektur model (ukuran input, hidden layers, output)
- Fungsi aktivasi
- Metode inisialisasi bobot
- Parameter training (batch size, learning rate, epochs)
- Path data dan model

## Fitur

- Implementasi neural network dari awal menggunakan NumPy
- Mendukung berbagai fungsi aktivasi (ReLU, Sigmoid, Tanh, Softmax)
- Mendukung berbagai metode inisialisasi bobot (Xavier, He, dll)
- Visualisasi arsitektur model dan distribusi bobot
- Konfigurasi melalui file YAML
- Progress bar untuk monitoring training
