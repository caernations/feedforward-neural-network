# Feedforward Neural Network Implementation

Implementasi neural network sederhana menggunakan Python dan NumPy.

## Struktur Proyek

```
├── docs/
├── src/
│   ├── models/    # Direktori untuk menyimpan model model neural network
│   ├── configs/
│   │   └── config_loader.py      # Fungsi untuk memuat konfigurasi
│   ├── utils/    # Direktori file pembantu
│   ├── main.py            # Script utama
│   ├── notebook.py     # File test
    └── notebook.ipynb
└── requirements.txt      # Dependensi proyek
```

## Instalasi

1. Clone repositori:

```bash
git clone https://github.com/caernations/feedforward-neural-network.git
cd feedforward-neural-network
```

1. Buat virtual environment (opsional):

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

## Author
| **Name**                                                | **NIM**  | **Tugas** |
| ------------------------------------------------------- | -------- | --------- |
| [Yasmin Farisah Salma](https://github.com/caernations)  | 13522140 | Semuanya  |
| [Mohammad Akmal Ramadan](https://github.com/akmalrmn)   | 13522161 | Semuanya  |