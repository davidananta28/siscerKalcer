import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# =============================
# 1. MF Umur (Gaussian)
# =============================
x_umur = np.arange(0, 101, 1)
muda = fuzz.gaussmf(x_umur, 25, 10)
paruh_baya = fuzz.gaussmf(x_umur, 50, 10)
tua = fuzz.gaussmf(x_umur, 70, 10)

plt.figure()
plt.plot(x_umur, muda, label='Muda', linewidth=2)
plt.plot(x_umur, paruh_baya, label='Paruh Baya', linewidth=2)
plt.plot(x_umur, tua, label='Tua', linewidth=2)
plt.title('Fungsi Keanggotaan Umur')
plt.xlabel('Umur')
plt.ylabel('Derajat Keanggotaan')
plt.legend()
plt.grid(True)

# =============================
# 2. MF Skor Gejala (Triangular)
# =============================
x_gejala = np.arange(0, 101, 1)
ringan = fuzz.trimf(x_gejala, [0, 0, 40])
sedang = fuzz.trimf(x_gejala, [30, 50, 70])
berat = fuzz.trimf(x_gejala, [60, 100, 100])

plt.figure()
plt.plot(x_gejala, ringan, label='Ringan', linewidth=2)
plt.plot(x_gejala, sedang, label='Sedang', linewidth=2)
plt.plot(x_gejala, berat, label='Berat', linewidth=2)
plt.title('Fungsi Keanggotaan Skor Gejala')
plt.xlabel('Skor Gejala')
plt.ylabel('Derajat Keanggotaan')
plt.legend()
plt.grid(True)

# =============================
# 3. MF Skor Risiko (Triangular)
# =============================
x_risiko = np.arange(0, 101, 1)
rendah = fuzz.trimf(x_risiko, [0, 0, 40])
sedang_r = fuzz.trimf(x_risiko, [30, 50, 70])
tinggi = fuzz.trimf(x_risiko, [60, 100, 100])

plt.figure()
plt.plot(x_risiko, rendah, label='Rendah', linewidth=2)
plt.plot(x_risiko, sedang_r, label='Sedang', linewidth=2)
plt.plot(x_risiko, tinggi, label='Tinggi', linewidth=2)
plt.title('Fungsi Keanggotaan Skor Risiko')
plt.xlabel('Skor Risiko')
plt.ylabel('Derajat Keanggotaan')
plt.legend()
plt.grid(True)

# =============================
# 4. MF Diagnosis (Triangular)
# =============================
x_diag = np.arange(0, 101, 1)
negatif = fuzz.trimf(x_diag, [0, 0, 55])
positif = fuzz.trimf(x_diag, [55, 100, 100])

plt.figure()
plt.plot(x_diag, negatif, label='Negatif', linewidth=2)
plt.plot(x_diag, positif, label='Positif', linewidth=2)
plt.title('Fungsi Keanggotaan Diagnosis')
plt.xlabel('Nilai Diagnosis')
plt.ylabel('Derajat Keanggotaan')
plt.legend()
plt.grid(True)

plt.show()
