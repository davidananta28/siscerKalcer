#====================================
#--------- TRAIN & EVALUASI ---------
#====================================

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/cleaned_lung_cancer.csv")

# Feature Engineering
gejala_cols = [
    'COUGHING','SHORTNESS_OF_BREATH','WHEEZING',
    'CHEST_PAIN','SWALLOWING_DIFFICULTY','FATIGUE'
]
gejala_bobot = [0.22, 0.22, 0.08, 0.22, 0.08, 0.18]
df['skor_gejala'] = df[gejala_cols].fillna(0).dot(gejala_bobot)

risiko_cols = [
    'SMOKING','ALCOHOL_CONSUMING','YELLOW_FINGERS',
    'CHRONIC_DISEASE','GENDER','ANXIETY',
    'PEER_PRESSURE','ALLERGY'
]
risiko_bobot = [0.27,0.12,0.23,0.08,0.05,0.05,0.10,0.10]
df['skor_risiko'] = df[risiko_cols].fillna(0).dot(risiko_bobot)

# Normalisasi ke 0–100
df['skor_gejala_norm'] = df['skor_gejala'] / df['skor_gejala'].max() * 100
df['skor_risiko_norm'] = df['skor_risiko'] / df['skor_risiko'].max() * 100

# Train Test Split
train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df['LUNG_CANCER']
)

# Variabel Fuzzy
umur = ctrl.Antecedent(np.arange(0,101,1), 'umur')
umur['Muda'] = fuzz.gaussmf(umur.universe, 25, 10)
umur['Paruh_Baya'] = fuzz.gaussmf(umur.universe, 50, 10)
umur['Tua'] = fuzz.gaussmf(umur.universe, 70, 10)

gejala = ctrl.Antecedent(np.arange(0,101,1), 'gejala')
gejala['Ringan'] = fuzz.trimf(gejala.universe,[0,0,40])
gejala['Sedang'] = fuzz.trimf(gejala.universe,[30,50,70])
gejala['Berat']  = fuzz.trimf(gejala.universe,[60,100,100])

risiko = ctrl.Antecedent(np.arange(0,101,1), 'risiko')
risiko['Rendah'] = fuzz.trimf(risiko.universe,[0,0,40])
risiko['Sedang'] = fuzz.trimf(risiko.universe,[30,50,70])
risiko['Tinggi'] = fuzz.trimf(risiko.universe,[60,100,100])

diagnosa = ctrl.Consequent(np.arange(0,101,1), 'diagnosa')
diagnosa['Negatif'] = fuzz.trimf(diagnosa.universe,[0,0,48])
diagnosa['Positif'] = fuzz.trimf(diagnosa.universe,[48,100,100])

# Rule Base
rules = [
    ctrl.Rule(gejala['Berat'] | risiko['Tinggi'], diagnosa['Positif']),
    ctrl.Rule(gejala['Sedang'] & risiko['Sedang'], diagnosa['Positif']),
    ctrl.Rule(gejala['Ringan'] & risiko['Tinggi'], diagnosa['Positif']),
    ctrl.Rule(gejala['Ringan'] & risiko['Sedang'] & umur['Tua'], diagnosa['Positif']),
    ctrl.Rule(gejala['Ringan'] & risiko['Rendah'], diagnosa['Negatif']),
    ctrl.Rule(gejala['Sedang'] & risiko['Rendah'], diagnosa['Negatif']),
    ctrl.Rule(gejala['Berat'] & risiko['Rendah'], diagnosa['Positif']),
    ctrl.Rule(gejala['Ringan'] & umur['Muda'], diagnosa['Negatif']),
    ctrl.Rule(gejala['Sedang'] & umur['Muda'], diagnosa['Negatif']),
    ctrl.Rule(gejala['Berat'] & umur['Tua'], diagnosa['Positif']),
    ctrl.Rule(gejala['Sedang'] & risiko['Sedang'] & umur['Tua'], diagnosa['Positif'])
]

system = ctrl.ControlSystem(rules)

# Inferensi 
hasil_score = []

for _, row in test_df.iterrows():
    sim = ctrl.ControlSystemSimulation(system)
    sim.input['umur'] = row['AGE']
    sim.input['gejala'] = row['skor_gejala_norm']
    sim.input['risiko'] = row['skor_risiko_norm']
    sim.compute()
    hasil_score.append(sim.output['diagnosa'])

test_df['Fuzzy_Score'] = hasil_score
test_df['Hasil_Label'] = (test_df['Fuzzy_Score'] >= 55).astype(int)

# Evaluasi Model
accuracy = accuracy_score(test_df['LUNG_CANCER'], test_df['Hasil_Label'])
precision = precision_score(test_df['LUNG_CANCER'], test_df['Hasil_Label'])
recall = recall_score(test_df['LUNG_CANCER'], test_df['Hasil_Label'])
f1 = f1_score(test_df['LUNG_CANCER'], test_df['Hasil_Label'])
loss = log_loss(test_df['LUNG_CANCER'], test_df['Fuzzy_Score'] / 100)

print("\n=== HASIL EVALUASI FIS (DATA UJI) ===")
print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1 Score : {f1*100:.2f}%")
print(f"Log Loss : {loss:.4f}")

# Visualisasi
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy*100, precision*100, recall*100, f1*100]

plt.figure(figsize=(8,5))
plt.bar(metrics, values)
plt.title("Evaluasi Kinerja Fuzzy Inference System")
plt.ylabel("Persentase (%)")
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
