import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import matplotlib.pyplot as plt

#LOAD DATASET
file_path = "data/cleaned_lung_cancer.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("File CSV tidak ditemukan. Menggunakan dummy data...")
    df = pd.DataFrame(columns=[
        'GENDER','AGE','SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC_DISEASE',
        'FATIGUE','ALLERGY','WHEEZING','ALCOHOL_CONSUMING','COUGHING','SHORTNESS_OF_BREATH',
        'SWALLOWING_DIFFICULTY','CHEST_PAIN','LUNG_CANCER'
    ])

#FEATURE ENGINEERING
gejala_cols = ['COUGHING','SHORTNESS_OF_BREATH','WHEEZING','CHEST_PAIN','SWALLOWING_DIFFICULTY','FATIGUE']
gejala_bobot = [0.22, 0.22, 0.08, 0.22, 0.08, 0.18]
gejala_cols_existing = [col for col in gejala_cols if col in df.columns]
df['skor_gejala'] = df[gejala_cols_existing].fillna(0).dot(
    np.array([gejala_bobot[gejala_cols.index(c)] for c in gejala_cols_existing])
)

risiko_cols = ['SMOKING','ALCOHOL_CONSUMING','YELLOW_FINGERS','CHRONIC_DISEASE','GENDER','ANXIETY','PEER_PRESSURE','ALLERGY']
risiko_bobot = [0.27,0.12,0.23,0.08,0.05,0.05,0.1,0.1]
risiko_cols_existing = [col for col in risiko_cols if col in df.columns]
df['skor_risiko'] = df[risiko_cols_existing].fillna(0).dot(
    np.array([risiko_bobot[risiko_cols.index(c)] for c in risiko_cols_existing])
)

#NORMALISASI SKOR GEJALA & RISIKO
df['skor_gejala_norm'] = df['skor_gejala'] / df['skor_gejala'].max() * 100 if df['skor_gejala'].max() > 0 else 0
df['skor_risiko_norm'] = df['skor_risiko'] / df['skor_risiko'].max() * 100 if df['skor_risiko'].max() > 0 else 0

#FUZZIFIKASI
umur = ctrl.Antecedent(np.arange(0,101,1), 'umur')
umur['Muda'] = fuzz.gaussmf(umur.universe, 25, 10)
umur['Paruh_Baya'] = fuzz.gaussmf(umur.universe, 50, 10)
umur['Tua'] = fuzz.gaussmf(umur.universe, 70, 10)

gejala = ctrl.Antecedent(np.arange(0,101,1), 'gejala')
gejala['Ringan'] = fuzz.trimf(gejala.universe,[0,0,40])
gejala['Sedang'] = fuzz.trimf(gejala.universe,[30,50,70])
gejala['Berat'] = fuzz.trimf(gejala.universe,[60,100,100])

risiko = ctrl.Antecedent(np.arange(0,101,1), 'risiko')
risiko['Rendah'] = fuzz.trimf(risiko.universe,[0,0,40])
risiko['Sedang'] = fuzz.trimf(risiko.universe,[30,50,70])
risiko['Tinggi'] = fuzz.trimf(risiko.universe,[60,100,100])

diagnosa = ctrl.Consequent(np.arange(0,101,1), 'diagnosa')
diagnosa['Negatif'] = fuzz.trimf(diagnosa.universe,[0,0,48])
diagnosa['Positif'] = fuzz.trimf(diagnosa.universe,[48,100,100])

#RULE
rule1 = ctrl.Rule(gejala['Berat'] | risiko['Tinggi'], diagnosa['Positif'])
rule2 = ctrl.Rule(gejala['Sedang'] & risiko['Sedang'], diagnosa['Positif'])
rule3 = ctrl.Rule(gejala['Ringan'] & risiko['Tinggi'], diagnosa['Positif'])
rule4 = ctrl.Rule(gejala['Ringan'] & risiko['Sedang'] & umur['Tua'], diagnosa['Positif'])
rule5 = ctrl.Rule(gejala['Ringan'] & risiko['Rendah'], diagnosa['Negatif'])
rule6 = ctrl.Rule(gejala['Sedang'] & risiko['Rendah'], diagnosa['Negatif'])
rule7 = ctrl.Rule(gejala['Berat'] & risiko['Rendah'], diagnosa['Positif'])
rule8 = ctrl.Rule(gejala['Ringan'] & umur['Muda'], diagnosa['Negatif'])
rule9 = ctrl.Rule(gejala['Sedang'] & umur['Muda'], diagnosa['Negatif'])
rule10 = ctrl.Rule(gejala['Berat'] & umur['Tua'], diagnosa['Positif'])
rule11 = ctrl.Rule(gejala['Sedang'] & risiko['Sedang'] & umur['Tua'], diagnosa['Positif'])

system_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5,
    rule6, rule7, rule8, rule9, rule10,
    rule11
])

simulasi = ctrl.ControlSystemSimulation(system_ctrl)


#PREDIKSI
def get_diagnosa_label(score):
    return "TERKENA LUNG CANCER" if score >= 55 else "TIDAK TERKENA"

hasil_score = []
hasil_label = []

for idx, row in df.iterrows():
    try:
        simulasi.input['umur'] = row['AGE']
        simulasi.input['gejala'] = row['skor_gejala_norm']
        simulasi.input['risiko'] = row['skor_risiko_norm']
        simulasi.compute()
        score = simulasi.output['diagnosa']
    except Exception:
        score = 0
    hasil_score.append(score)
    hasil_label.append(get_diagnosa_label(score))

df['Fuzzy_Score'] = hasil_score
df['Hasil_Diagnosa'] = hasil_label
    
#EVALUASI LENGKAP
df['Hasil_Label'] = df['Hasil_Diagnosa'].map({'TIDAK TERKENA':0,'TERKENA LUNG CANCER':1})

accuracy = accuracy_score(df['LUNG_CANCER'], df['Hasil_Label'])
precision = precision_score(df['LUNG_CANCER'], df['Hasil_Label'])
recall = recall_score(df['LUNG_CANCER'], df['Hasil_Label'])
f1 = f1_score(df['LUNG_CANCER'], df['Hasil_Label'])
loss = log_loss(df['LUNG_CANCER'], np.array(df['Fuzzy_Score'])/100)

print("\n=== HASIL EVALUASI FIS ===")
print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1 Score : {f1*100:.2f}%")
print(f"Log Loss : {loss:.4f}")

