from flask import Flask, render_template, request
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)

# ==========================================
# 1. INISIALISASI SISTEM FUZZY (Hanya sekali saat start)
# ==========================================

# -- Antecedents (Input) --
umur = ctrl.Antecedent(np.arange(0, 101, 1), 'umur')
gejala = ctrl.Antecedent(np.arange(0, 101, 1), 'gejala')
risiko = ctrl.Antecedent(np.arange(0, 101, 1), 'risiko')

# -- Consequent (Output) --
diagnosa = ctrl.Consequent(np.arange(0, 101, 1), 'diagnosa')

# -- Membership Functions (Sama seperti kode asli Tuan) --
umur['Muda'] = fuzz.gaussmf(umur.universe, 25, 10)
umur['Paruh_Baya'] = fuzz.gaussmf(umur.universe, 50, 10)
umur['Tua'] = fuzz.gaussmf(umur.universe, 70, 10)

gejala['Ringan'] = fuzz.trimf(gejala.universe, [0, 0, 40])
gejala['Sedang'] = fuzz.trimf(gejala.universe, [30, 50, 70])
gejala['Berat'] = fuzz.trimf(gejala.universe, [60, 100, 100])

risiko['Rendah'] = fuzz.trimf(risiko.universe, [0, 0, 40])
risiko['Sedang'] = fuzz.trimf(risiko.universe, [30, 50, 70])
risiko['Tinggi'] = fuzz.trimf(risiko.universe, [60, 100, 100])

diagnosa['Negatif'] = fuzz.trimf(diagnosa.universe, [0, 0, 55])
diagnosa['Positif'] = fuzz.trimf(diagnosa.universe, [55, 100, 100])

# -- Rules --
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
    rule6, rule7, rule8, rule9, rule10, rule11
])

simulasi = ctrl.ControlSystemSimulation(system_ctrl)

# ==========================================
# 2. KONFIGURASI BOBOT
# ==========================================
# Urutan harus sesuai dengan form HTML
GEJALA_KEYS = ['COUGHING', 'SHORTNESS_OF_BREATH', 'WHEEZING', 'CHEST_PAIN', 'SWALLOWING_DIFFICULTY', 'FATIGUE']
GEJALA_BOBOT = np.array([0.22, 0.22, 0.08, 0.22, 0.08, 0.18])

RISIKO_KEYS = ['SMOKING', 'ALCOHOL_CONSUMING', 'YELLOW_FINGERS', 'CHRONIC_DISEASE', 'GENDER', 'ANXIETY', 'PEER_PRESSURE', 'ALLERGY']
RISIKO_BOBOT = np.array([0.27, 0.12, 0.23, 0.08, 0.05, 0.05, 0.1, 0.1])

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            # 1. Ambil Data Form
            age = float(request.form.get('AGE'))
            
            # 2. Hitung Skor Gejala
            input_gejala = []
            for key in GEJALA_KEYS:
                # Form mengembalikan '1' (Ya) atau '0' (Tidak)
                val = int(request.form.get(key, 0))
                input_gejala.append(val)
            
            # Dot product (1/0 * bobot)
            raw_gejala_score = np.dot(input_gejala, GEJALA_BOBOT)
            # Normalisasi ke 0-100 (Asumsi max bobot sum = 1.0)
            norm_gejala_score = raw_gejala_score * 100

            # 3. Hitung Skor Risiko
            input_risiko = []
            for key in RISIKO_KEYS:
                val = int(request.form.get(key, 0))
                input_risiko.append(val)
            
            raw_risiko_score = np.dot(input_risiko, RISIKO_BOBOT)
            norm_risiko_score = raw_risiko_score * 100

            # 4. Proses Fuzzy Logic
            simulasi.input['umur'] = age
            simulasi.input['gejala'] = norm_gejala_score
            simulasi.input['risiko'] = norm_risiko_score
            
            simulasi.compute()
            final_score = simulasi.output['diagnosa']

            # 5. Tentukan Label
            status = "TERKENA LUNG CANCER" if final_score >= 55 else "TIDAK TERKENA"
            css_class = "danger" if final_score >= 55 else "success"

            result = {
                'score': round(final_score, 2),
                'status': status,
                'css_class': css_class,
                'details': {
                    'age': int(age),
                    'gejala_score': round(norm_gejala_score, 2),
                    'risiko_score': round(norm_risiko_score, 2)
                }
            }

        except Exception as e:
            result = {'error': str(e)}

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)