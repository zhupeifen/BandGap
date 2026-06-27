import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from pymatgen.core import Lattice, Structure, Composition
from pymatgen.core.periodic_table import Element, Specie
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from math import sqrt
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
PEROVSKITE_DATA_PATH = str(DATA_DIR / 'Tol_screened_ensemble_final.csv')
ELEMENTAL_PROPERTIES_PATH = str(DATA_DIR / 'Elemental_properties.xlsx')

df = pd.read_csv(PEROVSKITE_DATA_PATH)
data_len = len(df)

element_data = pd.read_excel(ELEMENTAL_PROPERTIES_PATH)

rb_index = 0
ge_index = 0
sn_index = 0
sr_index = 0
cl_index = 0

for i in range(len(element_data)):
    if element_data['M'][i] == 'Rb':
        rb_index = i
    if element_data['M'][i] == 'Ge':
        ge_index = i
    if element_data['M'][i] == 'Sn':
        sn_index = i
    if element_data['M'][i] == 'Sr':
        sr_index = i
    if element_data['M'][i] == 'Cl':
        cl_index = i

# testing with Rb Ge0.125 Sn0.250 Sr0.625 Cl3

# Electronegativity
rb = Element('Rb')
rb_ion = Specie('Rb', 1) # Rb+1

ge = Element('Ge')
sn = Element('Sn')
sr = Element('Sr')

ge_ion = Specie('Ge', 2) # Ge+2
sn_ion = Specie('Sn', 2) # Sn+2
sr_ion = Specie('Sr', 2) # Sr+2

cl = Element('Cl')
cl_ion = Specie('Cl', -1) # Cl-1

ge_ratio = 0.625
sn_ratio = 0.250
sr_ratio = 0.125

A_En = rb.X
B_En = ge_ratio*ge.X + sn_ratio*sn.X + sr_ratio*sr.X
X_En = cl.X

print('electronegativity')
print(A_En)
print(B_En)
print(X_En)

# Sn 2+ ionic radius = 1.18 (looked up)

A_ion_rad = element_data['Ion_rad'][rb_index]
B_ion_rad = ge_ratio*element_data['Ion_rad'][ge_index] + sn_ratio*element_data['Ion_rad'][sn_index] + sr_ratio*element_data['Ion_rad'][sr_index]
X_ion_rad = element_data['Ion_rad'][cl_index]
B_ion_rad = 0.99

print('ionic radius')
print(A_ion_rad)
print(B_ion_rad)
print(X_ion_rad)

A_MP = rb.melting_point
B_MP = ge_ratio*ge.melting_point + sn_ratio*sn.melting_point + sr_ratio*sr.melting_point
X_MP = cl.melting_point

A_BP = rb.boiling_point
B_BP = ge_ratio*ge.boiling_point + sn_ratio*sn.boiling_point + sr_ratio*sr.boiling_point
X_BP = cl.boiling_point

A_density = element_data['Density'][rb_index]
B_density = ge_ratio*element_data['Density'][ge_index] + sn_ratio*element_data['Density'][sn_index] + sr_ratio*element_data['Density'][sr_index]
X_density = element_data['Density'][cl_index]

print('melting point')
print(A_MP)
print(B_MP)
print(X_MP)

print('boiling point')
print(A_BP)
print(B_BP)
print(X_BP)

print('density')
print(A_density)
print(B_density)
print(X_density)


A_atomic_mass = rb.atomic_mass
B_atomic_mass = ge.atomic_mass*ge_ratio + sn.atomic_mass*sn_ratio + sr.atomic_mass*sr_ratio
X_atomic_mass = cl.atomic_mass

print('atomic mass')
print(A_atomic_mass)
print(B_atomic_mass)
print(X_atomic_mass)


A_EA = element_data['Elec_Aff'][rb_index]
B_EA = element_data['Elec_Aff'][ge_index]*ge_ratio + element_data['Elec_Aff'][sn_index]*sn_ratio + element_data['Elec_Aff'][sr_index]*sr_ratio
X_EA = element_data['Elec_Aff'][cl_index]


A_IE = element_data['Ion_Energy'][rb_index]
B_IE = element_data['Ion_Energy'][ge_index]*ge_ratio + element_data['Ion_Energy'][sn_index]*sn_ratio + element_data['Ion_Energy'][sr_index]*sr_ratio
X_IE = element_data['Ion_Energy'][cl_index]


print('electron affinity')
print(A_EA)
print(B_EA)
print(X_EA)

print('ionization energy')
print(A_IE)
print(B_IE)
print(X_IE)


# Couldn't find heat of fusion or heat of vaporization :(
# and I don't know about the period

A_hof = element_data['Heat_fusion'][rb_index]
B_hof = ge_ratio*element_data['Heat_fusion'][ge_index] + sn_ratio*element_data['Heat_fusion'][sn_index] + sr_ratio*element_data['Heat_fusion'][sr_index]
X_hof = element_data['Heat_fusion'][cl_index]

A_hov = element_data['Heat_vap'][rb_index]
B_hov = ge_ratio*element_data['Heat_vap'][ge_index] + sr_ratio*element_data['Heat_vap'][sr_index] + sn_ratio*element_data['Heat_vap'][sn_index]
X_hov = element_data['Heat_vap'][cl_index]

A_period = element_data['Period'][rb_index]
B_period = ge_ratio*element_data['Period'][ge_index] + sr_ratio*element_data['Period'][sr_index] + sn_ratio*element_data['Period'][sn_index]
X_period = element_data['Period'][cl_index]

A_at_num = rb.Z
B_at_num = ge_ratio*ge.Z + sn_ratio*sn.Z + sr_ratio*sr.Z
X_at_num = cl.Z

print('atomic number')
print(A_at_num)
print(B_at_num)
print(X_at_num)

t = (A_ion_rad + X_ion_rad) / (sqrt(2) * (B_ion_rad+X_ion_rad))
o = B_ion_rad/X_ion_rad
tao = (A_ion_rad + X_ion_rad) / (B_ion_rad + X_ion_rad)

print('t: ' + str(t))
print('o: ' + str(o))
print('tao: ' + str(tao))

# Ge, Sn, Pb are all +2 or +4
# Ca, Sr, Ba should be +2
# K, Rb, Cs should all be +1
# Cl, Br, I should all be -1

titles = ['K', 'Rb', 'Cs', 'MA', 'FA', 'Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb', 'Cl', 'Br', 'I', 'A_ion_rad', 'A_BP', 'A_MP', 'A_dens', 'A_at_wt', 'A_EA', 'A_IE', 'A_hof', 'A_hov', 'A_En', 'A_at_num', 'A_period', 'B_ion_rad', 'B_BP', 'B_MP', 'B_dens', 'B_at_wt', 'B_EA', 'B_IE', 'B_hof', 'B_hov', 'B_En', 'B_at_num', 'B_period', 'X_ion_rad', 'X_BP', 'X_MP', 'X_dens', 'X_at_wt', 'X_EA', 'X_IE', 'X_hof', 'X_hov', 'X_En', 'X_at_num', 'X_period', 't', 'o', 'tao', 'Cubic', 'Tetra', 'Ortho', 'Hex', 'PBE', 'HSE', 'Expt', 'Decomp(PBE)', 'Decomp(HSE)']
titles = ['K', 'Rb', 'Cs', 'MA', 'FA', 'Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb', 'Cl', 'Br', 'I', 'A_ion_rad', 'A_BP', 'A_MP', 'A_dens', 'A_at_wt', 'A_EA', 'A_IE', 'A_hof', 'A_hov', 'A_En', 'A_at_num', 'A_period', 'B_ion_rad', 'B_BP', 'B_MP', 'B_dens', 'B_at_wt', 'B_EA', 'B_IE', 'B_hof', 'B_hov', 'B_En', 'B_at_num', 'B_period', 'X_ion_rad', 'X_BP', 'X_MP', 'X_dens', 'X_at_wt', 'X_EA', 'X_IE', 'X_hof', 'X_hov', 'X_En', 'X_at_num', 'X_period', 't', 'o', 'tao', 'Cubic', 'Tetra', 'Ortho', 'Hex']
X = np.zeros((data_len, len(titles)))
y = np.zeros((data_len))

test_entry = [0, 1, 0, 0, 0, 0, sr_ratio, 0, ge_ratio, sn_ratio, 0, 1, 0, 0, A_ion_rad, A_BP, A_MP, A_density, A_atomic_mass, A_EA, A_IE, A_hof, A_hov, A_En, A_at_num, A_period, B_ion_rad, B_BP, B_MP, B_density, B_atomic_mass, B_EA, B_IE, B_hof, B_hov, B_En, B_at_num, B_period, X_ion_rad, X_BP, X_MP, X_density, X_atomic_mass, X_EA, X_IE, X_hof, X_hov, X_En, X_at_num, X_period, t, o, tao, 1, 0, 0, 0]

i = 0
for j, title in enumerate(titles):
    diff = 0
    if df[title][i] != 0:
        diff = (test_entry[j] - df[title][i]) / df[title][i]
    
    # greater than 2% difference
    if abs(diff)*100 > 2:
        print('bad number:')
        print(title)
        print(str(diff*100)+'%')
        print(test_entry[j])
        print(df[title][i])
    
    #test_entry[j] = df[title][i]

###
### BIGGEST FACTORS: 'Cubic', 'Tetra', 'Ortho', 'Hex'
###

for i in range(len(df)):
    
    for j, title in enumerate(titles):
        X[i, j] = df[title][i]
    
    y[i] = df['Band gap(HSE-mf1)'][i]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=46
)
model.fit(X_train, y_train)
model.save_model(str(Path(__file__).resolve().parent / 'perovskite_fractions_model.json'))

predictions = model.predict(X_test)

total, correct = 0, 0
arr = np.zeros((len(y_test)))
for i in range(len(y_test)):
    if abs(predictions[i] - y_test[i]) < 0.07:
        correct += 1
        arr[i] = 1
        
    total += 1

print('accuracy:')
print(correct/total)

# --- Generalization check -------------------------------------------------
# The random-split accuracy above is optimistic: this dataset is a dense grid
# (16,979 compositions x 4 structures), so a random split can put near-identical
# compositions in both train and test. Re-evaluate under harder, leakage-
# controlled splits that hold out whole compositions, then whole chemistries.
# (Full analysis: scripts/tol_generalization_test.py)
site_cols = ['K', 'Rb', 'Cs', 'MA', 'FA', 'Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb',
             'Cl', 'Br', 'I']
sites = df[site_cols].apply(pd.to_numeric, errors='coerce')
group_comp = sites.round(4).astype(str).agg('|'.join, axis=1).to_numpy()
group_chem = (sites > 0).astype(int).astype(str).agg(''.join, axis=1).to_numpy()

print('\ngeneralization (acc<0.07 eV / MAE eV):')
print('  random split          : %.3f / %.4f'
      % (correct / total, np.mean(np.abs(predictions - y_test))))
for name, groups in [('held-out compositions', group_comp),
                     ('held-out chemistries ', group_chem)]:
    gtr, gte = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                      random_state=46).split(X, y, groups))
    gm = xgb.XGBRegressor(n_estimators=300, max_depth=9, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=46)
    gm.fit(X[gtr], y[gtr])
    ge = np.abs(gm.predict(X[gte]) - y[gte])
    print('  %s: %.3f / %.4f' % (name, np.mean(ge < 0.07), ge.mean()))
# --------------------------------------------------------------------------

print(model.predict([test_entry]))

data = {'predictions': predictions, 'test data': y_test, 'difference (test data - ai prediction)': y_test-predictions,
        'index': np.linspace(0, len(predictions), len(predictions)), 'accuracy': arr}
data = pd.DataFrame(data)

sns.set_theme()

# Create a visualization
sns.relplot(
    data=data,
    x='index', y='difference (test data - ai prediction)', hue='test data', style='accuracy'
)

plt.show()