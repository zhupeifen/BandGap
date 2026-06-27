from matminer.datasets import load_dataset
from matminer.utils.io import load_dataframe_from_json

import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pymatgen.core import Element

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

df = load_dataframe_from_json(str(DATA_DIR / 'wolverton_oxides.json'))

non_zero_values = []
bad_nums = [32, 35, 85, 92, 170, 459, 593, 666, 667, 679, 682, 686, 692, 693, 704, 723, 785, 817, 845, 876, 880, 882, 884, 885, 886, 889, 890, 894, 896, 900, 901, 902, 903, 904, 905, 908, 913, 915, 916, 917, 918, 920, 921, 923, 924, 926, 927, 931, 932, 937, 942, 944, 946, 947, 953, 955, 957, 958, 963, 981, 991, 996, 1009, 1034, 1035, 1041, 1047, 1055, 1060, 1062, 1069, 1080, 1104, 1180, 1182, 1218, 1255, 1260, 1387, 1393, 1395, 1396, 1401, 1409, 1413, 1415, 1418, 1419, 1422, 1424, 1426, 1428, 1429, 1430, 1432, 1434, 1435, 1438, 1439, 1440, 1442, 1443, 1449, 1453, 1454, 1455, 1456, 1502, 1545, 1547, 1615, 1616, 1647, 1664, 1688, 1693, 1698, 1705, 1721, 1833, 1839, 1912, 1917, 1934, 1939, 1956, 1958, 1964, 1985, 2119, 2131, 2156, 2166, 2172, 2180, 2207, 2277, 2336, 2348, 2418, 2423, 2496, 2524, 2529, 2637, 2642, 2644, 2667, 2675, 2715, 2743, 2776, 2778, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2795, 2796, 2799, 2800, 2801, 2802, 2803, 2804, 2805, 2806, 2809, 2811, 2812, 2813, 2814, 2815, 2816, 2818, 2819, 2820, 2821, 2822, 2825, 2827, 2829, 2830, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2843, 2844, 2845, 2853, 2855, 2856, 2857, 2858, 2861, 2863, 2865, 2870, 2872, 2873, 2875, 2876, 2878, 2884, 2886, 2891, 2892, 2894, 2895, 2896, 2902, 2912, 2913, 2916, 2929, 2934, 2953, 2967, 2998, 2999, 3001, 3002, 3004, 3005, 3007, 3010, 3012, 3015, 3019, 3021, 3027, 3030, 3032, 3034, 3040, 3041, 3048, 3049, 3059, 3082, 3089, 3102, 3105, 3107, 3113, 3124, 3129, 3132, 3134, 3138, 3178, 3226, 3254, 3297, 3299, 3326, 3327, 3372, 3404, 3431, 3432, 3434, 3435, 3436, 3437, 3439, 3440, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 3452, 3453, 3455, 3456, 3457, 3458, 3459, 3460, 3461, 3462, 3464, 3466, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 3476, 3477, 3478, 3479, 3480, 3481, 3482, 3484, 3485, 3486, 3487, 3488, 3490, 3493, 3494, 3495, 3496, 3497, 3498, 3500, 3560, 3591, 3625, 3627, 3664, 3703, 3732, 3737, 3750, 3765, 3796, 3878, 3883, 3897, 4029, 4052, 4057, 4097, 4100, 4135, 4200, 4233, 4243, 4248, 4260, 4321, 4389, 4394, 4418, 4422, 4535, 4540, 4552, 4574, 4608, 4633, 4638, 4646, 4754, 4759, 4824, 4826, 4827, 4830, 4832, 4833, 4835, 4843, 4844, 4849, 4855, 4857, 4859, 4860, 4863, 4879, 4884, 4891, 4899, 4900, 4905, 4910]


data_len = len(df) - len(bad_nums)
X = np.zeros((data_len, 9))
y = np.zeros((data_len))
y_bin = np.zeros((data_len))

# This is how to get atomic number
elem = Element(df['atom a'][60])
atomic_number = elem.Z

print(data_len)

# Using vpa made it go down, so don't use that ╰*°▽°*╯

j = 0
for i in range(len(df)):
    if i in bad_nums:
        continue
    
    #print(df['atom a'][i] + ' ' + df['atom b'][i] + ' O3')
    
    X[j, 0] = Element(df['atom a'][i]).Z
    X[j, 1] = Element(df['atom b'][i]).Z
    
    X[j, 4] = df['a'][i]
    X[j, 5] = df['b'][i]
    X[j, 6] = df['c'][i]
    X[j, 7] = df['e_hull'][i]
    X[j, 8] = df['e_form oxygen'][i]
    
    mu_b = df['mu_b'][i]
    if mu_b == '-':
        X[j, 2] = 0
    else:
        X[j, 2] = df['mu_b'][i]
    X[j, 3] = df['e_form'][i]
    
    y[j] = df['gap pbe'][i]
    if df['gap pbe'][i] == 0:
        y_bin[j] = 0
    else:
        y_bin[j] = 1
        non_zero_values.append(i)
    
    j += 1

X_non_zero = np.zeros((len(non_zero_values), 9))
y_non_zero = np.zeros((len(non_zero_values)))


j = 0
for i in non_zero_values:
    
    X_non_zero[j, 0] = Element(df['atom a'][i]).Z
    X_non_zero[j, 1] = Element(df['atom b'][i]).Z
    
    X_non_zero[j, 4] = df['a'][i]
    X_non_zero[j, 5] = df['b'][i]
    X_non_zero[j, 6] = df['c'][i]
    X_non_zero[j, 7] = df['e_hull'][i]
    X_non_zero[j, 8] = df['e_form oxygen'][i]
    
    mu_b = df['mu_b'][i]
    if mu_b == '-':
        X_non_zero[j, 2] = 0
    else:
        X_non_zero[j, 2] = df['mu_b'][i]
    X_non_zero[j, 3] = df['e_form'][i]
    
    y_non_zero[j] = df['gap pbe'][i]
    j += 1


random_num = 390813498
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.2, random_state=random_num)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_num)

bin_model = xgb.XGBClassifier(
    n_estimators=42,
    max_depth=7,
    objective='binary:logistic',
    random_state=random_num
)
bin_model.fit(X_train_bin, y_train_bin)

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.07,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=random_num
)
model.fit(X_train, y_train)

bin_predictions = bin_model.predict(X_test)
predictions = model.predict(X_test)
real_predictions = np.zeros(len(y_test))

total, correct, non_zero_correct, non_zero = 0, 0, 0, 0
arr = np.zeros((len(y_test)))
for i in range(len(y_test)):
    
    prediction = bin_predictions[i] * predictions[i]
    real_predictions[i] = prediction
    
    if y_test_bin[i]:
        non_zero += 1
    
    if abs(prediction - y_test[i]) < 0.07:
        correct += 1
        arr[i] = 1
        if y_test_bin[i]:
            non_zero_correct += 1
    
    total += 1

print('accuracy:')
print(correct/total)

print('non zero accuracy')
print(non_zero_correct)
print(non_zero)
print(non_zero_correct/non_zero)

data = {'predictions': real_predictions, 'test data': y_test, 'difference (test data - ai prediction)': y_test-real_predictions,
        'index': np.linspace(0, len(predictions), len(predictions)), 'accuracy': arr}
data = pd.DataFrame(data)

sns.set_theme()

# Create a visualization
sns.relplot(
    data=data,
    x='index', y='difference (test data - ai prediction)', style='accuracy', hue='test data'
)

plt.show()
