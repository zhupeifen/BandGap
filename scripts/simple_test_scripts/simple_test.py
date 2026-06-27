import numpy as np
from xgboost import DMatrix

# Generate synthetic data
X = np.random.rand(100, 5)
y = np.random.randint(2, size=100)

print(y)

# Create DMatrix from NumPy arrays
dmatrix = DMatrix(data=X, label=y)

# Print DMatrix information
print(f"Number of rows: {dmatrix.num_row()}")
print(f"Number of columns: {dmatrix.num_col()}")