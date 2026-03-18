import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# =========================
# Step 1: Define XRD simulation
# =========================

# x-axis (2θ)
x = np.linspace(10, 60, 500)

def generate_xrd(peaks):
    y = np.zeros_like(x)
    
    for p in peaks:
        # random shift
        p_shifted = p + np.random.normal(0, 0.5)
        
        # random intensity
        height = np.random.uniform(0.8, 1.2)
        
        y += height * np.exp(-(x - p_shifted)**2 / (2 * 1.5**2))
    
    # add noise
    y += np.random.normal(0, 0.02, size=x.shape)
    
    return y

# =========================
# Step 2: Create Dataset
# =========================

X = []
y = []

phases = {
    0: [20, 30, 40],   # Phase A
    1: [22, 32, 42],   # Phase B
    2: [25, 35, 45]    # Phase C
}

for label, peaks in phases.items():
    for _ in range(100):
        pattern = generate_xrd(peaks)
        X.append(pattern)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)

# =========================
# Step 3: Train Model
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# =========================
# Step 4: Evaluate Model
# =========================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# =========================
# Step 5: Confusion Matrix
# =========================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")
plt.show()