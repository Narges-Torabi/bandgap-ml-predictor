import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


np.random.seed(42)
n_samples = 600


electronegativity_diff = np.random.uniform(0.1, 2.5, n_samples)
atomic_radius_ratio = np.random.uniform(0.5, 1.8, n_samples)
valence_electron_sum = np.random.choice([4, 8, 10, 16], n_samples)
density_g_cm3 = np.random.uniform(2.0, 9.5, n_samples)


true_bandgap = (
    0.8 * electronegativity_diff
    + 1.2 / atomic_radius_ratio
    + 0.05 * valence_electron_sum
    + np.random.normal(0, 0.25, n_samples)
)
true_bandgap = np.clip(true_bandgap, 0.0, 6.0)

df = pd.DataFrame({
    'electronegativity_diff': electronegativity_diff,
    'atomic_radius_ratio': atomic_radius_ratio,
    'valence_electron_sum': valence_electron_sum,
    'density_g_cm3': density_g_cm3,
    'bandgap_eV': true_bandgap
})


X = df.drop(columns=['bandgap_eV'])
y = df['bandgap_eV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.3f} eV")
print(f"R² Score: {r2:.3f}")


plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color='#1f77b4', edgecolors='k')
plt.plot([0, 6], [0, 6], '--r', linewidth=2, label='Ideal Fit')
plt.xlabel('DFT / True Bandgap (eV)')
plt.ylabel('ML Predicted Bandgap (eV)')
plt.title(f'Bandgap Prediction (R² = {r2:.2f}, RMSE = {rmse:.2f} eV)')
plt.legend()
plt.tight_layout()
plt.savefig('bandgap_parity_plot.png', dpi=300)
plt.show()


feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(7, 3.5))
feature_importance.plot(kind='barh', color='#2ca02c')
plt.title('Physical Feature Importances in Bandgap Prediction')
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()