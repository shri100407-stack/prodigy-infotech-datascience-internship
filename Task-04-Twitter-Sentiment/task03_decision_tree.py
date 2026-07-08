import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset 
df = pd.read_csv('D:/internship/task 3/bank-additional-full.csv', sep=';')
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nTarget Distribution:")
print(df['y'].value_counts())

#  Encode Categorical Columns 
le = LabelEncoder()
cat_cols = df.select_dtypes(include='str').columns.tolist()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("\nAfter encoding:")
print(df.head())

# Split Features and Target
X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\nTraining size: {X_train.shape}")
print(f"Testing size : {X_test.shape}")

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


#  COMBINED FIGURE — like Task 02
BG   = "#0F0F1A"
CARD = "#1A1A2E"
P1   = "#7C3AED"
P2   = "#06B6D4"
P3   = "#F59E0B"
TEXT = "#E2E8F0"
SUB  = "#94A3B8"

fig = plt.figure(figsize=(22, 14), facecolor=BG)

# Main Title
fig.suptitle("PRODIGY INFOTECH — DATA SCIENCE TASK 03",
             fontsize=22, fontweight='bold', color=TEXT, y=0.98)

# Grid Layout: top row 3 charts, bottom row 2 charts 
gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.45, wspace=0.35,
                       top=0.92, bottom=0.07,
                       left=0.06, right=0.97)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1:])  # spans last 2 columns

for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.set_facecolor(CARD)
    ax.tick_params(colors=SUB)
    ax.xaxis.label.set_color(SUB)
    ax.yaxis.label.set_color(SUB)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2D2D4A')

# Chart 1: Target Distribution 
counts = df['y'].value_counts().sort_index()
bars = ax1.bar(['No', 'Yes'], counts.values,
               color=[P1, P2], width=0.4, edgecolor='none')
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f'{val:,}', ha='center', fontsize=11,
             fontweight='bold', color=TEXT)
ax1.set_title('Will the Customer Subscribe?\n(Target Variable)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Subscribed to Term Deposit', fontsize=10)
ax1.set_ylabel('Count', fontsize=10)
ax1.set_ylim(0, counts.max() * 1.18)
ax1.grid(axis='y', alpha=0.3)
ax1.spines[:].set_visible(False)

#  Chart 2: Age Distribution 
ax2.hist(df['age'], bins=30, color=P1, edgecolor=BG, linewidth=0.5)
ax2.set_title('Age Distribution of Customers', fontsize=12, fontweight='bold')
ax2.set_xlabel('Age', fontsize=10)
ax2.set_ylabel('Count', fontsize=10)
ax2.grid(axis='y', alpha=0.3)
ax2.spines[:].set_visible(False)

#  Chart 3: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
            linewidths=2, linecolor=BG, ax=ax3, cbar=False,
            annot_kws={"size": 14, "weight": "bold"})
ax3.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
ax3.set_xlabel('Predicted', fontsize=10)
ax3.set_ylabel('Actual', fontsize=10)

# Chart 4: Feature Importance 
fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
fi_top = fi.tail(10)
colors = [P2 if v == fi_top.max() else P1 for v in fi_top.values]
ax4.barh(fi_top.index, fi_top.values, color=colors, edgecolor='none', height=0.6)
ax4.set_title('Top 10 Important Features', fontsize=12, fontweight='bold')
ax4.set_xlabel('Importance Score', fontsize=10)
ax4.grid(axis='x', alpha=0.3)
ax4.spines[:].set_visible(False)

#  Chart 5: Decision Tree 
plot_tree(model, max_depth=3, feature_names=X.columns.tolist(),
          class_names=['No', 'Yes'], filled=True, rounded=True,
          fontsize=8, ax=ax5, proportion=False, impurity=True)
ax5.set_title('Decision Tree Visualization (Top 3 Levels Shown)',
              fontsize=12, fontweight='bold', color=TEXT)
ax5.set_facecolor(CARD)

# Save 
plt.savefig('D:/internship/task 3/task03_charts.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.show()

print(f"   Model Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"   Tree Max Depth : {model.get_depth()}")
