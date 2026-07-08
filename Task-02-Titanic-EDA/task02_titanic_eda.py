import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# 1. LOAD DATA
url = ('https://raw.githubusercontent.com/'
       'datasciencedojo/datasets/master/titanic.csv')
df = pd.read_csv(url)
print("DATASET OVERVIEW")
print(f"Shape  : {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic stats:\n{df.describe()}")
# 2. DATA CLEANING
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Cabin_Known'] = df['Cabin'].notna().astype(int)
df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['AgeGroup'] = pd.cut(df['Age'],
                         bins=[0, 12, 18, 35, 60, 100],
                         labels=['Child','Teen','Adult','Middle-Age','Senior'])

print("AFTER CLEANING")
print(f"Shape  : {df.shape}")
print(f"Missing: {df.isnull().sum().sum()}")
# 3. KEY STATS
print("KEY INSIGHTS")
print(f"Overall survival rate : {df['Survived'].mean()*100:.1f}%")
print(f"Survival by gender    :\n{df.groupby('Sex')['Survived'].mean()*100}")
print(f"Survival by class     :\n{df.groupby('Pclass')['Survived'].mean()*100}")
print(f"Survival by embarked  :\n{df.groupby('Embarked')['Survived'].mean()*100}")
# 4. VISUALISATION SETUP
DARK  = '#0D1B2A'
BLUE  = '#1E5F9E'
TEAL  = '#2ABFBF'
CORAL = '#E8553E'
GOLD  = '#F4C430'
WHITE = '#FFFFFF'
sns.set_theme(style='darkgrid', rc={
    'figure.facecolor': DARK, 'axes.facecolor': '#162032',
    'axes.edgecolor': '#2a3a50', 'grid.color': '#1e2e42',
    'text.color': WHITE, 'axes.labelcolor': WHITE,
    'xtick.color': WHITE, 'ytick.color': WHITE,
    'axes.titlecolor': WHITE
})

# 5. FIGURE 1 — Overview
fig = plt.figure(figsize=(18, 6), facecolor=DARK)
fig.suptitle('PRODIGY INFOTACH — DATA SCIENCE TASK 02',
             fontsize=22, fontweight='bold', color=WHITE, y=1.02)

gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
# Missing values bar chart
ax = fig.add_subplot(gs[0, 0])
orig = pd.Series({'Age': 177, 'Cabin': 687, 'Embarked': 2})
bars = ax.barh(orig.index, orig.values, color=[CORAL, BLUE, TEAL], height=0.5)
for b, v in zip(bars, orig.values):
    ax.text(v+5, b.get_y()+b.get_height()/2, str(v),
            va='center', color=WHITE, fontsize=11, fontweight='bold')
ax.set_title('Missing Values (Before Cleaning)', fontsize=13, pad=8)
ax.set_xlabel('Count'); ax.invert_yaxis()

# Survival by Pclass
ax = fig.add_subplot(gs[0, 1])
ps = df.groupby('Pclass')['Survived'].mean() * 100
ax.bar(['1st','2nd','3rd'], ps.values, color=[GOLD, TEAL, CORAL], width=0.5)
for i, v in enumerate(ps.values):
    ax.text(i, v+1, f'{v:.1f}%', ha='center', color=WHITE, fontweight='bold')
ax.set_title('Survival Rate by Class', fontsize=13, pad=8)
ax.set_ylabel('Survival Rate (%)'); ax.set_ylim(0, 85)
# Survival by gender
ax = fig.add_subplot(gs[0, 2])
ss = df.groupby('Sex')['Survived'].mean() * 100
ax.bar(ss.index, ss.values, color=[TEAL, CORAL], width=0.4)
for i, (k, v) in enumerate(ss.items()):
    ax.text(i, v+1, f'{v:.1f}%', ha='center', color=WHITE, fontweight='bold', fontsize=12)
ax.set_title('Survival Rate by Gender', fontsize=13, pad=8)
ax.set_ylabel('Survival Rate (%)'); ax.set_ylim(0, 85)
plt.savefig('titanic_v2_1.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.show()
# 6. FIGURE 2 — Patterns & Relationships
fig2 = plt.figure(figsize=(12, 6), facecolor=DARK)
gs2 = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.35)

# Age group stacked bar
ax = fig2.add_subplot(gs2[0, 0])
ag = df.groupby('AgeGroup')['Survived'].value_counts(normalize=True).unstack() * 100
ag.columns = ['Did Not Survive','Survived']
ag[['Did Not Survive','Survived']].plot(kind='bar', ax=ax,
    color=[CORAL, TEAL], width=0.55, stacked=True)
ax.set_title('Survival by Age Group', fontsize=13, pad=8)
ax.set_xlabel('Age Group'); ax.set_ylabel('%')
ax.tick_params(axis='x', rotation=30); ax.legend(fontsize=9, loc='lower right')

# Family size line chart
ax = fig2.add_subplot(gs2[0, 1])
fs = df.groupby('FamilySize')['Survived'].mean() * 100
ax.plot(fs.index, fs.values, color=TEAL, lw=2.5,
        marker='o', markersize=8, markerfacecolor=GOLD)
ax.fill_between(fs.index, fs.values, alpha=0.15, color=TEAL)
ax.set_title('Survival Rate by Family Size', fontsize=13, pad=8)
ax.set_xlabel('Family Size'); ax.set_ylabel('Survival Rate (%)')
ax.set_xticks(fs.index)

plt.savefig('titanic_v2_2.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.show()

