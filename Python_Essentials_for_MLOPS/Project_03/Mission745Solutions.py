import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


logging.basicConfig(level=logging.INFO)
np.random.seed(42)

sns.set_style('whitegrid')


customers = pd.read_csv('./customer_segmentation.csv')

columns_list = ['gender', 'education_level', 'marital_status']
for col in columns_list:
    print(col)
    print(customers[col].value_counts(), end='\n\n')


fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(round(customers.drop('customer_id', axis=1).corr(), 2),
            cmap='Blues', annot=True, ax=ax)

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(12, 10))

# Removing the customer's id before plotting the distributions
customers.drop('customer_id', axis=1).hist(ax=ax)

plt.tight_layout()
plt.show()


customers_modif = customers.copy()
customers_modif['gender'] = customers['gender'].apply(lambda x: 1 if x == 'M' else 0)
customers_modif.head()


customers_modif.replace(to_replace={'Uneducated': 0,
                                    'High School': 1,
                                    'College': 2,
                                    'Graduate': 3,
                                    'Post-Graduate': 4,
                                    'Doctorate': 5}, inplace=True)

customers_modif['education_level'].head()

dummies = pd.get_dummies(customers_modif[['marital_status']], drop_first=True)

customers_modif = pd.concat([customers_modif, dummies], axis=1)
customers_modif.drop(['marital_status'], axis=1, inplace=True)

# print(customers_modif.shape)
customers_modif.head()


X = customers_modif.drop('customer_id', axis=1)

scaler = StandardScaler()

logging.info("starting training")
scaler.fit(X)
logging.info("training completed")

X_scaled = scaler.transform(X)

X = pd.DataFrame(X_scaled)
inertias = []

for k in range(1, 11):
    model = KMeans(n_clusters=k)
    y = model.fit_predict(X)
    inertias.append(model.inertia_)

plt.figure(figsize=(12, 8))
plt.plot(range(1, 11), inertias, marker='o')
plt.xticks(ticks=range(1, 11), labels=range(1, 11))
plt.title('Inertia vs Number of Clusters')

plt.tight_layout()
plt.show()

model = KMeans(n_clusters=6)
logging.info("starting prediction")
y = model.fit_predict(X_scaled)
logging.info("prediction completed")
customers['CLUSTER'] = y + 1


numeric_columns = customers.select_dtypes(include=np.number)
numeric_columns = numeric_columns.drop(
                    ['customer_id', 'CLUSTER'], axis=1).columns
fig = plt.figure(figsize=(20, 20))
for i, column in enumerate(numeric_columns):
    df_plot = customers.groupby('CLUSTER')[column].mean()
    ax = fig.add_subplot(5, 2, i+1)
    ax.bar(df_plot.index, df_plot, color=sns.color_palette('Set1'), alpha=0.6)
    ax.set_title(f'Average {column.title()} per Cluster', alpha=0.5)
    ax.xaxis.grid(False)

plt.tight_layout()
plt.show()


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
sns.scatterplot(x='age', y='months_on_book', hue='CLUSTER',
                data=customers, palette='tab10', alpha=0.4, ax=ax1)

sns.scatterplot(x='estimated_income', y='credit_limit',
                hue='CLUSTER', data=customers, palette='tab10',
                alpha=0.4, ax=ax2, legend=False)

sns.scatterplot(x='credit_limit', y='avg_utilization_ratio',
                hue='CLUSTER', data=customers, palette='tab10',
                alpha=0.4, ax=ax3)

sns.scatterplot(x='total_trans_count', y='total_trans_amount',
                hue='CLUSTER', data=customers, palette='tab10',
                alpha=0.4, ax=ax4, legend=False)

plt.tight_layout()
plt.show()


cat_columns = customers.select_dtypes(include=['object'])

fig = plt.figure(figsize=(18, 6))
for i, col in enumerate(cat_columns):
    plot_df = pd.crosstab(index=customers['CLUSTER'], columns=customers[col],
                          values=customers[col], aggfunc='size', normalize='index')
    ax = fig.add_subplot(1, 3, i+1)
    plot_df.plot.bar(stacked=True, ax=ax, alpha=0.6)
    ax.set_title(f'% {col.title()} per Cluster', alpha=0.5)

    ax.set_ylim(0, 1.4)
    ax.legend(frameon=False)
    ax.xaxis.grid(False)

    labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ax.set_yticklabels(labels)

plt.tight_layout()
plt.show()
