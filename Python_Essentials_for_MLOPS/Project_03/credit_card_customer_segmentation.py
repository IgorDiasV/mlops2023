"""
code for customer segmentation.
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

PATH = 'Python_Essentials_for_MLOPS/Project_03/'

logging.basicConfig(level=logging.INFO)
np.random.seed(42)

sns.set_style('whitegrid')


def show_scartplot():
    """function to group scartplot graphs"""
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
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


def visualize_data_correlation(data):
    """ show a heatmap with of correlation"""
    _, axes = plt.subplots(figsize=(12, 8))
    sns.heatmap(round(data.drop('customer_id', axis=1).corr(), 2),
                cmap='Blues',
                annot=True,
                ax=axes)
    plt.tight_layout()
    plt.show()


def load_file():
    """ load file used in the code"""
    df_customer = pd.read_csv(PATH + 'customer_segmentation.csv')
    return df_customer


def preprocess_customer_data(data):
    """ function to process customer data"""
    data['gender'] = data['gender'].apply(lambda x: 1 if x == 'M' else 0)

    data.replace(to_replace={'Uneducated': 0,
                             'High School': 1,
                             'College': 2,
                             'Graduate': 3,
                             'Post-Graduate': 4,
                             'Doctorate': 5}, inplace=True)

    data['education_level'].head()

    dummies = pd.get_dummies(data[['marital_status']], drop_first=True)

    data = pd.concat([data, dummies], axis=1)
    data.drop(['marital_status'], axis=1, inplace=True)

    data = data.drop('customer_id', axis=1)

    scaler = StandardScaler()

    logging.info("starting training")
    scaler.fit(data)
    logging.info("training completed")

    data_scaled = scaler.transform(data)

    preprocess_data = pd.DataFrame(data_scaled)

    return preprocess_data.copy(), data_scaled.copy()


def find_optimal_clusters(data):
    """Find the optimal number of clusters"""
    inertias = []
    for k in range(1, 11):
        model = KMeans(n_clusters=k)
        _ = model.fit_predict(data)
        inertias.append(model.inertia_)

    return inertias


def visualize_inertia_n_clusters(inertias):
    """shows a graph of the relationship in number of clusters and Inertia"""
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, 11), inertias, marker='o')
    plt.xticks(ticks=range(1, 11), labels=range(1, 11))
    plt.title('Inertia vs Number of Clusters')

    plt.tight_layout()
    plt.show()


def train_kmeans_model(data_scaled):
    """train the model"""
    model = KMeans(n_clusters=6)
    logging.info("starting prediction")
    train_model = model.fit_predict(data_scaled)
    logging.info("prediction completed")

    return train_model


def visualize_cluster_data(data):
    """ show the cluster data"""
    numeric_columns = data.select_dtypes(include=np.number)
    numeric_columns = numeric_columns.drop(
                        ['customer_id', 'CLUSTER'], axis=1).columns
    fig = plt.figure(figsize=(20, 20))
    for i, column in enumerate(numeric_columns):
        df_plot = data.groupby('CLUSTER')[column].mean()
        axes = fig.add_subplot(5, 2, i+1)
        axes.bar(df_plot.index, df_plot, color=sns.color_palette('Set1'), alpha=0.6)
        axes.set_title(f'Average {column.title()} per Cluster', alpha=0.5)
        axes.xaxis.grid(False)

    plt.tight_layout()
    plt.show()

    show_scartplot()

    cat_columns = customers.select_dtypes(include=['object'])

    fig = plt.figure(figsize=(18, 6))
    for i, col in enumerate(cat_columns):
        plot_df = pd.crosstab(index=customers['CLUSTER'], columns=customers[col],
                              values=customers[col], aggfunc='size', normalize='index')
        axes = fig.add_subplot(1, 3, i+1)
        plot_df.plot.bar(stacked=True, ax=axes, alpha=0.6)
        axes.set_title(f'% {col.title()} per Cluster', alpha=0.5)

        axes.set_ylim(0, 1.4)
        axes.legend(frameon=False)
        axes.xaxis.grid(False)

        labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
        axes.set_yticklabels(labels)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    customers = load_file()
    visualize_data_correlation(customers.copy())

    _, axle = plt.subplots(figsize=(12, 10))
    customers.drop('customer_id', axis=1).hist(ax=axle)
    plt.tight_layout()
    plt.show()

    customers_modif = customers.copy()
    preprocessed_data, preprocess_data_scaled = preprocess_customer_data(customers_modif)
    inertias_list = find_optimal_clusters(preprocessed_data.copy())

    visualize_inertia_n_clusters(inertias_list)
    trained_model = train_kmeans_model(preprocess_data_scaled)
    customers['CLUSTER'] = trained_model + 1

    visualize_cluster_data(customers)
