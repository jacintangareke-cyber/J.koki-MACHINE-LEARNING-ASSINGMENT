from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# prepare customer data
customers = [
    [1000,2],
    [1500,3],
    [2000,2],
    [5000,6],
    [6000,7],
    [5500,5],
    [12000,12],
    [15000,14],
    [13000,11],
    [3000,4]
]

# scale the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customers)

# create a KMeans model
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

# train the model
clusters = model.fit_predict(scaled_data)

# display each customer with their cluster assignment
for i, customer in enumerate(customers):
    print(f"Customer {i+1}: {customer} -> Cluster {clusters}")

# display the number of customers in each cluster
print("\nNumber of customers in each cluster:")
for cluster in range(3):
    count = sum(1 for value in clusters if value == cluster)
    print(f"Cluster {cluster}: {count} customers")

# plot the customer groups
for cluster in range(3):
    cluster_customers = [customers[i] for i in range(len(customers)) if clusters[i] == cluster]
    plt.scatter(
        [customer[0] for customer in cluster_customers],
        [customer[1] for customer in cluster_customers],
        label=f'Cluster {cluster}'
    )

    plt.xlabel("Monthly spending")
    plt.ylabel("Monthly visits")
    plt.title("Customer Segmentation Using KMeans")
    plt.legend()
    plt.show()