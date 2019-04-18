import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

K = 50

X_train = np.fromfile("train_embedding.txt", dtype=np.float32).reshape(-1, 4096)
X_test = np.fromfile("test_embedding.txt", dtype=np.float32).reshape(-1, 4096)

train_df = pd.read_csv("training_triplet_sample.csv")
test_df = pd.read_csv("test_triplet_sample.csv")

y_test = test_df["query"].apply(lambda x:x.split("/")[2])
y_train = train_df["query"].apply(lambda x:x.split("/")[2])

knn_model = KNeighborsClassifier(n_neighbors=K, weights='distance', algorithm='kd_tree', n_jobs=-1)
knn_model.fit(X=X_train, y=y_train)

print(knn_model.predict_proba(X_test))