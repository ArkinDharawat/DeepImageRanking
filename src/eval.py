import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

K = 30
X_train = np.fromfile("train_embedding.txt", dtype=np.float32).reshape(-1, 4096)
X_test = np.fromfile("test_embedding.txt", dtype=np.float32).reshape(-1, 4096)

train_df = pd.read_csv("training_triplet_sample.csv")
test_df = pd.read_csv("test_triplet_sample.csv")

X_train = X_train[0::4]
train_df = train_df.loc[0::4, :].reset_index(drop=True)

y_test = test_df["query"].apply(lambda x: x.split("/")[2])[0:X_test.shape[0]]
y_train = train_df["query"].apply(lambda x: x.split("/")[2])[0:X_train.shape[0]]

knn_model = KNeighborsClassifier(n_neighbors=K, n_jobs=-1, p=2)
knn_model.fit(X=X_train, y=y_train)

_, indx = knn_model.kneighbors(X_test, n_neighbors=K)

n_indx = []
for i in xrange(0, len(indx)):
    n_indx.append([y_train[x] for x in indx[i]])

sum([1 if (y_test[i] in n_indx[i]) else 0 for i in range(0, len(indx))]) / float(len(indx))

# code for plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def show_images(arr, name=""):
    fig = plt.figure(figsize=(64, 64))
    columns = 11
    rows = 1
    ax = []
    for i in range(1, columns * rows + 1):
        dist, img = arr[i - 1]
        img = mpimg.imread(img)
        ax.append(fig.add_subplot(rows, columns, i))
        if i == 1:
            ax[-1].set_title("query image", fontsize=50)
        else:
            ax[-1].set_title("img_:" + str(i - 1), fontsize=50)
            ax[-1].set(xlabel='l2-dist=' + str(dist))
            ax[-1].xaxis.label.set_fontsize(25)
        plt.imshow(img)
    plt.savefig(str(name))
    # plt.show()


def ten_imgs(test_index):
    v = X_test[test_index]
    cat_class = y_test[test_index]

    img_distances = np.sum((X_train - v) ** 2, axis=1)
    img_distances = np.sqrt(img_distances)
    img_dist_indx = zip(img_distances, range(img_distances.shape[0]))
    sorted_imgs = sorted(img_dist_indx, key=lambda x: x[0])
    top_ten = [(x[0], train_df.loc[x[1], "query"]) for x in sorted_imgs[0:10]]
    bottom_ten = [(x[0], train_df.loc[x[1], "query"]) for x in sorted_imgs[-10:]]

    show_images([(0.0, test_df.loc[test_index, "query"])] + top_ten, cat_class + "top_ten.png")
    show_images([(0.0, test_df.loc[test_index, "query"])] + bottom_ten, cat_class + "bottom_ten.png")


# get images for 5 Validation set
test_classes = [207, 223, 249, 362, 414]
for i in test_classes:
    a = ten_imgs(i)
