from sklearn.neighbors import KNeighborsClassifier
x_train = [[0], [1], [2], [3]]
y_train = [0, 0, 1, 1]
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(x_train, y_train)
x_text = [[1.5],[4]]
y_text = neigh.predict(x_text)
print(y_text)  