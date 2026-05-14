import time

fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

times = []
n_samples = len(X_scaled)

for frac in fractions:

    size = int(frac * n_samples)

    X_sub = X_scaled[:size]
    y_sub = y[:size]

    clf = SVC(kernel = 'rbf')

    s = time.time()
    clf.fit(X_sub, y_sub)

    e = time.time()
    times.append(e - s)

sizes = [int(frac * n_samples) for frac in fractions]

plt.figure(figsize = (8,8))

plt.plot(sizes, times, marker = 'o')

plt.xlabel("Dataset Size ")
plt.ylabel("Training Time (s) ")

plt.title("SVM Training Time Scaling : ")

plt.grid(True)
plt.show()
