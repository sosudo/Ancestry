# Import required Python libraries and genotypes dataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

genotypes=np.load("foo.npy")

# Create a new PCA object.
pca = PCA(n_components=3)

# Find three principal components, and project the dataset onto them.
# This outputs an array of three-dimensional vectors, one per sample.
genotypes_fit = pca.fit_transform(genotypes)

## Plot the dataset projected onto one of the principal directions
# Prepare axis for scatter, and for histogram to show distribution along axis.
scatter_axes = plt.subplot2grid((3, 2), (1, 0), rowspan=2, colspan=2)
x_hist_axes = plt.subplot2grid((3, 2), (0, 0), colspan=2,
                               sharex=scatter_axes)

# Plot scatter of genotypes on PC 2
scatter_axes.scatter(genotypes_fit[:,2], np.zeros_like(genotypes_fit[:,2]), color='blue', marker='.', s=3)

# Plot histogram of distribution on PC2
x_hist_axes.hist(genotypes_fit[:,2],histtype='step',bins=40)

plt.savefig("scatter.png")

# Plot distribution of PC2 contribution across genome
components=pca.components_
plt.close()
plt.plot(np.absolute(components[2]))
plt.savefig("gender_component.png")