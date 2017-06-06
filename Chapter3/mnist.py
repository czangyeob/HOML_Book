from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
mnist

X,y = mnist["data"], mnist["target"]
X.shape
y.shape

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()