import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Load the dataset
(x_train, y_train), _ = fashion_mnist.load_data()

# Class labels
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Create a 2x5 grid for displaying images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    index = (y_train == i).argmax()  # Find first occurrence of class i
    ax.imshow(x_train[index], cmap="gray")
    ax.axis("off")
    ax.set_title(class_names[i])

plt.tight_layout()
plt.show()
