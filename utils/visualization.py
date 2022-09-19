import matplotlib.pyplot as plt

def plot_horizontal(images, x_labels, figsize=(15,3)):
    n = len(images)
    fig, axes = plt.subplots(1,n, figsize=figsize)
    for i in range(n):
        axes[i].imshow(images[i])
        axes[i].set_xlabel(f"{x_labels[i]}")