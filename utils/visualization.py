import numpy as np
import matplotlib.pyplot as plt


def visualize_sar_image(images: np.ndarray, i: int):
    fig, ax = plt.subplots()
    ax.imshow(images[i, :, :], cmap='gray')
    ax.set_axis_off()
    plt.show()
    plt.close(fig)


def visualize_change_variables(variables: np.ndarray):
    variable_names = ["Ratio+ (CPT)", "Ratio- (CPT)", "KLD (CPT)", "Point (CPT)"]
    fig, axs = plt.subplots(1, 4)
    fig.tight_layout()
    for i in range(4):
        axs[i].imshow(variables[:, :, i])
        axs[i].set_title(variable_names[i])
        axs[i].set_axis_off()
    plt.show()
    plt.close(fig)
