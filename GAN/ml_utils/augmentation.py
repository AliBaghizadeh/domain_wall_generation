import numpy as np
import cv2
import matplotlib.pyplot as plt
 


def add_salt_pepper_noise(image, prob=0.04):
  """
  Adds salt & pepper noise to a grayscale image using opencv.

  Parameters:
  - image (tf.Tensor): Input image tensor in float32 format.
  - prob (float): Probability of a pixel being affected by noise (default: 2%).

  Returns:
  - numpy image: Noisy image with salt (white) and pepper (black) noise applied.
  """

  image_np = image.copy()
  rand = np.random.uniform(size=image_np.shape)

  # Apply pepper noise (black pixels)
  image_np[rand < prob / 2] = 0.0

  # Apply salt noise (white pixels)
  image_np[rand > (1 - prob / 2)] = 1.0

  return image_np.astype(np.float32)

def atomic_plane_distortion_fixed(image, frequency=10, intensity=0.05):
  """
  Adds periodic distortion to simulate fixed-frequency atomic plane noise.

  - frequency: Every n-th row will be distorted.
  - intensity: Strength of noise applied to selected rows.
  """
  image_np = image.copy()  # Convert Tensor to NumPy before processing
  rows, cols = image_np.shape
  sin_pattern = np.sin(np.linspace(0.0, np.pi * 2 * (rows // frequency), rows))[:, np.newaxis]

  distorted_image = np.clip(image_np + sin_pattern * intensity, 0.0, 1.0)
  return distorted_image.astype(np.float32)


def atomic_plane_distortion_random(image, probability=0.1, intensity=0.1):
  """
  Randomly distorts some atomic planes in the image.

  - probability: Fraction of rows affected.
  - intensity: Strength of distortion added.
  """
  # Convert to NumPy before applying transformations
  image_np = image.copy()
  rows, cols = image_np.shape
  # Select random rows
  mask = np.random.uniform(0, 1, size=(rows,)) < probability
  noise = np.random.uniform(-intensity, intensity, size=(rows, cols))
  # Apply noise to selected rows
  distorted_image = np.where(mask[:, np.newaxis], image_np + noise, image_np)
  return np.clip(distorted_image, 0.0, 1.0).astype(np.float32)

def scan_distortion(image, frequency=5, intensity=3):
  """
  Applies sinusoidal scan distortion to an image, shifting pixels periodically.

  Parameters:
  - image (numpy.ndarray): Input grayscale image (H, W).
  - frequency (int): Number of atomic planes between distortions.
  - intensity (float): Maximum pixel shift for distortion.

  Returns:
  - numpy.ndarray: Distorted image.
  """
  distorted_image = image.copy()
  rows, cols = image.shape

  for i in range(0, rows, frequency):
      shift = int(intensity * np.sin(i / frequency * np.pi))  # Sinusoidal shift pattern
      distorted_image[i, :] = np.roll(image[i, :], shift, axis=0)  # Shift row

  return distorted_image

def drift_distortion(image, frequency=6, intensity=2):
  """
  Introduces gradual drift distortion across atomic planes.

  Parameters:
  - image (numpy.ndarray): Input grayscale image (H, W).
  - frequency (int): Number of atomic planes between drift steps.
  - intensity (int): Maximum drift shift in pixels.

  Returns:
  - numpy.ndarray: Distorted image.
  """
  distorted_image = image.copy()
  rows, cols = image.shape

  shift = 0  # Initial drift shift
  for i in range(0, rows, frequency):
      shift = np.random.randint(-intensity, intensity + 1)  # Random shift per row
      distorted_image[i, :] = np.roll(image[i, :], shift, axis=0)  # Apply shift

  return distorted_image


#Classes

class SaltPepperNoise:
    """
    Add salt-and-pepper noise (random black and white pixels) to a grayscale image or displacement map.

    Parameters
    ----------
    prob : float
        Probability of a pixel being set to noise (default: 0.04).

    Notes
    -----
    - Expects input as a NumPy array of shape (H, W), with float32 type and values in [0, 1].
    - Returns an array of the same shape and type.
    """
    def __init__(self, prob=0.04):
        self.prob = prob
    def __call__(self, image):
        return add_salt_pepper_noise(image, prob=self.prob)

class AtomicPlaneDistortionFixed:
    """
    Apply fixed-frequency periodic distortion to simulate atomic plane irregularities.

    Parameters
    ----------
    frequency : int
        Number of rows between distortion cycles (default: 10).
    intensity : float
        Strength of the periodic distortion (default: 0.05).

    Notes
    -----
    - Expects input as a NumPy array of shape (H, W), float32, values in [0, 1].
    - Returns an array of the same shape and type.
    """
    def __init__(self, frequency=10, intensity=0.05):
        self.frequency = frequency
        self.intensity = intensity
    def __call__(self, image):
        return atomic_plane_distortion_fixed(image, frequency=self.frequency, intensity=self.intensity)

class AtomicPlaneDistortionRandom:
    """
    Apply random distortion to some atomic planes, mimicking random physical defects.

    Parameters
    ----------
    probability : float
        Probability that a row will be distorted (default: 0.1).
    intensity : float
        Maximum strength of random distortion (default: 0.1).

    Notes
    -----
    - Expects input as a NumPy array of shape (H, W), float32, values in [0, 1].
    - Returns an array of the same shape and type.
    """
    def __init__(self, probability=0.1, intensity=0.1):
        self.probability = probability
        self.intensity = intensity
    def __call__(self, image):
        return atomic_plane_distortion_random(image, probability=self.probability, intensity=self.intensity)

class ScanDistortion:
    """
    Apply periodic scan distortion by shifting rows sinusoidally, simulating microscope scan artifacts.

    Parameters
    ----------
    frequency : int
        Number of rows between distortion cycles (default: 5).
    intensity : float
        Maximum number of pixels to shift per row (default: 3).

    Notes
    -----
    - Expects input as a NumPy array of shape (H, W), float32, values in [0, 1].
    - Returns an array of the same shape and type.
    """
    def __init__(self, frequency=5, intensity=3):
        self.frequency = frequency
        self.intensity = intensity
    def __call__(self, image):
        return scan_distortion(image, frequency=self.frequency, intensity=self.intensity)

class DriftDistortion:
    """
    Introduce random drift distortion across atomic planes, simulating gradual drift during acquisition.

    Parameters
    ----------
    frequency : int
        Number of rows between drift steps (default: 6).
    intensity : int
        Maximum pixel shift per drift step (default: 2).

    Notes
    -----
    - Expects input as a NumPy array of shape (H, W), float32, values in [0, 1].
    - Returns an array of the same shape and type.
    """
    def __init__(self, frequency=6, intensity=2):
        self.frequency = frequency
        self.intensity = intensity
    def __call__(self, image):
        return drift_distortion(image, frequency=self.frequency, intensity=self.intensity)


class Apply_Augment:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            if np.random.rand() < 0.5:  # Randomly apply each augmentation
                img = t(img)
        return img



def visualize_augmentations_grid(dataset, idx_list=None, num_examples=8, cols=2):
    """
    Show several augmented images from STEMImageDataset in a grid.
    
    idx_list: list of dataset indices to visualize. If None, picks first `num_examples`.
    num_examples: total images to show.
    cols: number of columns in the grid.
    """
    import math

    if idx_list is None:
        idx_list = list(range(min(len(dataset), num_examples)))

    rows = math.ceil(len(idx_list) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten()

    for ax, idx in zip(axes, idx_list):
        img, label = dataset[idx]
        img_np = img.permute(1, 2, 0).numpy()
        ax.imshow(img_np, cmap='gray')
        ax.axis('off')
        ax.set_title(f"Label: {label}")

    # Hide any extra axes if num_examples < rows*cols
    for ax in axes[len(idx_list):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()