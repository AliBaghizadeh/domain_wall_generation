# Import Libraries
import random
import torchvision.transforms.functional as TF

# Import Finished

class JointTransform:
    """
    Apply identical random geometric and photometric transformations to a pair of images:
    - a displacement map and
    - a real STEM image.

    This class is used to ensure consistency between input (displacement map)
    and target (real image) during training of conditional GANs.

    Transformations applied:
        - Resize
        - Horizontal/Vertical Flip
        - Small Rotation
        - Normalization to [-1, 1]
    """
    def __init__(self, resize=(128, 128), p_flip=0.5, rotation=5):
        self.resize = resize
        self.p_flip = p_flip
        self.rotation = rotation

    def __call__(self, disp, img):
        disp = TF.resize(disp, self.resize)
        img = TF.resize(img, self.resize)

        if random.random() < self.p_flip:
            disp = TF.hflip(disp)
            img = TF.hflip(img)
        if random.random() < self.p_flip:
            disp = TF.vflip(disp)
            img = TF.vflip(img)

        angle = random.uniform(-self.rotation, self.rotation)
        disp = TF.rotate(disp, angle)
        img = TF.rotate(img, angle)

        disp = TF.to_tensor(disp)
        img = TF.to_tensor(img)

        disp = TF.normalize(disp, [0.5], [0.5])
        img = TF.normalize(img, [0.5], [0.5])

        return disp, img
