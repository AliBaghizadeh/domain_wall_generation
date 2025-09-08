# Import Libraries
import torch
import torch.nn as nn

# Import Finished

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=2, n_classes=3, img_size=256):
        super().__init__()
        self.img_size = img_size
        self.label_embed = nn.Embedding(n_classes, img_size * img_size)

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels + 1, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, disp_map, image, labels):
        B, _, H, W = disp_map.shape
        # make sure embedding matches the actual image resolution
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}x{W}) does not match discriminator img_size={self.img_size}"

        label_map = self.label_embed(labels).view(B, 1, H, W)
        x = torch.cat([disp_map, image, label_map], dim=1)
        return self.model(x)
