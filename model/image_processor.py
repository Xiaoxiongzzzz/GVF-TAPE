import torch
import torch.nn as nn


class ImageProcessor(nn.Module):
    def __init__(self, camera_matrix):
        super().__init__()
        self.register_buffer(
            "camera_matrix", torch.tensor(camera_matrix, dtype=torch.float32)
        )
        self.register_buffer(
            "reference_matrix",
            torch.tensor(
                [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        # x should be batch of images (B,C,H,W)
        B, C, H, W = x.shape

        # Compute transformation matrix
        transformation_matrix = (
            torch.linalg.inv(self.reference_matrix) @ self.camera_matrix
        )

        # Create sampling grid
        theta = transformation_matrix[:2, :].unsqueeze(0).expand(B, -1, -1)
        grid = torch.nn.functional.affine_grid(theta, [B, C, H, W], align_corners=False)

        # Apply the transformation using grid sampling
        aligned_images = torch.nn.functional.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        return aligned_images
