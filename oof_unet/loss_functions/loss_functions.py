import torch
import torch.nn as nn


class MaskedSSELoss(nn.Module):
    """
    Computes the Masked Sum of Squared Errors (SSE) loss.

    This loss function calculates the squared error between the predicted outputs and the targets,
    applies a mask to focus only on specific entries, and returns the sum of these squared errors.

    The mask should be a binary tensor (or have values that can be interpreted as weights),
    where a value of 1 indicates that the corresponding error should be included in the loss,
    and 0 indicates that it should be ignored.

    Attributes:
        volume (float): The fixed volume (or constant) by which to divide the computed loss.

    Example:
        loss_fn = MaskedSSELoss()
        loss = loss_fn.forward(outputs, targets, mask)
    """

    def __init__(self, volume: float = 10_000_000):
        """
        Initializes the MaskedSSELoss with a fixed volume for normalization

        Args:
            volume (float): The fixed volume (or constant) by which to divide the computed loss
        """

        super(MaskedSSELoss, self).__init__()
        self.volume = volume

    def forward(self, outputs, targets, mask):
        """
        Forward pass for computing the Masked Sum of Squared Errors (SSE) loss :

        Args:
            outputs (torch.Tensor): The predicted outputs of the model.
            targets (torch.Tensor): The target values.
            mask (torch.Tensor): The mask to apply to the loss.

        Returns:
            torch.Tensor: The Masked Sum of Squared Errors (SSE) loss.
        """
        return ((outputs - targets) ** 2 * mask).sum() / self.volume


class MaskedMSELoss(nn.Module):
    """
    Computes the Masked Mean Squared Error (MSE) Loss.

    This loss function calculates the squared error between the predicted outputs and the target values,
    applies a mask to focus only on specific (valid) elements, and then computes the mean squared error by
    dividing the summed squared errors by the total count of valid entries (i.e., the sum of the mask).
    A small epsilon is added to the denominator to ensure numerical stability and prevent division by zero.

    Attributes:
        epsilon (float): A small constant for numerical stability.

    Example:
        loss_fn = MaskedMSELoss(epsilon=1e-8)
        loss = loss_fn(outputs, targets, mask)
    """

    def __init__(self, epsilon=1e-8):
        """
        Initializes the MaskedMSELoss with a small constant for numerical stability

        Args:
            epsilon (float): A small constant for numerical stability
        """

        super(MaskedMSELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, outputs, targets, mask):
        """
        Forward pass for computing the Masked Mean Squared Error (MSE) loss, the loss is computed on the mask voxels
        only

        Args:
            outputs (torch.Tensor): The predicted outputs of the model.
            targets (torch.Tensor): The target values.
            mask (torch.Tensor): The mask to apply to the loss.

        Returns:
            torch.Tensor: The Masked Mean Squared Error (MSE) loss.
        """
        return ((outputs - targets) ** 2 * mask).sum() / (mask.sum() + self.epsilon)
