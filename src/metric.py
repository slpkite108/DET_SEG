import torch
from monai.metrics import CumulativeIterationMetric
from monai.metrics.utils import do_metric_reduction
from monai.utils import MetricReduction
from typing import Union, Tuple
from src.utils import calculate_bounding_box

class BoundingBoxMeanIoU(CumulativeIterationMetric):
    """
    Compute average Intersection over Union (IoU) score between two sets of bounding boxes.
    This class supports the calculation of IoU for multiple bounding boxes in a batch.
    """

    def __init__(
        self,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Computes Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (torch.Tensor): Predicted bounding boxes, shape (N, 4), where N is the number of boxes.
            box2 (torch.Tensor): Ground truth bounding boxes, shape (N, 4).

        Returns:
            torch.Tensor: IoU scores per bounding box, shape (N,).
        """
        # Calculate intersection
        # print(box1,box2)
        
        
        xA = torch.max(box1[:, 0], box2[:, 0])
        yA = torch.max(box1[:, 1], box2[:, 1])
        zA = torch.max(box1[:, 2], box2[:, 2])
        xB = torch.min(box1[:, 3], box2[:, 3])
        yB = torch.min(box1[:, 4], box2[:, 4])
        zB = torch.min(box1[:, 5], box2[:, 5])

        interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0) * torch.clamp(zB - zA, min=0)

        # Calculate union
        box1Area = (box1[:, 3] - box1[:, 0]) * (box1[:, 4] - box1[:, 1]) * (box1[:, 5] - box1[:, 2])
        box2Area = (box2[:, 3] - box2[:, 0]) * (box2[:, 4] - box2[:, 1]) * (box2[:, 5] - box2[:, 2])
        unionArea = box1Area + box2Area - interArea

        # Compute IoU
        iou = interArea / unionArea

        return iou

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Overridden method to compute the metric for each batch.

        Args:
            y_pred (torch.Tensor): Predicted bounding boxes.
            y (torch.Tensor): Ground truth bounding boxes.

        Returns:
            torch.Tensor: Computed IoU values for the batch.
        """
        return self.compute_iou(calculate_bounding_box(y_pred), calculate_bounding_box(y))

    def aggregate(
        self, reduction: Union[MetricReduction, str, None] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Overridden method to aggregate the metrics over all batches.

        Args:
            reduction (MetricReduction | str | None): Reduction type to apply to the aggregated data.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: The aggregated metric.
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("The data to aggregate must be a PyTorch Tensor.")

        # Apply metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f