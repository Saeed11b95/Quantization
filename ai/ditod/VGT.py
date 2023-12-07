# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict, List, Optional, Tuple
from detectron2.modeling.postprocessing import detector_postprocess
import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from torch import nn

from .Wordnn_embedding import WordnnEmbedding

__all__ = ["VGT"]


def torch_memory(device, tag=""):
    # Checks and prints GPU memory
    print(tag, f"{torch.cuda.memory_allocated(device)/1024/1024:.2f} MB USED")
    print(tag, f"{torch.cuda.memory_reserved(device)/1024/1024:.2f} MB RESERVED")
    print(tag, f"{torch.cuda.max_memory_allocated(device)/1024/1024:.2f} MB USED MAX")
    print(
        tag, f"{torch.cuda.max_memory_reserved(device)/1024/1024:.2f} MB RESERVED MAX"
    )
    print("")


@META_ARCH_REGISTRY.register()
class VGT(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        *,
        vocab_size: int = 30552,
        hidden_size: int = 768,
        embedding_dim: int = 64,
        bros_embedding_path: str = "",
        use_pretrain_weight: bool = True,
        use_UNK_text: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.training = (False,)
        # TODO Remove this, it's jsut for testing
        self.embedding_dim = embedding_dim
        self.embedding_proj = nn.Linear(hidden_size, embedding_dim, bias=False)
        # self.Wordgrid_embedding = WordnnEmbedding(
        #     vocab_size,
        #     hidden_size,
        #     embedding_dim,
        #     bros_embedding_path,
        #     use_pretrain_weight,
        #     use_UNK_text,
        # )

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update(
            {
                "vocab_size": cfg.MODEL.WORDGRID.VOCAB_SIZE,
                "hidden_size": cfg.MODEL.WORDGRID.HIDDEN_SIZE,
                "embedding_dim": cfg.MODEL.WORDGRID.EMBEDDING_DIM,
                "bros_embedding_path": cfg.MODEL.WORDGRID.MODEL_PATH,
                "use_pretrain_weight": cfg.MODEL.WORDGRID.USE_PRETRAIN_WEIGHT,
                "use_UNK_text": cfg.MODEL.WORDGRID.USE_UNK_TEXT,
            }
        )
        return ret

    def forward(self, images, grid, image_sizes, instances=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        # if not self.training:
        return self.inference(images, grid, image_sizes)
        # images = self.preprocess_image(batched_inputs)
        gt_instances = instances
        chargrid = self.embedding_proj(grid).permute(0, 3, 1, 2).contiguous()
        features = self.backbone(images.tensor, chargrid)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses

    def inference(
        self,
        images,
        grid,
        image_sizes,
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training
        # images = self.preprocess_image(batched_inputs)
        chargrid = self.embedding_proj(grid).permute(0, 3, 1, 2).contiguous()
        features = self.backbone(images.tensor, chargrid)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            assert (
                not torch.jit.is_scripting()
            ), "Scripting is not supported for postprocess."
            return self._final_postprocess(
                GeneralizedRCNN._postprocess(results, image_sizes, images.tensor)
            )
        else:
            return results

    def _final_postprocess(self, results):
        out = []
        for res in results:
            out.append(
                {
                    "bboxes": res["instances"].pred_boxes.tensor,
                    "scores": res["instances"].scores,
                    "class": res["instances"].pred_classes,
                }
            )
        return out

    @staticmethod
    def _postprocess(
        instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes
    ):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
