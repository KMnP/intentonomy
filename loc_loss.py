#!/usr/bin/env python3
"""
this script contains the implementation of localization loss we introduced in our paper.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import cv2
import pycocotools.mask as mask_util
import numpy as np

import torchvision
logger = logging.getLogger("intentology_trainer")
CAM_REGULATE_CLS = {
    'object': [0, 3, 10, 11, 12, 16, 23],
    'context': [7, 8],
}
# TODO: change this for your dir for the coco_maskrcnn.json
MASK_ROOT = "/checkpoint/menglin/projects/2020intent"


class Localizationloss(nn.Module):
    '''
    regulate localization for specific classes, so the model can focus on obj/context more. The default value is what we used in the paper
    Args:
        cam_alpha: float, determines the strength / weight of this loss.
        binary_cam_mask: bool, if true, use binary mask
        binary_target_mask: bool, if true, use binary mask
    '''
    def __init__(
        self,
        cam_alpha=0.1,
        binary_cam_mask=False,
        binary_target_mask=True
    ):
        super(Localizationloss, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.cam_regulate_clses = CAM_REGULATE_CLS
        self.all_clses = self.cam_regulate_clses["object"] + \
            self.cam_regulate_clses["context"]
        self.cam_alpha = torch.tensor(
            cam_alpha, dtype=torch.float, device=self.device)

        self._prepare_CAM()

        self.mask_length = 7  # small size eliminate small objects
        self.binary_mask = binary_cam_mask
        self.binary_target = binary_target_mask

    def _prepare_CAM(self):
        # read in coco_dict contains MaskRCNN resutls
        with open(f"{MASK_ROOT}/coco_maskrcnn.json", "rb") as fin:
            self.coco_objects = json.load(fin, encoding="utf-8")

        self.imgid2objmasks = {}  # cache in obj masks

    def get_objmasks(self, image_ids):
        obj_masks = []
        for img_id in image_ids:
            if img_id in self.imgid2objmasks:
                obj_masks.append(self.imgid2objmasks[img_id])
            else:
                m = _get_mask(
                    self.coco_objects[f"low/{img_id}.jpg"],
                    (self.mask_length, self.mask_length)
                )
                obj_masks.append(m)
                self.imgid2objmasks[img_id] = m
        obj_masks = np.vstack(obj_masks)
        return torch.tensor(obj_masks, dtype=torch.float, device=self.device)

    def _get_CAM(self, model, X, cls_idx):
        # TODO: this here should be changed according to your model defintion.
        if isinstance(X, tuple):  # image + hs
            layer = model.feature_getter.images.model.layer4
        else:  # image model only
            layer = model.feature_getter.model.layer4

        with GradCam(model, [layer]) as gcam:
            out_b = gcam(X, image_ids="")[1]  # [bs, C]
            out_b[:, cls_idx].mean().backward()

            gcam_b = gcam.get(layer)  # [bs, 1, fmpH, fmpW]
            norm_img = normalize(gcam_b)  # [bs, 1, fmpH, fmpW]
        if self.binary_mask:
            norm_img = norm_img > 0.5
        return norm_img.type(torch.float)

    def get_CAM(self, model, X, batch_size, total_cls):
        # get CAM masks for object classes and CAM masks for context classes
        total_cls = 28
        CAM_masks = torch.zeros(
            (batch_size, total_cls, self.mask_length, self.mask_length)
        )

        for cls_idx in range(28):
            if cls_idx in self.all_clses:
                CAM_masks[:, cls_idx, :, :] = self._get_CAM(
                    model, X, cls_idx).squeeze(1)
        return CAM_masks  # [bs, cls, 7, 7]

    def forward(self, model, X, image_ids, targets) -> torch.Tensor:
        """
        major actions here
        Args:
            model: the nn.Module object
            X: the input to the model, so we can get the CAM masks
            image_ids: indices of the images in this batch, so we can retrieve the object/context masks.
            targets: torch.Tensor of shape batch_Size x total_cls. to get the target mask
        """
        batch_size, total_cls = targets.shape

        if self.binary_target:
            targets = (targets > 1 / 3).type(torch.float).to(self.device)

        # torch.Size([128, 28, 7, 7])
        CAM_masks = self.get_CAM(model, X, batch_size, total_cls).to(self.device)
        # torch.Size([128, 1, 7, 7])
        obj_masks = self.get_objmasks(image_ids).to(self.device)

        # get CAM_masks for obj/context seperately:
        o_cams = CAM_masks[:, self.cam_regulate_clses["object"], :, :]
        c_cams = CAM_masks[:, self.cam_regulate_clses["context"], :, :]

        # Eq 2 in the paper
        o_cam_loss = torch.mul(
            torch.sum(o_cams * (1 - obj_masks), dim=[-1, -2]),
            targets[:, self.cam_regulate_clses["object"]]
        )    # (bs, 12)
        o_cam_loss = torch.sum(o_cam_loss) / torch.tensor(
            len(self.cam_regulate_clses["object"]),
            dtype=torch.float, device=self.device
        )

        # Eq 3 in the paper
        c_cam_loss = torch.mul(
            torch.sum(c_cams * obj_masks, dim=[-1, -2]),
            targets[:, self.cam_regulate_clses["context"]]
        )      # (bs, 2)
        c_cam_loss = torch.sum(c_cam_loss) / torch.tensor(
            len(self.cam_regulate_clses["context"]),
            dtype=torch.float, device=self.device
        )
        cam_loss = (o_cam_loss + c_cam_loss) * self.cam_alpha

        return cam_loss / torch.sum(targets)


def _get_mask(coco_dict, image_size):
    # get binary mask for single image
    objects_ids = [
        i for i, s in enumerate(coco_dict["scores"]) if s > 0.6
    ]

    img_shape = coco_dict["image_shape"][:2]
    obj_mask = np.zeros(img_shape).astype(np.int)

    for i in objects_ids:
        rle = coco_dict["masks"][i]
        rle = mask_util.frPyObjects(rle, *rle["size"])

        # binary masks
        m = mask_util.decode(rle)
        obj_mask = np.logical_or(obj_mask==1, m == 1).astype(np.int)

    obj_mask = cv2.resize(obj_mask.astype('float32'), image_size)
    # [1, 1, 7, 7]
    return obj_mask[np.newaxis, np.newaxis, :, :]


class GradCam:
    """Credits: https://amoshyc.github.io/blog/2019/grad-cam-using-pytorch.html"""
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.hooks = []
        self.fmap_pool = dict()
        self.grad_pool = dict()

        def forward_hook(module, input, output):
            self.fmap_pool[module] = output.detach().cpu()

        def backward_hook(module, grad_in, grad_out):
            self.grad_pool[module] = grad_out[0].detach().cpu()

        for layer in layers:
            self.hooks.append(layer.register_forward_hook(forward_hook))
            self.hooks.append(layer.register_backward_hook(backward_hook))

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __call__(self, *args, **kwargs):
        self.model.zero_grad()
        return self.model(*args, **kwargs)

    def get(self, layer):
        assert layer in self.layers, f'{layer} not in {self.layers}'
        fmap_b = self.fmap_pool[layer]  # [N, C, fmpH, fmpW]
        grad_b = self.grad_pool[layer]  # [N, C, fmpH, fmpW]

        grad_b = F.adaptive_avg_pool2d(grad_b, (1, 1))  # [N, C, 1, 1]
        gcam_b = (fmap_b * grad_b).sum(dim=1, keepdim=True)  # [N, 1, fmpH, fmpW]
        return F.relu(gcam_b)


def normalize(tensor, eps=1e-8):
    '''Normalize each tensor in mini-batch like Min-Max Scaler
    Args:
        tensor: (FloatTensor), sized [N, C, H, W]
    Return:
        tensor: (FloatTensor) ranged [0, 1], sized [N, C, H, W]
    '''
    N = tensor.size(0)
    min_val = tensor.contiguous().view(N, -1).min(dim=1)[0]
    tensor = tensor - min_val.view(N, 1, 1, 1)
    max_val = tensor.contiguous().view(N, -1).max(dim=1)[0]
    tensor = tensor / (max_val + eps).view(N, 1, 1, 1)
    return tensor


def resize_tensor(input_tensors, h, w):
    final_output = None
    batch_size, height, width = input_tensors.shape
    # input_tensors = torch.squeeze(input_tensors, 1)

    for img in input_tensors:
        img_PIL = torchvision.transforms.ToPILImage()(img)
        img_PIL = torchvision.transforms.Resize([h,w])(img_PIL)
        img_PIL = torchvision.transforms.ToTensor()(img_PIL)
        if final_output is None:
            final_output = img_PIL
        else:
            final_output = torch.cat((final_output, img_PIL), 0)
    # final_output = torch.unsqueeze(final_output, 1)
    return final_output
