import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class SCOLDAttentionExtractor(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def get_last_layer_attention(self, pixel_values):

        outputs = self.model.vision_encoder(
            pixel_values=pixel_values,
            output_attentions=True,
            return_dict=True
        )

        attn = outputs.attentions[-1].mean(dim=1)
        cls_attn = attn[:, 0, 1:]

        grid = int(np.sqrt(cls_attn.shape[-1]))

        return cls_attn.reshape(
            -1, 1, grid, grid
        )

    @torch.no_grad()
    def get_attention_rollout(
        self,
        pixel_values,
        discard_ratio=0.9
    ):

        outputs = self.model.vision_encoder(
            pixel_values=pixel_values,
            output_attentions=True,
            return_dict=True
        )

        attentions = outputs.attentions
        batch, _, seq_len, _ = attentions[0].shape

        eye = torch.eye(
            seq_len,
            device=pixel_values.device
        ).unsqueeze(0)

        rollout = eye.repeat(
            batch, 1, 1
        )

        for attn in attentions:

            attn = attn.mean(dim=1)

            if discard_ratio > 0:

                flat = attn.flatten(1)

                k = max(
                    1,
                    int(
                        flat.shape[-1]
                        * (1 - discard_ratio)
                    )
                )

                threshold = torch.topk(
                    flat,
                    k,
                    dim=-1
                ).values[:, -1].view(
                    batch, 1, 1
                )

                attn = torch.where(
                    attn >= threshold,
                    attn,
                    torch.zeros_like(attn)
                )

            attn = (
                0.5 * attn +
                0.5 * eye
            )

            attn = attn / (
                attn.sum(
                    dim=-1,
                    keepdim=True
                ) + 1e-8
            )

            rollout = torch.bmm(
                attn,
                rollout
            )

        cls_rollout = rollout[:, 0, 1:]

        grid = int(
            np.sqrt(
                cls_rollout.shape[-1]
            )
        )

        return cls_rollout.reshape(
            -1, 1, grid, grid
        )


class SCOLDGradCAM:

    def __init__(
        self,
        model,
        target_layer=None
    ):

        self.model = model
        self.model.eval()

        self.target_layer = (
            target_layer
            if target_layer is not None
            else model.vision_encoder.encoder.layers[-1]
        )

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(
            self._forward_hook
        )

        self.target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output
    ):

        self.activations = (
            output[0]
            if isinstance(output, tuple)
            else output
        )

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = (
            grad_output[0]
            if isinstance(grad_output, tuple)
            else grad_output
        )

    def generate_heatmap(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        target_class_idx=0
    ):

        self.model.zero_grad(set_to_none=True)

        self.activations = None
        self.gradients = None

        logits, _, _, _ = self.model(
            pixel_values,
            input_ids,
            attention_mask
        )

        logits[0, target_class_idx].backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        acts = self.activations[:, 1:, :]
        grads = self.gradients[:, 1:, :]

        cam = (
            acts * grads
        ).sum(dim=-1)

        cam = F.relu(cam)

        if cam.max().item() == 0:

            cam = torch.abs(
                acts * grads
            ).sum(dim=-1)

        grid = int(
            np.sqrt(
                cam.shape[-1]
            )
        )

        if grid * grid != cam.shape[-1]:
            raise RuntimeError(
                f"Invalid number of visual tokens: "
                f"{cam.shape[-1]}"
            )

        cam = cam.reshape(
            -1, 1, grid, grid
        )

        cam_min = cam.amin(
            dim=(2, 3),
            keepdim=True
        )

        cam_max = cam.amax(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        return cam.detach()


class SCOLDVisualizer:

    @staticmethod
    def overlay_heatmap(
        image,
        heatmap,
        alpha=0.5,
        colormap=cv2.COLORMAP_JET
    ):

        h, w = image.shape[:2]

        if torch.is_tensor(heatmap):
            heatmap = (
                heatmap.detach()
                .cpu()
                .numpy()
            )

        heatmap = np.squeeze(
            heatmap
        ).astype(np.float32)

        heatmap = cv2.resize(
            heatmap,
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )

        heatmap -= heatmap.min()

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            colormap
        )

        colored = cv2.cvtColor(
            colored,
            cv2.COLOR_BGR2RGB
        )

        blended = cv2.addWeighted(
            image.astype(np.uint8),
            1 - alpha,
            colored,
            alpha,
            0
        )

        return blended, heatmap

    @staticmethod
    def draw_disease_bounding_boxes(
        image,
        heatmap,
        threshold=0.55,
        min_area=100
    ):

        image = np.asarray(
            image
        ).copy()

        h, w = image.shape[:2]

        heatmap = np.asarray(
            heatmap
        ).squeeze().astype(
            np.float32
        )

        heatmap -= heatmap.min()

        if heatmap.max() == 0:
            return image, []

        heatmap /= heatmap.max()

        heatmap = cv2.resize(
            heatmap,
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )

        mask = (
            heatmap >= threshold
        ).astype(
            np.uint8
        ) * 255

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []

        for cnt in contours:

            if cv2.contourArea(cnt) < min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            boxes.append(
                (y, x, y + bh, x + bw)
            )

        boxes.sort(
            key=lambda b: (b[0], b[1])
        )

        for i, (y1, x1, y2, x2) in enumerate(
            boxes,
            1
        ):

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                f"Spot {i}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        return image, boxes