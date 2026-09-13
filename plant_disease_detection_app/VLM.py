import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer, AutoModel
from PIL import Image

class SCOLD(nn.Module):
    def __init__(self, base_model_name="./saved_clip_model", projection_dim=512):
        super(SCOLD, self).__init__()
        clip_model = AutoModel.from_pretrained(base_model_name, local_files_only=True, attn_implementation = 'eager')
        self.vision_encoder = clip_model.vision_model
        self.text_encoder = clip_model.text_model
        self.visual_projection = nn.Linear(self.vision_encoder.config.hidden_size, projection_dim)
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, projection_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def encode_image(self, pixel_values):
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embeds = vision_outputs.pooler_output 
        image_embeds = self.visual_projection(image_embeds)
        return F.normalize(image_embeds, p=2, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.pooler_output
        text_embeds = self.text_projection(text_embeds)
        return F.normalize(text_embeds, p=2, dim=-1)

    def forward(self, pixel_values, input_ids, attention_mask):
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * torch.matmul(image_embeds, text_embeds.t())
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text, image_embeds, text_embeds

class ContextAwareSoftTargetLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.05):
        super(ContextAwareSoftTargetLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def generate_soft_targets(self, text_embeds):
        text_sim = torch.matmul(text_embeds, text_embeds.t())
        text_sim = torch.clamp(text_sim, min=0.0)
        identity = torch.eye(text_sim.size(0), device=text_sim.device)
        soft_targets = identity + self.alpha * (text_sim * (1 - identity)) - self.beta * (1 - identity)
        soft_targets = F.softmax(soft_targets, dim=-1)
        return soft_targets

    def forward(self, logits_per_image, logits_per_text, text_embeds):
        soft_targets_img = self.generate_soft_targets(text_embeds)
        soft_targets_txt = soft_targets_img.t()
        log_probs_img = F.log_softmax(logits_per_image, dim=-1)
        log_probs_txt = F.log_softmax(logits_per_text, dim=-1)
        loss_img = F.kl_div(log_probs_img, soft_targets_img, reduction='batchmean')
        loss_txt = F.kl_div(log_probs_txt, soft_targets_txt, reduction='batchmean')
        return (loss_img + loss_txt) / 2.0