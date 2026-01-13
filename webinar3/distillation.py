"""
Knowledge Distillation Implementation
=====================================
Train smaller student models from larger teachers.

Part of Webinar 3: Optimizing Multilingual NLP Models
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation:
    L = α * L_soft(student, teacher) + (1-α) * L_hard(student, labels)
    
    Args:
        temperature: Softmax temperature for soft targets
        alpha: Weight for distillation loss vs hard label loss
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        # Soft target loss (distillation)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        distillation_loss = self.kl_loss(soft_student, soft_targets) * (self.temperature ** 2)
        
        # Hard target loss
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        return self.alpha * distillation_loss + (1 - self.alpha) * hard_loss


class MultilingualDistillationTrainer:
    """
    Trainer for multilingual knowledge distillation with language-stratified sampling.
    
    Key features:
    - Language-balanced batches to prevent bias toward high-resource languages
    - Per-language loss tracking
    - Validation on each language separately
    """
    
    def __init__(
        self,
        teacher_model,
        student_model,
        tokenizer,
        temperature: float = 4.0,
        alpha: float = 0.7,
        device: str = "cuda"
    ):
        self.teacher = teacher_model.to(device).eval()
        self.student = student_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.loss_fn = DistillationLoss(temperature, alpha)
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def create_language_balanced_batch(
        self,
        data_by_language: Dict[str, List],
        batch_size: int
    ) -> List[Tuple]:
        """
        Create batches with equal representation from each language.
        Critical for preventing distillation bias toward high-resource languages.
        """
        languages = list(data_by_language.keys())
        samples_per_lang = batch_size // len(languages)
        
        batch = []
        for lang in languages:
            lang_data = data_by_language[lang]
            indices = np.random.choice(len(lang_data), samples_per_lang, replace=True)
            batch.extend([lang_data[i] for i in indices])
        
        np.random.shuffle(batch)
        return batch
    
    def train_step(self, batch_texts: List[str], batch_labels: List[int]) -> float:
        """Single training step with distillation."""
        self.student.train()
        
        # Tokenize
        inputs = self.tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        labels = torch.tensor(batch_labels).to(self.device)
        
        # Get teacher predictions (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Get student predictions
        student_outputs = self.student(**inputs)
        student_logits = student_outputs.logits
        
        # Calculate loss
        loss = self.loss_fn(student_logits, teacher_logits, labels)
        
        return loss
    
    def evaluate_per_language(
        self,
        eval_data: Dict[str, List[Tuple[str, int]]]
    ) -> Dict[str, float]:
        """Evaluate student model accuracy per language."""
        self.student.eval()
        results = {}
        
        for lang, data in eval_data.items():
            correct = 0
            total = 0
            
            for text, label in data:
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.student(**inputs)
                    pred = outputs.logits.argmax(dim=-1).item()
                
                correct += int(pred == label)
                total += 1
            
            results[lang] = correct / total if total > 0 else 0.0
        
        return results


def create_student_model(num_layers: int = 4, hidden_size: int = 768, num_labels: int = 3):
    """
    Create a smaller transformer model for distillation.
    
    Args:
        num_layers: Number of transformer layers (teacher typically has 12)
        hidden_size: Hidden dimension size
        num_labels: Number of output classes
    """
    from transformers import BertConfig, BertForSequenceClassification
    
    config = BertConfig(
        vocab_size=250002,  # XLM-R vocab size
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=12,
        intermediate_size=hidden_size * 4,
        num_labels=num_labels
    )
    
    return BertForSequenceClassification(config)
