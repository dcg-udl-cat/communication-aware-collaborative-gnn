"""Encoders de imagen preentrenados (congelados) para reemplazar flatten+PCA.

Por que esto importa mas que cualquier otro cambio:
  flatten -> PCA produce representaciones que mezclan textura, brillo y forma
  de forma indiscriminada. Un encoder preentrenado en ImageNet ha aprendido
  jerarquias de features semanticamente significativas: bordes, texturas,
  partes de objetos, conceptos. Dos imagenes de la misma clase produciran
  embeddings similares; dos imagenes de clases distintas produciran embeddings
  distintos. El grafo KNN construido sobre estos embeddings tendra vecindades
  semanticamente coherentes, que es exactamente lo que un GNN necesita para
  que el paso de mensajes sea util.

Encoders disponibles:
  ResNet18Encoder — penultimate layer de ResNet18, 512-D. Rapido, bien conocido.
  DinoEncoder    — ViT-S/16 con DINO, 384-D. Mejores clusters sin supervision.

Uso tipico:
  encoder = ResNet18Encoder()
  features = encoder.encode(images)   # (N, 512), no grad, en CPU

Nota de despliegue en el borde:
  El encoder solo se ejecuta UNA vez (en el nodo central o durante setup).
  El GNN tiny opera sobre los embeddings resultantes, que son mucho mas
  compactos (512-D) que los pixels crudos (3072-D). En inferencia incremental
  (un nuevo nodo llega), solo se necesita un forward del encoder + un paso
  de KNN lookup + un forward del GNN tiny.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ResNet18Encoder(nn.Module):
    """ResNet18 preentrenado en ImageNet, capa penultima congelada.

    Produce embeddings de 512 dimensiones. El encoder no se entrena:
    sus pesos son fijos durante todo el pipeline. Solo el GNN tiny
    (y opcionalmente el MLP baseline) se entrenan.

    Por que la capa penultima y no la ultima:
      La ultima capa (fc, 1000 clases ImageNet) esta sesgada hacia las
      clases de ImageNet. La penultima (avgpool output, 512-D) contiene
      representaciones generales que transfieren mejor a otros datasets.
    """

    OUTPUT_DIM = 512

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
        except ImportError:
            raise ImportError(
                "torchvision es necesario para ResNet18Encoder. "
                "Instala con: pip install torchvision"
            )

        # Todo excepto el clasificador final
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Congelar todos los parametros: el encoder no se entrena
        for param in self.parameters():
            param.requires_grad_(False)

        self.eval()

    @torch.no_grad()
    def encode(self, images: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        """Codifica un tensor de imagenes en batches para no agotar memoria.

        Args:
            images:     (N, C, H, W) float en [0, 1] o [0, 255].
            batch_size: Numero de imagenes por batch (ajustar segun RAM/VRAM).

        Returns:
            (N, 512) float32, L2-normalizado.
        """
        _validate_images(images)
        images = _to_float_unit(images)

        # ResNet18 espera imagenes normalizadas con medias/stds de ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        self.eval()
        device = next(self.parameters()).device
        embeddings: list[torch.Tensor] = []

        for i in range(0, images.shape[0], batch_size):
            batch = images[i : i + batch_size].to(device)
            emb = self.features(batch).flatten(1)   # (B, 512)
            embeddings.append(emb.cpu())

        out = torch.cat(embeddings, dim=0)
        return F.normalize(out, p=2, dim=1)          # L2-normalizado

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)


class DinoEncoder(nn.Module):
    """ViT-S/16 con DINO (self-supervised), embeddings de 384-D.

    DINO produce clusters semanticos sin supervision que son especialmente
    utiles para grafos KNN: vecindades semanticamente puras sin necesitar
    etiquetas. Requiere conexion a internet en el primer uso (descarga ~85MB).
    """

    OUTPUT_DIM = 384

    def __init__(self) -> None:
        super().__init__()
        try:
            self.backbone = torch.hub.load(
                "facebookresearch/dino:main",
                "dino_vits16",
                pretrained=True,
                verbose=False,
            )
        except Exception as e:
            raise RuntimeError(
                f"No se pudo cargar DINO desde torch.hub: {e}\n"
                "Verifica la conexion a internet o usa ResNet18Encoder."
            )

        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.backbone.eval()

    @torch.no_grad()
    def encode(self, images: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
        """Codifica imagenes con DINO ViT-S/16.

        Args:
            images:     (N, C, H, W) float en [0, 1] o [0, 255].
                        Deben ser 224x224 o se redimensionaran internamente.
            batch_size: Imagenes por batch.

        Returns:
            (N, 384) float32, L2-normalizado.
        """
        _validate_images(images)
        images = _to_float_unit(images)

        # DINO ViT espera 224x224
        if images.shape[-2:] != (224, 224):
            try:
                import torch.nn.functional as F_interp
                images = F_interp.interpolate(
                    images, size=(224, 224), mode="bilinear", align_corners=False
                )
            except Exception:
                pass  # si falla, dejar que el backbone gestione el error

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = (images - mean) / std

        self.backbone.eval()
        embeddings: list[torch.Tensor] = []

        for i in range(0, images.shape[0], batch_size):
            batch = images[i : i + batch_size]
            emb = self.backbone(batch)               # (B, 384)
            embeddings.append(emb.cpu())

        out = torch.cat(embeddings, dim=0)
        return F.normalize(out, p=2, dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_images(images: torch.Tensor) -> None:
    if images.ndim != 4:
        raise ValueError(f"images debe ser (N,C,H,W), recibido: {tuple(images.shape)}")
    if images.shape[0] == 0:
        raise ValueError("images no puede estar vacio.")
    if images.shape[1] != 3:
        raise ValueError(f"Se esperan 3 canales RGB, recibido: {images.shape[1]}")


def _to_float_unit(images: torch.Tensor) -> torch.Tensor:
    """Convierte a float32 en [0,1] sea cual sea el rango original."""
    images = images.float()
    if images.max() > 1.0:
        images = images / 255.0
    return images
