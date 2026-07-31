"""Canonical architecture settings shared by every execution path."""

BASE_MODEL = "efficientnet_b0"
BACKBONE_PRETRAIN = "imagenet"

# Each entry is (hidden dimensions, dropout before each linear layer).
HEAD_SPECS = {
    "flat": ((512,), (0.3, 0.2)),
    "class": ((512,), (0.3, 0.2)),
    "order": ((512, 256), (0.3, 0.2, 0.1)),
    "family": ((512, 256), (0.3, 0.2, 0.1)),
    "genus": ((1024, 512, 256), (0.3, 0.2, 0.2, 0.1)),
    "species": ((1024, 512, 256), (0.3, 0.2, 0.2, 0.1)),
}

