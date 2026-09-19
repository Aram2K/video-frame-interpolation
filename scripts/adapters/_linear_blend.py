"""Test adapter: linear cross-fade (no model). Used only to validate vfi_harness.py."""
import torch


class Adapter:
    def __init__(self, device, **kwargs):
        self.device = device

    def info(self):
        return {"name": "linear-blend (test)", "license": "n/a"}

    def interpolate(self, img0, img1, ts):
        return [img0 * (1 - t) + img1 * t for t in ts]
