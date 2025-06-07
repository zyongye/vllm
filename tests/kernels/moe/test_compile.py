from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class BaseFlexData:
    dtype: torch.dtype | None = None

    def view(self, x: torch.Tensor):
        if self.dtype is None:
            return x
        return x.view(self.dtype)

    def reinterpret(self, x):
        if self.dtype is None or x.dtype.itemsize > 1:
            return x
        return x.view(self.dtype)

@dataclass(frozen=True)
class InFlexData(BaseFlexData):
    scale: torch.Tensor | None = None

    @property
    def is_per_batch(self):
        return False if self.scale is None else len(self.scale) > 1

@dataclass(frozen=True)
class OutFlexData(BaseFlexData):
    expected_scale: torch.Tensor | None = None
    actual_scale: torch.Tensor | None = None
    checksum_scale: torch.Tensor | None = None

    def __iter__(self):
        yield self.expected_scale
        yield self.actual_scale
        yield self.checksum_scale

@dataclass(frozen=True)
class FlexCtx:
    lhs_data: InFlexData = InFlexData()
    rhs_data: InFlexData = InFlexData()
    out_data: OutFlexData = OutFlexData()

@dataclass
class DummyClass:
    flex_ctx: FlexCtx = FlexCtx()

    def __post_init__(self):
        assert self.flex_ctx.rhs_data.scale is None, "flex and mx_ctx cannot be used together"

@torch.compile(fullgraph=True)
def dummy_method():
    var = DummyClass(flex_ctx=FlexCtx(rhs_data=InFlexData()))
    return var

dummy_method()
