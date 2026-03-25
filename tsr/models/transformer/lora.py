import math
from typing import Dict, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear layers.

    Freezes the original weight W and adds trainable low-rank matrices A and B.
    Output: Wx + (alpha / r) * B(A(x))

    Supports runtime enable/disable and in-place weight swapping for fast
    adapter switching.
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.enabled = True

        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        # Freeze original linear layer
        self.linear = linear_layer
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        # Low-rank decomposition matrices
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        # Dropout on the LoRA path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize A with Kaiming uniform, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        if self.enabled:
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
            return base_out + lora_out * self.scaling
        return base_out

    def reset_lora_parameters(self) -> None:
        """Re-initialize LoRA weights (useful before training a new adapter)."""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)


def apply_lora_to_linear(
    module: nn.Module,
    target_attr: str,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> None:
    """Replace a named nn.Linear attribute on a module with a LoRALinear wrapper."""
    linear = getattr(module, target_attr)
    if isinstance(linear, nn.Linear):
        setattr(module, target_attr, LoRALinear(linear, r=r, alpha=alpha, dropout=dropout))


# ---------------------------------------------------------------------------
# Adapter management utilities
# ---------------------------------------------------------------------------

def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract ONLY the LoRA parameters (lora_A, lora_B) from a model.

    Returns a dict suitable for torch.save(). The resulting file is small
    (typically a few MB) since it excludes all base-model weights.
    """
    return {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if "lora_" in name
    }


def load_lora_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> None:
    """Load LoRA weights into an existing model that already has LoRA layers.

    Args:
        model: The model containing LoRALinear modules.
        state_dict: A dict of ``{name: tensor}`` produced by
            ``get_lora_state_dict`` or ``torch.load(path)``.
        strict: If True, every key in *state_dict* must match a LoRA
            parameter in *model* (and vice-versa).
    """
    current_lora = {
        name: param
        for name, param in model.named_parameters()
        if "lora_" in name
    }

    if strict:
        missing = set(current_lora.keys()) - set(state_dict.keys())
        unexpected = set(state_dict.keys()) - set(current_lora.keys())
        if missing or unexpected:
            raise RuntimeError(
                f"LoRA state dict mismatch. "
                f"Missing keys: {missing}, Unexpected keys: {unexpected}"
            )

    for name, param in current_lora.items():
        if name in state_dict:
            param.data.copy_(state_dict[name])


@torch.no_grad()
def set_active_lora(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    """Hot-swap the active LoRA adapter by copying new A/B weights in-place.

    This is the fast path for inference-time adapter switching:
    - Does NOT recreate any modules
    - Does NOT reload base weights
    - ONLY overwrites lora_A and lora_B tensors via ``Tensor.copy_``

    Typical latency: <5 ms on GPU, <20 ms on CPU.
    """
    load_lora_state_dict(model, state_dict, strict=True)


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    """Enable or disable all LoRA contributions across the model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled


def reset_lora(model: nn.Module) -> None:
    """Re-initialize all LoRA weights to their starting values (B=0)."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.reset_lora_parameters()


# ---------------------------------------------------------------------------
# Multi-adapter manager
# ---------------------------------------------------------------------------

class LoRAAdapterManager:
    """Manages a collection of named LoRA adapters for a single model.

    Adapters are loaded lazily from disk on first use and cached in CPU memory.
    Switching adapters copies only the small LoRA tensors to the model's device.

    Example::

        manager = LoRAAdapterManager(model, {
            "chair": "adapters/lora_chair.pt",
            "table": "adapters/lora_table.pt",
            "sofa":  "adapters/lora_sofa.pt",
        })
        manager.set_adapter("chair")   # fast switch
        manager.set_adapter("table")   # fast switch
        manager.disable()              # run with base model only
        manager.enable()               # re-enable last adapter
    """

    def __init__(
        self,
        model: nn.Module,
        adapter_map: Optional[Dict[str, str]] = None,
    ):
        self.model = model
        self.adapter_map: Dict[str, str] = adapter_map or {}
        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._active: Optional[str] = None

    # -- Registration -------------------------------------------------------

    def register_adapter(self, name: str, path: str) -> None:
        """Register an adapter file path (does not load it yet)."""
        self.adapter_map[name] = path
        self._cache.pop(name, None)

    def register_state_dict(
        self, name: str, state_dict: Dict[str, torch.Tensor]
    ) -> None:
        """Register an already-loaded state dict (skips disk I/O on switch)."""
        self._cache[name] = {
            k: v.cpu() for k, v in state_dict.items()
        }

    # -- Loading / caching --------------------------------------------------

    def _ensure_cached(self, name: str) -> Dict[str, torch.Tensor]:
        if name in self._cache:
            return self._cache[name]
        path = self.adapter_map.get(name)
        if path is None:
            raise KeyError(
                f"Unknown adapter '{name}'. "
                f"Registered: {list(self.adapter_map.keys())}"
            )
        sd = torch.load(path, map_location="cpu", weights_only=True)
        self._cache[name] = sd
        return sd

    # -- Switching ----------------------------------------------------------

    @torch.no_grad()
    def set_adapter(self, name: str) -> None:
        """Activate a named adapter. Loads from disk on first use, then cached."""
        sd = self._ensure_cached(name)
        set_active_lora(self.model, sd)
        set_lora_enabled(self.model, True)
        self._active = name

    def disable(self) -> None:
        """Disable LoRA (pure base-model inference)."""
        set_lora_enabled(self.model, False)

    def enable(self) -> None:
        """Re-enable LoRA with the last active adapter."""
        set_lora_enabled(self.model, True)

    @property
    def active_adapter(self) -> Optional[str]:
        return self._active

    @property
    def available_adapters(self):
        return list(self.adapter_map.keys())

    # -- Saving -------------------------------------------------------------

    def save_current_adapter(self, path: str) -> None:
        """Save the model's current LoRA weights to a file."""
        sd = get_lora_state_dict(self.model)
        torch.save(sd, path)

    # -- Cache management ---------------------------------------------------

    def clear_cache(self) -> None:
        """Free all cached adapter state dicts from CPU memory."""
        self._cache.clear()

    def preload_all(self) -> None:
        """Load every registered adapter into the CPU cache."""
        for name in self.adapter_map:
            self._ensure_cached(name)


# ---------------------------------------------------------------------------
# Classifier-based routing
# ---------------------------------------------------------------------------

@torch.no_grad()
def route_to_adapter(
    image: torch.Tensor,
    classifier,
    model: nn.Module,
    adapter_manager: LoRAAdapterManager,
) -> str:
    """Classify an image and activate the matching LoRA adapter.

    Args:
        image: Preprocessed image tensor for the classifier.
        classifier: Callable that returns a category string.
        model: The TripoSR model (unused directly; the adapter_manager
            already holds a reference to it).
        adapter_manager: Manager with registered adapters whose keys match
            the classifier's output categories.

    Returns:
        The category string that was selected.
    """
    category = classifier(image)
    adapter_manager.set_adapter(category)
    return category
