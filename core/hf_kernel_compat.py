"""Hugging Face Hub kernel統合の互換回避。"""

from __future__ import annotations

import os
import sys
import types
from contextlib import contextmanager
from typing import Any, Callable


def disable_hub_kernel_integration() -> None:
    """任意のHub kernel統合をno-op化して、依存版数不整合のimport失敗を避ける。"""
    os.environ.setdefault("USE_HUB_KERNELS", "NO")

    module_name = "transformers.integrations.hub_kernels"
    if module_name in sys.modules:
        return

    def identity_decorator(_name: str) -> Callable[[Any], Any]:
        return lambda target: target

    def unavailable_kernel(*_args: Any, **_kwargs: Any) -> Any:
        raise ImportError("Hub kernel統合はこの実行では無効化されています。")

    def use_kernelized_func(module_names: list[Callable[..., Any]] | Callable[..., Any]):
        """kernelize用の関数を通常属性として登録するだけのno-op decorator。"""
        if callable(module_names):
            functions = [module_names]
        else:
            functions = list(module_names)

        def decorator(cls: type) -> type:
            original_init = cls.__init__

            def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
                original_init(self, *args, **kwargs)
                for function in functions:
                    setattr(self, "rotary_fn", function)

            cls.__init__ = new_init
            return cls

        return decorator

    class LayerRepository:
        """Hub kernelを使わない実行向けのダミーRepository。"""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def load(self) -> Any:
            raise ImportError("Hub kernel統合はこの実行では無効化されています。")

    def register_kernel_mapping(*_args: Any, **_kwargs: Any) -> None:
        return None

    def lazy_load_kernel(*_args: Any, **_kwargs: Any) -> Any:
        return unavailable_kernel(*_args, **_kwargs)

    @contextmanager
    def allow_all_hub_kernels():
        yield

    def is_kernel(_attn_implementation: str | None) -> bool:
        return False

    stub = types.ModuleType(module_name)
    stub.ALLOW_ALL_KERNELS = False
    stub.LayerRepository = LayerRepository
    stub.lazy_load_kernel = lazy_load_kernel
    stub.register_kernel_mapping = register_kernel_mapping
    stub.use_kernel_forward_from_hub = identity_decorator
    stub.use_kernel_func_from_hub = identity_decorator
    stub.use_kernelized_func = use_kernelized_func
    stub.get_kernel = unavailable_kernel
    stub.replace_kernel_forward_from_hub = identity_decorator
    stub.allow_all_hub_kernels = allow_all_hub_kernels
    stub.is_kernel = is_kernel
    sys.modules[module_name] = stub
