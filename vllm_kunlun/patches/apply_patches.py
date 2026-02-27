"""
VLLM Kunlun Compatibility Patches
This module applies necessary patches to vllm for kunlun PyTorch 2.5.1 compatibility.
These patches are automatically applied when vllm_kunlun is registered as a vllm plugin.

Patches are divided into two categories:
1. Early patches (apply_early_patches): Applied at module load time, before any vllm imports
   - OpenAI compatibility patch
   - torch._higher_order_ops.auto_functionalized mock

2. File patches (apply_all_patches): Applied during plugin registration
   - vllm/compilation/decorators.py - patched_inline_call signature
   - vllm/compilation/backends.py - VllmBackend.__call__ signature
   - vllm/compilation/piecewise_backend.py - bundled_autograd_cache
   - vllm/attention/layer.py - torch.Size compatibility
"""

import logging
import os
import sys
import types

logger = logging.getLogger("vllm_kunlun")


# ============================================================================
# Early Patches - Applied at module load time (before any vllm imports)
# ============================================================================

_EARLY_PATCHES_APPLIED = False


def _patch_openai_compat():
    """Patch openai.types.chat for compatibility with newer openai versions

    Problem: vLLM 0.15.1 imports ChatCompletionFunctionToolParam which was
    removed in openai >= 1.50
    """
    try:
        import openai.types.chat as chat_module

        if not hasattr(chat_module, "ChatCompletionFunctionToolParam"):
            if hasattr(chat_module, "ChatCompletionToolParam"):
                chat_module.ChatCompletionFunctionToolParam = (
                    chat_module.ChatCompletionToolParam
                )
                if (
                    hasattr(chat_module, "__all__")
                    and "ChatCompletionFunctionToolParam" not in chat_module.__all__
                ):
                    chat_module.__all__ = list(chat_module.__all__) + [
                        "ChatCompletionFunctionToolParam"
                    ]
    except Exception:
        pass


def _patch_torch_higher_order_ops():
    """Patch torch._higher_order_ops for compatibility with kunlun PyTorch

    Problem: kunlun PyTorch 2.5.1+cu118 doesn't have auto_functionalized module
    that vLLM 0.15.1 requires
    """
    try:
        import torch._higher_order_ops as higher_order_ops

        if hasattr(higher_order_ops, "auto_functionalized"):
            return

        class MockAutoFunctionalized:
            """Mock auto_functionalized for kunlun PyTorch compatibility"""

            def __init__(self, op):
                self.op = op

            def __call__(self, *args, **kwargs):
                return self.op(*args, **kwargs)

        mock_module = types.ModuleType("torch._higher_order_ops.auto_functionalized")
        mock_module.auto_functionalized = MockAutoFunctionalized
        mock_module.AutoFunctionalized = MockAutoFunctionalized

        sys.modules["torch._higher_order_ops.auto_functionalized"] = mock_module
        higher_order_ops.auto_functionalized = MockAutoFunctionalized

        if hasattr(higher_order_ops, "__all__"):
            if "auto_functionalized" not in higher_order_ops.__all__:
                higher_order_ops.__all__ = list(higher_order_ops.__all__) + [
                    "auto_functionalized"
                ]
    except Exception:
        pass


def apply_early_patches():
    """Apply early patches at module load time (before any vllm imports)

    These patches must be applied before any vllm module is imported because they
    modify third-party modules that vllm depends on.
    """
    global _EARLY_PATCHES_APPLIED
    if _EARLY_PATCHES_APPLIED:
        return

    _patch_openai_compat()
    _patch_torch_higher_order_ops()

    _EARLY_PATCHES_APPLIED = True


# ============================================================================
# File Patches - Applied during plugin registration
# ============================================================================


# Get vllm installation path dynamically
def _get_vllm_path():
    """Get the vllm installation path"""
    try:
        import vllm

        return os.path.dirname(vllm.__file__)
    except ImportError:
        # Fallback to site-packages
        for path in sys.path:
            vllm_path = os.path.join(path, "vllm")
            if os.path.isdir(vllm_path):
                return vllm_path
        return None


_PATCHES_APPLIED = False


def _patch_file(filepath, old_content, new_content, description=""):
    """Patch a file by replacing content"""
    try:
        if not os.path.exists(filepath):
            logger.debug(f"[KUNLUN] File not found: {filepath}")
            return False

        with open(filepath, "r") as f:
            content = f.read()

        # Check if already patched
        if "KUNLUN PATCH" in content:
            logger.debug(f"[KUNLUN] Already patched: {filepath}")
            return True

        if old_content in content:
            content = content.replace(old_content, new_content)
            with open(filepath, "w") as f:
                f.write(content)
            logger.info(
                f"[KUNLUN] Patched: {os.path.basename(filepath)} ({description})"
            )
            return True
        else:
            logger.debug(f"[KUNLUN] Content not found in: {filepath}")
            return False
    except Exception as e:
        logger.warning(f"[KUNLUN] Failed to patch {filepath}: {e}")
        return False


def _patch_decorators(vllm_path):
    """Patch vllm/compilation/decorators.py - patched_inline_call signature

    Problem: kunlun PyTorch 2.5.1's InliningInstructionTranslator.inline_call_
    has signature (parent, func, args, kwargs) with 4 parameters,
    but vLLM 0.15.1's patched_inline_call expects only (self_: Any) signature.
    """
    filepath = os.path.join(vllm_path, "compilation", "decorators.py")

    old = """        def patched_inline_call(self_: Any) -> Any:
            code = self_.f_code
            self.compilation_config.traced_files.add(code.co_filename)
            return inline_call(self_)"""

    new = """        # KUNLUN PATCH: Modified signature for kunlun PyTorch 2.5.1 compatibility
        # Original signature: (self_: Any) -> Any
        # Kunlun PyTorch 2.5.1 signature: (parent, func, args, kwargs) -> Any
        def patched_inline_call(parent, func, args, kwargs) -> Any:
            code = parent.f_code
            self.compilation_config.traced_files.add(code.co_filename)
            return inline_call(parent, func, args, kwargs)"""

    return _patch_file(filepath, old, new, "patched_inline_call signature")


def _patch_backends(vllm_path):
    """Patch vllm/compilation/backends.py - VllmBackend.__call__ signature

    Problem: kunlun PyTorch 2.5.1 passes extra 'options' kwarg to backend callable,
    which vLLM 0.15.1's VllmBackend.__call__ doesn't accept.
    """
    filepath = os.path.join(vllm_path, "compilation", "backends.py")

    old = "def __call__(self, graph: fx.GraphModule, example_inputs: Sequence[Any]) -> Any:"
    new = "def __call__(self, graph: fx.GraphModule, example_inputs: Sequence[Any], **kwargs) -> Any:  # KUNLUN PATCH: Added **kwargs for kunlun PyTorch compatibility"

    return _patch_file(filepath, old, new, "VllmBackend.__call__ signature")


def _patch_piecewise_backend(vllm_path):
    """Patch vllm/compilation/piecewise_backend.py - bundled_autograd_cache

    Problem: kunlun PyTorch 2.5.1's torch._functorch.config doesn't have
    'bundled_autograd_cache' configuration option.
    """
    filepath = os.path.join(vllm_path, "compilation", "piecewise_backend.py")

    try:
        with open(filepath, "r") as f:
            content = f.read()

        if "KUNLUN PATCH" in content:
            logger.debug(f"[KUNLUN] Already patched: {filepath}")
            return True

        modified = False

        # Add contextlib import
        if "import contextlib" not in content:
            content = content.replace(
                "import io", "import io\nimport contextlib  # KUNLUN PATCH"
            )
            modified = True

        # Replace bundled_autograd_cache
        old1 = 'with torch._functorch.config.patch("bundled_autograd_cache", True):'
        new1 = "with contextlib.nullcontext():  # KUNLUN PATCH: bundled_autograd_cache not supported in kunlun PyTorch"

        old2 = 'torch._functorch.config.patch("bundled_autograd_cache", True),'
        new2 = "contextlib.nullcontext(),  # KUNLUN PATCH: bundled_autograd_cache not supported in kunlun PyTorch"

        if old1 in content:
            content = content.replace(old1, new1)
            modified = True
        if old2 in content:
            content = content.replace(old2, new2)
            modified = True

        if modified:
            with open(filepath, "w") as f:
                f.write(content)
            logger.info(
                "[KUNLUN] Patched: piecewise_backend.py (bundled_autograd_cache)"
            )
            return True
        return False
    except Exception as e:
        logger.warning(f"[KUNLUN] Failed to patch {filepath}: {e}")
        return False


def _patch_attention_layer(vllm_path):
    """Patch vllm/attention/layer.py - torch.Size compatibility

    Problem: torch.Size is not supported as a UserDefinedClassVariable
    in kunlun PyTorch dynamo during graph compilation.
    """
    filepath = os.path.join(vllm_path, "attention", "layer.py")

    old = """                output_shape = torch.Size(
                    (num_tokens, self.num_heads * self.head_size_v)
                )"""

    new = """                # KUNLUN PATCH: Use tuple instead of torch.Size for dynamo compatibility
                output_shape = (
                    (num_tokens, self.num_heads * self.head_size_v)
                )"""

    return _patch_file(filepath, old, new, "torch.Size compatibility")


def apply_all_patches():
    """Apply all kunlun compatibility patches to vllm

    This function is idempotent - running it multiple times is safe.
    Patches are only applied if the target content hasn't been patched yet.
    """
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return True

    vllm_path = _get_vllm_path()
    if not vllm_path:
        logger.warning("[KUNLUN] Could not find vllm installation path")
        return False

    logger.info("[KUNLUN] Applying kunlun PyTorch compatibility patches...")
    logger.debug(f"[KUNLUN] VLLM path: {vllm_path}")

    results = []
    results.append(_patch_decorators(vllm_path))
    results.append(_patch_backends(vllm_path))
    results.append(_patch_piecewise_backend(vllm_path))
    results.append(_patch_attention_layer(vllm_path))

    _PATCHES_APPLIED = True

    if any(results):
        logger.info("[KUNLUN] Patches applied successfully")
    else:
        logger.debug("[KUNLUN] All patches were already applied")

    return True


if __name__ == "__main__":
    # Enable logging for standalone execution
    logging.basicConfig(level=logging.INFO)
    apply_all_patches()
