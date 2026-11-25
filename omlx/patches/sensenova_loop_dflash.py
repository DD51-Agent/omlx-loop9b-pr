# SPDX-License-Identifier: Apache-2.0
"""SenseNova-Loop DFlash2 speculative decoding adaptation (5-feature fc).

The SenseNova-Loop-9B-DFlash2 draft checkpoint carries a 5-feature ``fc``
input: ``target_feature_sources = [3, 27, 51, 54, "final_postnorm"]``.
The base ``DFlashDraftModel`` computes the fc input width from
``len(target_layer_ids)`` (4 -> 16384), but the checkpoint's ``fc.weight``
is ``[4096, 20480]`` (5 -> 20480).  This module adapts the DFlash2 drafter
to the 5-feature convention and teaches the SenseNova-Loop backbone to
capture the final post-norm hidden as the 5th feature.

All changes are applied at runtime via monkey-patching so the only source
edit required is a single ``apply()`` call from
``utils/model_loading.py`` after the SenseNova-Loop architecture registers.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

# The marker used inside ``target_feature_sources`` for the backbone's final
# post-norm hidden state.
_FINAL_POSTNORM = "final_postnorm"

# SenseNova-Loop model types that carry a DFlash2 5-feature drafter.
_SENSENOVA_LOOP_MODEL_TYPES = ("sensenova_loop", "sensenova_loop_text")

# ---------------------------------------------------------------------------
# 1. DFlash2Config: parse target_feature_sources from the checkpoint config.
# ---------------------------------------------------------------------------


def _patch_dflash2_config() -> None:
    """Teach ``DFlash2Config.from_dict`` to carry ``target_feature_sources``."""
    try:
        from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config
    except Exception as e:  # pragma: no cover
        logger.debug("DFlash2 config patch skipped: %s", e)
        return

    if getattr(DFlash2Config, "_sl_dflash_tfs_patched", False):
        return

    orig_from_dict = DFlash2Config.from_dict.__func__

    def _patched_from_dict(cls, params):
        config = orig_from_dict(cls, params)
        tfs = params.get("target_feature_sources")
        if tfs is None:
            dflash = params.get("dflash_config") or {}
            tfs = dflash.get("target_feature_sources")
        if tfs is not None:
            config.target_feature_sources = list(tfs)
        return config

    DFlash2Config.from_dict = classmethod(_patched_from_dict)
    DFlash2Config.from_hf_dict = classmethod(_patched_from_dict)
    setattr(DFlash2Config, "_sl_dflash_tfs_patched", True)
    logger.debug("DFlash2Config.target_feature_sources parse patched")


# ---------------------------------------------------------------------------
# 2. DFlashDraftModel.__init__: fc input width from target_feature_sources.
# ---------------------------------------------------------------------------


def _patch_dflash_draft_model() -> None:
    """Make ``DFlashDraftModel.__init__`` size ``fc`` from the feature count."""
    try:
        from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import (
            DFlashDraftModel,
        )
    except Exception as e:  # pragma: no cover
        logger.debug("DFlashDraftModel patch skipped: %s", e)
        return

    if getattr(DFlashDraftModel, "_sl_dflash_fc_patched", False):
        return

    orig_init = DFlashDraftModel.__init__

    def _patched_init(self, config):
        tfs = getattr(config, "target_feature_sources", None)
        if tfs is not None and len(tfs) > len(config.target_layer_ids):
            # Temporarily widen target_layer_ids so the base __init__ sizes
            # fc as len(tfs)*hidden_size; restore afterwards so capture
            # logic still uses the real 4 layer ids.
            saved = config.target_layer_ids
            config.target_layer_ids = list(tfs)
            try:
                orig_init(self, config)
            finally:
                config.target_layer_ids = saved
            return
        orig_init(self, config)

    DFlashDraftModel.__init__ = _patched_init
    setattr(DFlashDraftModel, "_sl_dflash_fc_patched", True)
    logger.debug("DFlashDraftModel fc width patched")


# ---------------------------------------------------------------------------
# 3. validate_dflash_target: count execution layers for SenseNova-Loop.
# ---------------------------------------------------------------------------


def _patch_validate_dflash_target() -> None:
    """Teach ``validate_dflash_target`` the 56-slot loop execution plan.

    SenseNova-Loop maps 32 physical layers to 56 execution slots via
    ``_loop_plan``.  The drafter's ``num_target_layers`` counts execution
    slots, so the layer-count check must compare against ``len(_loop_plan)``
    instead of the physical layer count.

    Delegates to the **stock** validator (zero behavioural drift for other
    models) and only overrides ``config.num_target_layers`` to ``len(_loop_plan)``
    while the stock check runs.  This replaces the previous full rewrite
    (74 lines) with a ~20-line wrapper: the hidden/vocab/rollback checks stay
    in the upstream implementation, so upstream changes never silently drift.
    """
    try:
        from mlx_vlm.speculative.drafters import compatibility as compat
    except Exception as e:  # pragma: no cover
        logger.debug("validate_dflash_target patch skipped: %s", e)
        return

    orig = getattr(compat, "validate_dflash_target", None)
    if orig is None or getattr(orig, "_sl_dflash_patched", False):
        return

    def _patched(config, target_model, algorithm):
        try:
            language_model = compat._target_metadata(target_model)[0]
        except Exception:
            language_model = None
        loop_plan = getattr(language_model, "_loop_plan", None)
        if loop_plan is None and language_model is not None:
            loop_plan = getattr(getattr(language_model, "model", None), "_loop_plan", None)
        if loop_plan is None:
            # Not SenseNova-Loop: stock validator unchanged.
            return orig(config, target_model, algorithm)
        saved = config.num_target_layers
        config.num_target_layers = len(loop_plan)
        try:
            return orig(config, target_model, algorithm)
        finally:
            config.num_target_layers = saved

    compat.validate_dflash_target = _patched
    # ``dflash2``/``dspark`` bind the symbol via direct name import, so a
    # module-attribute swap on ``compat`` alone does not reach them.
    # Force-import each consumer (idempotent) and rebind the module-level
    # name so the patched version actually runs.
    for _mod_name in (
        "mlx_vlm.speculative.drafters.dflash2.dflash2",
        "mlx_vlm.speculative.drafters.dspark.dspark",
    ):
        try:
            _mod = importlib.import_module(_mod_name)
            if getattr(_mod, "validate_dflash_target", None) is not _patched:
                _mod.validate_dflash_target = _patched
        except Exception as _e:  # pragma: no cover
            logger.debug("validate_dflash_target rebind %s failed: %s", _mod_name, _e)
    setattr(_patched, "_sl_dflash_patched", True)
    logger.debug("validate_dflash_target patched for loop plan")


# ---------------------------------------------------------------------------
# 4. dflash_batched_supported: allow SenseNova-Loop on the batched engine.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. SenseNova-Loop TextModel.__call__: capture the final post-norm hidden.
# ---------------------------------------------------------------------------


def _patch_text_model_capture() -> None:
    """Make the SenseNova-Loop ``TextModel`` emit the final post-norm hidden.

    When a DFlash2 drafter with ``target_feature_sources`` containing
    ``"final_postnorm"`` is attached, the backbone's ``hidden_states`` list
    gains the post-norm final hidden as the trailing capture so the drafter's
    ``fc`` can consume 5 features.
    """
    # The runtime module is the one registered under ``mlx_lm.models`` by
    # ``sensenova_loop.apply()`` (spec_from_file_location re-execs this file
    # as a separate module object).  Patching the ``omlx.patches...`` alias
    # would be a no-op: the model loads from ``mlx_lm.models.sensenova_loop``.
    try:
        from mlx_lm.models import sensenova_loop as sl
        TextModel = sl.TextModel
    except Exception as e:  # pragma: no cover
        logger.warning("TextModel capture patch skipped: %s", e)
        return

    if getattr(TextModel, "_sl_dflash_capture_patched", False):
        return

    # ``apply()`` runs before ``apply_mlx_lm_mtp_patch`` in the load path, so
    # the runtime ``TextModel.__call__`` at this point may still be the stock
    # signature (no ``return_hidden``/``capture_layer_ids``).  qwen35_model
    # retroactively patches ``qwen3_5.TextModel.__call__`` and that propagates
    # to the ``sensenova_loop.TextModel`` subclass via MRO; ensure it is
    # applied BEFORE we capture ``orig_call`` so the wrapper forwards the
    # patched signature.  Idempotent: already-applied is a no-op.
    try:
        from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch

        apply_mlx_lm_mtp_patch()
    except Exception as e:  # pragma: no cover
        logger.debug("pre-apply mlx_lm_mtp patch skipped: %s", e)

    orig_call = TextModel.__call__

    def _patched_call(self, *args, **kwargs):
        result = orig_call(self, *args, **kwargs)
        # Stock TextModel.__call__ returns SimpleNamespace(logits, gdn_states)
        # with NO hidden_states; the exact 5-feature capture comes from
        # _SL_CAPTURE (stashed by the inner-model wrapper).
        if not hasattr(result, "logits"):
            return result
        drafter = getattr(self, "_omlx_drafter", None)
        if drafter is None:
            return result
        tfs = getattr(drafter, "target_feature_sources", None)
        if not tfs or _FINAL_POSTNORM not in tfs:
            return result
        cls = kwargs.get("capture_layer_ids")
        if cls is None or len(cls) != 4:
            return result
        capture = _SL_CAPTURE.pop(id(getattr(self, "model", None)), None)
        if capture is None or len(capture) != len(cls) + 1:
            return result
        # hidden = [c1..cN, final_postnorm] (exactly len(cls)+1 = 5).
        result.hidden_states = capture
        result._sl_dflash_capture = list(capture)
        logger.debug(
            "SL-DFlash: capture injected (n=%d) tli=%s",
            len(result.hidden_states),
            getattr(drafter, "target_layer_ids", None),
        )
        return result

    TextModel.__call__ = _patched_call
    setattr(TextModel, "_sl_dflash_capture_patched", True)
    logger.debug("Sensenova-Loop TextModel final_postnorm capture patched")


# ---------------------------------------------------------------------------
# 5b. Inner model capture: stash the final post-norm hidden on capture_sink.
# ---------------------------------------------------------------------------


# id(capture_sink) -> post-norm final hidden.  A plain ``capture_sink`` is a
# list and cannot carry an attribute, so stash by identity in a module-level
# dict.  The drafter.observe wrapper pops it in the same call stack.
_SL_NORMED = {}
# Keyed by id(inner Qwen3_5TextModel): the exact 5-feature capture
# [c1..cN, final_postnorm] stashed by the inner-model wrapper so the outer
# TextModel wrapper can rebuild ``hidden_states`` without any source edit.
_SL_CAPTURE = {}


def _patch_inner_model_capture() -> None:
    """Capture the final post-norm hidden on the inner text model.

    The outer ``TextModel.__call__`` observe branch computes
    ``normed = self.model.norm(hidden)`` and feeds it as the 5th feature.
    The inner ``Qwen3_5TextModel.__call__`` returns the pre-norm final
    hidden (slot 55); intercept it there and stash the post-norm hidden so a
    wrapped ``drafter.observe`` can append it without any edit to the core
    ``qwen35_model.py``.
    """
    try:
        from mlx_lm.models import sensenova_loop as sl

        InnerModel = sl.Qwen3_5TextModel
    except Exception as e:  # pragma: no cover
        logger.warning("inner model capture patch skipped: %s", e)
        return

    if getattr(InnerModel, "_sl_inner_capture_patched", False):
        return

    orig_call = InnerModel.__call__

    def _patched_call(self, *args, **kwargs):
        capture_sink = kwargs.get("capture_sink")
        result = orig_call(self, *args, **kwargs)
        if capture_sink is not None and hasattr(result, "shape"):
            norm = getattr(self, "norm", None)
            if norm is not None:
                post = norm(result)
                # Observe path: drafter.observe reads the post-norm hidden.
                _SL_NORMED[id(capture_sink)] = post
                # Verify path: the outer TextModel wrapper rebuilds
                # ``hidden_states`` from this exact 5-feature capture.
                layers = list(capture_sink)
                if layers:
                    _SL_CAPTURE[id(self)] = layers + [post]
        return result

    InnerModel.__call__ = _patched_call
    setattr(InnerModel, "_sl_inner_capture_patched", True)
    logger.debug("SenseNova-Loop inner model final_postnorm capture patched")


# ---------------------------------------------------------------------------
# 5c. drafter.observe: append the final post-norm feature.
# ---------------------------------------------------------------------------


def _patch_drafter_observe() -> None:
    """Append the final post-norm feature in ``drafter.observe``.

    The observe branch feeds the committed token's 4 layer captures to the
    drafter; DFlash2's fc needs 5 features (4 layer hiddens + final
    post-norm).  The inner model capture patch stashes the post-norm hidden
    on ``capture_sink``; read it here and append it so the drafter's
    ``seed`` sees the same width the verify/draft paths use.
    """
    try:
        from omlx.speculative.dflash.drafter import DFlashDrafter
    except Exception as e:  # pragma: no cover
        logger.debug("drafter.observe patch skipped: %s", e)
        return

    if getattr(DFlashDrafter, "_sl_observe_patched", False):
        return

    orig_observe = DFlashDrafter.observe

    def _patched_observe(self, uids, captured):
        # Append the stashed post-norm hidden (5th feature) when present.
        # Only DFlash2 (final_postnorm) drafters carry it; pop here (never
        # re-inject) since the inner capture patch stashed it by id() in the
        # same call stack.
        extra = _SL_NORMED.pop(id(captured), None)
        if extra is not None:
            try:
                return orig_observe(self, uids, list(captured) + [extra])
            except Exception:
                pass
        return orig_observe(self, uids, captured)

    DFlashDrafter.observe = _patched_observe
    setattr(DFlashDrafter, "_sl_observe_patched", True)
    logger.debug("drafter.observe appends final_postnorm for SenseNova-Loop")


# ---------------------------------------------------------------------------
# 6. batch_generator._call_backbone_impl: accept the extra final_postnorm.
# ---------------------------------------------------------------------------


def _drafter_needs_final_postnorm(model) -> bool:
    """True when an attached block drafter wants the final post-norm capture."""
    try:
        from omlx.patches.mlx_lm_mtp import batch_generator as _bg

        drafter = _bg._drafter_for(model)
    except Exception:
        return False
    if drafter is None:
        return False
    tfs = getattr(drafter, "target_feature_sources", None)
    return bool(tfs and _FINAL_POSTNORM in tfs)


def _patched_call_backbone_impl(model, inputs, cache, n_confirmed, capture_layer_ids):
    """Drop-in for ``batch_generator._call_backbone_impl``.

    Delegates to the stock implementation for every non-SenseNova-Loop
    forward so upstream behaviour is preserved bit-for-bit.  Only when the
    attached block drafter declares ``target_feature_sources`` with
    ``final_postnorm`` does the custom capture-count path run: the
    SenseNova-Loop backbone emits 4 layer captures plus the final post-norm
    hidden, so the count is ``len(capture_layer_ids) + 1``.
    """
    from omlx.patches.mlx_lm_mtp import batch_generator as _bg

    if not _drafter_needs_final_postnorm(model) or not capture_layer_ids:
        stock = getattr(_bg, "_call_backbone_impl_stock", None)
        if stock is not None:
            return stock(model, inputs, cache, n_confirmed, capture_layer_ids)
        # Fallback: no stock reference (should not happen) -> plain forward.
        kwargs = {"cache": cache, "return_hidden": True}
        if n_confirmed:
            kwargs["n_confirmed"] = n_confirmed
        if capture_layer_ids:
            kwargs["capture_layer_ids"] = list(capture_layer_ids)
        result = model(inputs, **kwargs)
        if hasattr(result, "logits"):
            return result.logits, None, None, None
        if isinstance(result, tuple) and len(result) >= 2:
            return (*result[:3], None)
        raise TypeError(f"backbone returned unexpected shape: {type(result).__name__}")

    kwargs = {"cache": cache, "return_hidden": True}
    if n_confirmed:
        kwargs["n_confirmed"] = n_confirmed
    if capture_layer_ids:
        kwargs["capture_layer_ids"] = list(capture_layer_ids)
    exact_verify = bool(n_confirmed and _bg._exact_verify_host(model) is not None)
    _bg._rollback_mod.set_undo_armed(True)
    _bg._set_verify_qmm_armed(
        not exact_verify,
        row_exact=inputs.shape[0] == 1 and _bg._row_exact_verify(model),
    )
    _bg._set_dspark_target_verify(model, exact_verify)
    try:
        result = model(inputs, **kwargs)
    finally:
        if exact_verify:
            _bg._set_dspark_target_verify(model, False)
        _bg._set_verify_qmm_armed(False)
        _bg._rollback_mod.set_undo_armed(False)

    if hasattr(result, "logits") and hasattr(result, "hidden_states"):
        hidden = result.hidden_states
        captured = None
        if isinstance(hidden, list):
            if capture_layer_ids:
                # Prefer the exact 5-feature capture stashed by _patched_call;
                # hidden[:-1] cannot distinguish a trailing pre-norm from a
                # missing one (return_hidden=False strips it).
                exact = getattr(result, "_sl_dflash_capture", None)
                if exact is not None and len(exact) == len(capture_layer_ids) + 1:
                    captured = list(exact)
                else:
                    captured = list(hidden[:-1])
                # final_postnorm injection adds exactly one extra capture;
                # a plain capture keeps the original count.
                nfeat = len(capture_layer_ids) + (
                    1 if len(captured) == len(capture_layer_ids) + 1 else 0
                )
                if len(captured) != nfeat:
                    raise TypeError(
                        "backbone returned %d captures for %d requested layers"
                        % (len(captured), len(capture_layer_ids))
                    )
            hidden = hidden[-1] if hidden else None
        rollback_state = getattr(result, "gdn_states", None)
        try:
            from mlx_vlm.speculative.cache_state import SpeculativeCacheTransaction
        except ImportError:
            SpeculativeCacheTransaction = ()
        if not n_confirmed and isinstance(
            rollback_state, SpeculativeCacheTransaction
        ):
            model.rollback_speculative_cache(
                cache,
                rollback_state,
                [inputs.shape[1] - 1] * inputs.shape[0],
                inputs.shape[1],
            )
            rollback_state = None
        return result.logits, hidden, rollback_state, captured
    if isinstance(result, tuple):
        if capture_layer_ids:
            raise TypeError("backbone tuple output cannot carry layer captures")
        if len(result) == 3:
            return (*result, None)
        if len(result) == 2:
            return result[0], result[1], None, None
    raise TypeError(f"backbone returned unexpected shape: {type(result).__name__}")


def _patch_call_backbone_impl() -> None:
    """Swap the capture-count check so DFlash2's 5th feature is accepted.

    The patch keeps a reference to the **stock** implementation so the
    wrapper can delegate every non-loop9b forward to it (zero behavioural
    drift for other models); only SenseNova-Loop with a final_postnorm
    drafter takes the custom capture-count path.
    """
    try:
        from omlx.patches.mlx_lm_mtp import batch_generator as _bg
    except Exception as e:  # pragma: no cover
        logger.debug("_call_backbone_impl patch skipped: %s", e)
        return

    impl = getattr(_bg, "_call_backbone_impl", None)
    if impl is None or getattr(impl, "_sl_dflash_patched", False):
        return

    # Preserve the stock implementation for delegation.
    _bg._call_backbone_impl_stock = impl
    _bg._call_backbone_impl = _patched_call_backbone_impl
    setattr(_patched_call_backbone_impl, "_sl_dflash_patched", True)
    logger.debug("_call_backbone_impl capture-count check relaxed for DFlash2")


# ---------------------------------------------------------------------------
# 7. load_dflash_drafter: stamp target_feature_sources on the drafter.
# ---------------------------------------------------------------------------


def _patch_load_dflash_drafter() -> None:
    """Propagate ``target_feature_sources`` to the DFlashDrafter instance.

    The drafter needs to know it must consume 5 features (4 layer hiddens +
    final post-norm) so the backbone capture hook and the capture-count
    check are triggered for DFlash2 checkpoints.
    """
    try:
        from omlx.speculative import dflash as _dflash_mod
    except Exception as e:  # pragma: no cover
        logger.debug("load_dflash_drafter patch skipped: %s", e)
        return

    load_fn = getattr(_dflash_mod, "load_dflash_drafter", None)
    if load_fn is None or getattr(load_fn, "_sl_dflash_patched", False):
        return

    orig = load_fn

    def _patched(path, target_model, **kwargs):
        drafter = orig(path, target_model, **kwargs)
        try:
            tfs = getattr(drafter.model.config, "target_feature_sources", None)
        except Exception as e:  # pragma: no cover
            logger.debug("target_feature_sources stamp failed: %s", e)
            return drafter
        if tfs is not None:
            drafter.target_feature_sources = list(tfs)
        return drafter

    _dflash_mod.load_dflash_drafter = _patched
    # ``engine_pool`` binds ``load_dflash_drafter`` via a ``from ... import``
    # statement, so it keeps the ORIGINAL function.  Rebind there too or the
    # drafter never gets ``target_feature_sources`` stamped.
    try:
        import omlx.engine_pool as ep

        ep.load_dflash_drafter = _patched
    except Exception as e:  # pragma: no cover
        logger.debug("engine_pool load_dflash_drafter rebind skipped: %s", e)
    setattr(_patched, "_sl_dflash_patched", True)
    logger.debug("load_dflash_drafter stamps target_feature_sources")



# ---------------------------------------------------------------------------
# 8. drafter_target.apply: keep the generic capture hooks off SenseNova-Loop.
# ---------------------------------------------------------------------------


def _patch_drafter_target_apply() -> None:
    """Keep ``drafter_target.apply`` from layering generic capture hooks.

    SenseNova-Loop carries its own final_postnorm capture (the TextModel
    patch injects the post-norm final hidden so ``_call_backbone_impl`` sees
    5 features).  Layering the generic ``_Sink``/``_wrap_layer`` hooks here
    would produce a 4-layer (observe) or 6-layer (return_hidden, after the
    injected final_postnorm) capture and break the drafter's fc
    concatenation.  This mirrors the source edit that was previously in
    ``drafter_target.apply``; keeping it here as a runtime patch lets the
    core file stay stock.
    """
    try:
        from omlx.patches.mlx_lm_mtp import drafter_target as _dt
    except Exception as e:  # pragma: no cover
        logger.debug("drafter_target.apply patch skipped: %s", e)
        return

    orig = getattr(_dt, "apply", None)
    if orig is None or getattr(orig, "_sl_dflash_patched", False):
        return

    def _patched_apply(model):
        # SenseNova-Loop: skip the generic hooks (mirrors the removed source
        # guard in drafter_target.apply).
        cls = type(model).__name__
        mod = str(getattr(model, "__module__", ""))
        if "sensenova_loop" in cls or "sensenova_loop" in mod:
            return False
        return orig(model)

    _dt.apply = _patched_apply
    # engine/vlm.py and engine/batched.py import the symbol via
    # ``from ..patches.mlx_lm_mtp import drafter_target`` and call
    # ``drafter_target.apply(...)``, so the module attribute swap above is
    # enough; no per-module rebind needed (the attribute is read through the
    # module object each call).
    setattr(_patched_apply, "_sl_dflash_patched", True)
    logger.debug("drafter_target.apply keeps generic hooks off SenseNova-Loop")


# ---------------------------------------------------------------------------
# 9. Scheduler._dflash_seed_prefill: feed the drafter exactly N+1 features.
# ---------------------------------------------------------------------------


def _patch_dflash_seed_prefill() -> None:
    """Teach ``Scheduler._dflash_seed_prefill`` the 5-feature slice.

    SenseNova-Loop's final_postnorm capture injects the post-norm final
    hidden plus the pre-norm hidden, so ``hidden_states`` holds
    ``[caps..., fp_post, fp_pre]``.  The block drafter's fc consumes
    ``len(target_layer_ids) + 1`` features (the layer captures plus the
    post-norm final).  Keep exactly that many so ``seed_request`` feeds the
    drafter the same width ``_call_backbone_impl`` uses.  This mirrors the
    source edit that was previously in ``Scheduler._dflash_seed_prefill``.
    """
    try:
        from omlx.scheduler import Scheduler
    except Exception as e:  # pragma: no cover
        logger.debug("Scheduler._dflash_seed_prefill patch skipped: %s", e)
        return

    orig = getattr(Scheduler, "_dflash_seed_prefill", None)
    if orig is None or getattr(orig, "_sl_dflash_patched", False):
        return

    def _patched_seed_prefill(self, request, output, keep_from, position):
        from omlx.patches.mlx_lm_mtp.batch_generator import (
            _drafter_for as _block_drafter_for,
        )

        drafter = _block_drafter_for(self.model)
        hidden = getattr(output, "hidden_states", None)
        if drafter is None or not hidden:
            return
        # Only SenseNova-Loop (final_postnorm) needs the slice; other
        # drafters keep the full hidden_states width.
        tfs = getattr(drafter, "target_feature_sources", None) or ()
        if _FINAL_POSTNORM in tfs:
            nfeat = len(getattr(drafter, "target_layer_ids", []) or []) + 1
            hidden = hidden[:nfeat]
        drafter.seed_request(
            request.request_id,
            [layer[:, keep_from:] for layer in hidden],
            position,
        )

    Scheduler._dflash_seed_prefill = _patched_seed_prefill
    setattr(_patched_seed_prefill, "_sl_dflash_patched", True)
    logger.debug("Scheduler._dflash_seed_prefill slices to N+1 for SenseNova-Loop")


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def apply() -> bool:
    """Apply all SenseNova-Loop DFlash2 adaptations (idempotent)."""
    _patch_dflash2_config()
    _patch_dflash_draft_model()
    _patch_validate_dflash_target()
    _patch_text_model_capture()
    _patch_inner_model_capture()
    _patch_drafter_observe()
    _patch_call_backbone_impl()
    # Debug-only confirmation the inner-model wrapper bound.
    try:
        from mlx_lm.models import sensenova_loop as _slv
        logger.debug(
            "SL-DFlash: inner patched=%s",
            getattr(_slv.Qwen3_5TextModel, "_sl_inner_capture_patched", False),
        )
    except Exception as _e:
        logger.debug("SL-DFlash: inner verify failed: %s", _e)
    _patch_load_dflash_drafter()
    _patch_drafter_target_apply()
    _patch_dflash_seed_prefill()
    return True


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.DEBUG)
    print("Applying...")
    apply()
    print("Applied.")
