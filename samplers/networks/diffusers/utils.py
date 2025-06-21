from ..base import DiffusionType

_SCHEDULER_MAP = {
    "ddpm": DiffusionType.VARIANCE_PRESERVING,
    "pndm": DiffusionType.VARIANCE_PRESERVING,
    "ddim": DiffusionType.VARIANCE_PRESERVING,
    "euler": DiffusionType.VARIANCE_EXPLODING,
    "k_dpm": DiffusionType.VARIANCE_EXPLODING,
}


def infer_diffusion_type(pipeline) -> DiffusionType:
    """Infer the diffusion flavor of a HuggingFace DiffusionPipeline.

    Checks scheduler class name against known patterns,
    then looks for .sde on continuous-time pipelines,
    and falls back to UNKNOWN.

    Args:
        pipeline: any HF DiffusionPipeline with .scheduler and optional .sde

    Returns:
        DiffusionType enum member
    """
    # 1. match scheduler name
    sched_name = type(pipeline.scheduler).__name__.lower()
    for key, dtype in _SCHEDULER_MAP.items():
        if key in sched_name:
            return dtype

    # 2. check continuous-time SDE pipelines
    sde_attr = getattr(pipeline, "sde", None)
    if isinstance(sde_attr, str):
        if "vesde" in sde_attr.lower():
            return DiffusionType.VARIANCE_EXPLODING
        if "vpsde" in sde_attr.lower():
            return DiffusionType.VARIANCE_PRESERVING

    # 3. optional: detect sub-VP by config flags
    cfg = getattr(pipeline.scheduler, "config", None)
    if cfg and getattr(cfg, "clip_sample", False):
        return DiffusionType.SUB_VARIANCE_PRESERVING

    return DiffusionType.UNKNOWN
