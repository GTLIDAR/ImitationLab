"""Refuse config overrides that name no declared field.

Isaac Lab's Hydra path merges CLI overrides into the config object with
``setattr``, so ``agent.ppo.some_typo=false`` becomes an attribute nobody
reads and the run proceeds as if the setting were applied. Five fine-tune
campaigns in 2026-09 passed ``agent.ppo.update_normalizers_after_rollout=false``
before that key existed in RLOpt; their READMEs claimed a frozen normalizer
that never was. ``env.*`` has the same trap: ``env.reset_fail_shar=0.3`` ran a
control arm that looked like a treatment arm. This check walks the resolved
config and raises on any public attribute that is not declared, so the mistake
fails at submission instead of at export.
"""

from __future__ import annotations

import dataclasses
from typing import Any


def unknown_config_keys(config: Any, *, prefix: str = "agent") -> list[str]:
    """Dotted paths of instance attributes that are not declared.

    Recurses into nested dataclass values. Private names (leading underscore)
    are skipped: config classes use them for runtime bindings. ``ClassVar``
    members count as declared. ``dataclasses.fields`` leaves them out, but
    ``configclass`` copies every class member onto the instance, so
    ``PhysxCfg.launcher_type`` appears in ``vars`` of each physics config.
    """
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        return []
    declared = set(config.__dataclass_fields__)
    unknown: list[str] = []
    for name, value in vars(config).items():
        if name.startswith("_"):
            continue
        path = f"{prefix}.{name}"
        if name not in declared:
            unknown.append(path)
            continue
        unknown.extend(unknown_config_keys(value, prefix=path))
    return sorted(unknown)


def assert_no_unknown_config_keys(config: Any, *, prefix: str = "agent") -> None:
    unknown = unknown_config_keys(config, prefix=prefix)
    if unknown:
        raise ValueError(
            "config overrides name fields that do not exist (typo, a key this "
            "task does not declare, or a key the installed RLOpt does not "
            "implement): " + ", ".join(unknown)
        )
