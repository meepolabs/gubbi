"""Lint env-var contract drift between pydantic Settings and compose files.

Statically walks the Settings AST + compose YAML to verify that every
required Settings field has its env-var declared in the docker-compose env
block (as a bare passthrough, KEY=value, or ${VAR} variable reference).

Exit 0 when the contract is satisfied.  Exit 1 on any drift.

Pure functions are exposed for testing; __main__ is the CLI entry.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Any

import yaml


def env_var_name(field_name: str, alias: str | None, prefix: str) -> str:
    """Return the environment-variable name for a Settings field.

    If the field has a validation_alias, the alias is the env var name.
    Otherwise use ``{prefix}{field_name.upper()}`` (standard pydantic-settings behaviour).
    """
    if alias:
        return alias
    return prefix + field_name.upper()


def _is_field_call(node: ast.expr) -> bool:
    """Return True if *node* is a ``Field(...)`` call (any module qualification)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "Field") or (
        isinstance(func, ast.Attribute) and func.attr == "Field"
    )


def parse_settings(path: str, cls_name: str = "Settings") -> dict[str, Any]:
    """Parse a pydantic Settings class and return required/optional field info.

    Handles nested sub-models: when a Settings field's annotation is a
    class defined in the same file (e.g. DbConfig, AuthConfig), it expands
    that model's fields using the double-underscore delimiter
    (JOURNAL_DB__APP_URL) and also records any flat legacy aliases extracted
    from ``_FLAT_TO_NESTED_ENV`` so compose checks can match either form.

    Returns::

        {
            "fields": {
                "logical_name": {
                    "env_var": str,       # canonical nested form
                    "aliases": list[str], # flat legacy names (may be empty)
                    "required": bool,
                    "alias": str | None,
                }
            },
            "prefix": str,
        }

    A field is REQUIRED when it has no Python-level default (no AST Assign
    default and no Field() RHS).  Fields with ``= ""`` or ``= 0`` etc.
    are optional regardless of pydantic model-validator logic.
    """
    source = Path(path).read_text()
    tree = ast.parse(source, filename=path)

    # Collect all class defs in the module for nested-model expansion.
    all_classes: dict[str, ast.ClassDef] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            all_classes[node.name] = node

    settings_class = all_classes.get(cls_name)
    if settings_class is None:
        sys.stderr.write(f"ERROR: no class named {cls_name} found in {path}\n")
        sys.exit(2)

    prefix = _extract_env_prefix(settings_class)

    # Extract flat->nested alias map from _FLAT_TO_NESTED_ENV if present.
    # Produces reverse mapping: nested_var -> [flat_var, ...]
    # The variable may be a plain Assign or an annotated AnnAssign.
    nested_to_flat: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        dict_value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_FLAT_TO_NESTED_ENV":
                    dict_value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_FLAT_TO_NESTED_ENV"
        ):
            dict_value = node.value
        if dict_value is not None and isinstance(dict_value, ast.Dict):
            for k, v in zip(dict_value.keys, dict_value.values, strict=True):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                    flat = str(k.value)
                    nested = str(v.value)
                    nested_to_flat.setdefault(nested, []).append(flat)

    fields: dict[str, dict[str, Any]] = {}
    _expand_class_fields(
        cls_node=settings_class,
        all_classes=all_classes,
        prefix=prefix,
        nested_delimiter="__",
        nested_to_flat=nested_to_flat,
        parent_required=True,
        fields=fields,
    )

    return {"fields": fields, "prefix": prefix}


def _expand_class_fields(
    cls_node: ast.ClassDef,
    all_classes: dict[str, ast.ClassDef],
    prefix: str,
    nested_delimiter: str,
    nested_to_flat: dict[str, list[str]],
    parent_required: bool,
    fields: dict[str, dict[str, Any]],
    logical_prefix: str = "",
) -> None:
    """Recursively expand a class's annotated fields into the ``fields`` dict.

    When a field's annotation refers to a class in ``all_classes``, recurse
    with an extended prefix (``JOURNAL_DB__`` for a ``db: DbConfig`` field).
    Otherwise emit a leaf field entry.
    """
    for body_item in ast.iter_child_nodes(cls_node):
        if not isinstance(body_item, ast.AnnAssign):
            continue
        if not isinstance(body_item.target, ast.Name):
            continue

        name, has_py_default, alias = _extract_field_info(body_item)
        field_required = parent_required and not has_py_default

        # Determine annotation class name if it's a simple Name reference.
        ann = body_item.annotation
        ann_cls_name: str | None = None
        if isinstance(ann, ast.Name):
            ann_cls_name = ann.id
        elif isinstance(ann, ast.Attribute):
            ann_cls_name = ann.attr

        nested_cls = all_classes.get(ann_cls_name) if ann_cls_name else None

        if nested_cls is not None:
            # Recurse into nested model with extended prefix segment.
            sub_prefix = prefix + name.upper() + nested_delimiter
            _expand_class_fields(
                cls_node=nested_cls,
                all_classes=all_classes,
                prefix=sub_prefix,
                nested_delimiter=nested_delimiter,
                nested_to_flat=nested_to_flat,
                parent_required=field_required,
                fields=fields,
                logical_prefix=logical_prefix + name + ".",
            )
        else:
            # Leaf field -- emit env var entry.
            canonical = env_var_name(name, alias, prefix)
            flat_aliases = nested_to_flat.get(canonical, [])
            logical_name = logical_prefix + name
            fields[logical_name] = {
                "env_var": canonical,
                "aliases": flat_aliases,
                "required": field_required,
                "alias": alias,
            }


def _extract_env_prefix(node: ast.ClassDef) -> str:
    """Extract ``env_prefix`` from a Settings ``model_config``."""
    for attr in ast.iter_child_nodes(node):
        if isinstance(attr, ast.Assign):
            for target in attr.targets:
                if isinstance(target, ast.Name) and target.id == "model_config":
                    if isinstance(attr.value, ast.Call):
                        call = attr.value
                        if isinstance(call.func, ast.Name | ast.Attribute):
                            is_settings_config = (
                                isinstance(call.func, ast.Name)
                                and call.func.id == "SettingsConfigDict"
                            )
                            is_config = (
                                isinstance(call.func, ast.Attribute) and call.func.attr == "Config"
                            )
                            if is_settings_config or is_config:
                                for kw in call.keywords:
                                    if kw.arg == "env_prefix" and isinstance(
                                        kw.value, ast.Constant
                                    ):
                                        return str(kw.value.value)
                    elif isinstance(attr.value, ast.Dict):
                        for key, val in zip(attr.value.keys, attr.value.values, strict=True):
                            if (
                                isinstance(key, ast.Constant)
                                and key.value == "env_prefix"
                                and isinstance(val, ast.Constant)
                            ):
                                return str(val.value)
    return ""


def _is_field_annassign(node: ast.stmt) -> bool:
    """Return True if *node* is a class-level field annotation."""
    return isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)


def _extract_field_info(
    node: ast.AnnAssign,
) -> tuple[str, bool, str | None]:
    """Extract (field_name, has_py_default, validation_alias) from an AnnAssign."""
    if not isinstance(node.target, ast.Name):
        return ("_unknown", False, None)
    name = node.target.id
    alias: str | None = None
    value = node.value
    has_py_default = False

    if value is not None:
        if isinstance(value, ast.Call):
            if _is_field_call(value):
                for kw in value.keywords:
                    if kw.arg in ("default", "default_factory"):
                        has_py_default = True
                    elif kw.arg == "validation_alias" and isinstance(kw.value, ast.Constant):
                        alias = str(kw.value.value)
            else:
                has_py_default = True
        else:
            has_py_default = True

    return name, has_py_default, alias


def parse_compose(paths: list[str]) -> dict[str, dict[str, set[str]]]:
    """Parse compose YAML files and return declared keys + variable refs."""
    services: dict[str, dict[str, set[str]]] = {}

    for filepath in paths:
        compose_path = Path(filepath)
        if not compose_path.exists():
            sys.stderr.write(f"WARNING: compose file {filepath} not found, skipping.\n")
            continue

        data = yaml.safe_load(compose_path.read_text())
        if not isinstance(data, dict):
            continue

        services_section = data.get("services")
        if not isinstance(services_section, dict):
            continue

        for svc_name, svc_def in services_section.items():
            if not isinstance(svc_def, dict):
                continue
            if svc_name not in services:
                services[svc_name] = {
                    "declared_keys": set(),
                    "referenced_vars": set(),
                }
            svc = services[svc_name]

            env_block = svc_def.get("environment")
            if isinstance(env_block, list):
                for entry in env_block:
                    entry_str = str(entry)
                    if entry_str and "=" in entry_str:
                        key = entry_str.split("=", 1)[0].strip()
                        val = entry_str.split("=", 1)[1]
                        svc["declared_keys"].add(key)
                        _collect_ref_vars(val, svc)
                    elif entry_str:
                        svc["declared_keys"].add(entry_str.strip())
            elif isinstance(env_block, dict):
                for key, val in env_block.items():
                    svc["declared_keys"].add(str(key))
                    if isinstance(val, str) and val:
                        _collect_ref_vars(val, svc)

            _collect_ref_vars_deep(svc_def, svc)

    return services


def _collect_ref_vars(value: str, svc: dict[str, set[str]]) -> None:
    """Extract all ``$VAR`` and ``${VAR}`` variable names from *value*.

    Uses two patterns:
    - ``${VAR}`` with braces (the ``{`` after ``$`` is mandatory).
    - ``$VAR`` without braces (negative lookahead ensures we do not
      double-match a braced form).
    """
    for m in re.finditer(r"\$\{([A-Z_][A-Z0-9_]*)\}", value):
        svc["referenced_vars"].add(m.group(1))
    for m in re.finditer(r"\$([A-Z_][A-Z0-9_]*)(?!\{)", value):
        svc["referenced_vars"].add(m.group(1))


def _collect_ref_vars_deep(obj: Any, svc: dict[str, set[str]]) -> None:
    """Recursively collect ${VAR} refs from all string values under *obj*."""
    if isinstance(obj, str):
        _collect_ref_vars(obj, svc)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_ref_vars_deep(v, svc)
    elif isinstance(obj, list):
        for item in obj:
            _collect_ref_vars_deep(item, svc)


def _find_compose_file_for_service(service_name: str, compose_files: list[str]) -> str:
    """Best-effort: return the compose file that actually declares *service_name*."""
    for p in compose_files:
        path = Path(p)
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict):
            svcs = data.get("services")
            if isinstance(svcs, dict) and service_name in svcs:
                return path.name
    if not compose_files:
        return "docker-compose.yml"
    return Path(compose_files[0]).name


def check_env_contract(
    settings: dict[str, Any],
    compose: dict[str, Any],
    target_service: str,
    compose_file_names: list[str],
) -> tuple[list[str], list[str]]:
    """Check that every required Settings field has its env var in the compose.

    Returns (drift_messages, stale_passthrough_warnings).
    """
    drifts: list[str] = []
    stale: list[str] = []

    svc = compose.get(target_service, {"declared_keys": set(), "referenced_vars": set()})
    all_declared = svc["declared_keys"] | svc["referenced_vars"]

    fields = settings.get("fields", {})
    compose_file = _find_compose_file_for_service(target_service, compose_file_names)

    for fname, finfo in fields.items():
        if finfo["required"]:
            env_var = finfo["env_var"]
            aliases: list[str] = finfo.get("aliases", [])
            # A field is satisfied if its canonical nested name OR any flat
            # legacy alias appears in the compose declared/referenced vars.
            satisfied = env_var in all_declared or any(a in all_declared for a in aliases)
            if not satisfied:
                drifts.append(
                    f"DRIFT: field={fname} env={env_var} "
                    f"not declared in {compose_file} service {target_service}"
                )
                drifts.append(
                    f"Remediation: add ` - {env_var}` to the "
                    f"{target_service} service environment: list, OR "
                    f"add a default value to the Settings field."
                )

    # All canonical env vars plus all flat aliases count as known.
    known_env: set[str] = set()
    for finfo in fields.values():
        known_env.add(finfo["env_var"])
        known_env.update(finfo.get("aliases", []))
    for dk in sorted(svc["declared_keys"]):
        if dk not in known_env:
            stale.append(
                f"WARN: stale passthrough `{dk}` in {target_service}. "
                f"No matching Settings field found."
            )

    return drifts, stale


def _out(msg: str, file: Any = sys.stdout) -> None:
    file.write(msg + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check env-var contract drift between Settings and compose."
    )
    parser.add_argument(
        "--compose",
        nargs="+",
        default=["docker-compose.yml"],
        help="Compose file(s) to validate against (space-separated).",
    )
    parser.add_argument(
        "--settings",
        default="journalctl/config.py",
        help="Path to the pydantic Settings source file.",
    )
    parser.add_argument(
        "--cls",
        default="Settings",
        help="Name of the Settings class to parse.",
    )
    args = parser.parse_args()

    settings_data = parse_settings(args.settings, cls_name=args.cls)
    compose_data = parse_compose(args.compose)

    target_service = "journalctl"
    drifts, stale = check_env_contract(
        settings_data,
        compose_data,
        target_service,
        args.compose,
    )

    for msg in stale:
        _out(msg)

    for msg in drifts:
        _out(msg, file=sys.stderr)

    if drifts:
        _out("\nEnv-contract drift detected.", file=sys.stderr)
        sys.exit(1)
    _out("Env-contract check passed.")


if __name__ == "__main__":
    main()
