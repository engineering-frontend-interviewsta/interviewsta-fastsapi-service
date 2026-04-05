#!/usr/bin/env python3
"""Generate docs/prisma-database-schema.md from prisma/schema.prisma (run after `prisma db pull`)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "prisma" / "schema.prisma"
OUT = ROOT / "docs" / "prisma-database-schema.md"


def parse_enums(text: str) -> list[tuple[str, list[str]]]:
    enums: list[tuple[str, list[str]]] = []
    for m in re.finditer(r"^enum\s+(\w+)\s*\{([^}]+)\}", text, re.MULTILINE | re.DOTALL):
        name, body = m.group(1), m.group(2)
        vals = []
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            vals.append(line)
        enums.append((name, vals))
    return enums


def parse_models(text: str) -> list[dict]:
    models = []
    for m in re.finditer(r"^///[^\n]*\n(?:^///[^\n]*\n)*", text, re.MULTILINE):
        pass  # doc comments attached to following model — strip for simpler field parse
    # Remove generator and datasource
    body = text
    for block in re.finditer(r"^model\s+(\w+)\s*\{", body, re.MULTILINE):
        start = block.start()
        name = block.group(1)
        depth = 0
        i = block.end() - 1
        while i < len(body):
            if body[i] == "{":
                depth += 1
            elif body[i] == "}":
                depth -= 1
                if depth == 0:
                    block_text = body[block.end() : i]
                    models.append({"name": name, "body": block_text})
                    break
            i += 1
    return models


def field_lines(model_body: str) -> tuple[list[tuple[str, str]], list[str]]:
    fields: list[tuple[str, str]] = []
    meta: list[str] = []
    for raw in model_body.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("@@"):
            meta.append(line)
        else:
            parts = line.split(None, 1)
            if len(parts) == 2:
                fields.append((parts[0], parts[1]))
            else:
                fields.append((parts[0], ""))
    return fields, meta


def main() -> None:
    text = SCHEMA.read_text(encoding="utf-8")
    enums = parse_enums(text)
    models = parse_models(text)

    lines: list[str] = [
        "# Database schema (from Prisma introspection)",
        "",
        "This document lists **PostgreSQL `public` schema** tables and columns as reflected in "
        "[`prisma/schema.prisma`](../prisma/schema.prisma). It is generated from the live database via "
        "`npx prisma db pull`.",
        "",
        "**Regenerate:**",
        "",
        "```bash",
        "npx prisma@5 db pull --schema=prisma/schema.prisma",
        "python3 scripts/generate_prisma_schema_docs.py",
        "```",
        "",
        f"**Summary:** {len(models)} models (tables), {len(enums)} enums.",
        "",
        "---",
        "",
        "## Table of contents",
        "",
    ]
    for mo in models:
        anchor = mo["name"].replace("_", "-")
        lines.append(f"- [{mo['name']}](#{anchor})")
    lines.append("- [Enums](#enums)")
    lines.append("")
    lines.append("---")
    lines.append("")

    for mo in models:
        name = mo["name"]
        fl, meta = field_lines(mo["body"])
        lines.append(f"## `{name}`")
        lines.append("")
        lines.append("| Field | Type and attributes |")
        lines.append("| --- | --- |")
        for fname, fdef in fl:
            safe_def = fdef.replace("|", "\\|")
            lines.append(f"| `{fname}` | `{safe_def}` |")
        if meta:
            lines.append("")
            lines.append("**Constraints / indexes:**")
            for x in meta:
                lines.append(f"- `{x}`")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Enums")
    lines.append("")
    for ename, vals in enums:
        lines.append(f"### `{ename}`")
        lines.append("")
        for v in vals:
            lines.append(f"- `{v}`")
        lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
