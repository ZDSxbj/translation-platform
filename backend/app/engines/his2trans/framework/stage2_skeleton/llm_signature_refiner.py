#!/usr/bin/env python3
"""
LLM signature refiner (judge/correct a rule-based Rust FFI signature).

Motivation:
- Stage1 already has deterministic rule-based signature generation (TypeMapper).
- Stage2 BM25 matching is now avoidable via deterministic mapping.
- However, rule-based type mapping can still be wrong in tricky cases (typedef chains, callbacks, ABI-width types).
- This module lets an LLM *review* a candidate signature with minimal, sliced evidence from bindgen types.rs.

Design constraints:
- Do NOT dump full `types.rs` into the prompt: slice it using `context_slicer.TypesRsRegistry`.
- Print prompt length (chars and prompt_tokens if available).
- If the LLM output is invalid/unparseable, fall back to the candidate signature.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Tuple

from context_slicer import TypesRsRegistry, extract_identifiers, extract_types_from_rust_signature
from generate.generation import generation


_BOOL_TRUE = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in _BOOL_TRUE


def _collapse_ws(s: str) -> str:
    return " ".join((s or "").replace("\r", " ").replace("\n", " ").split()).strip()


def _extract_fn_rest(signature_text: str, func_name: str) -> Optional[str]:
    """
    Extract the `fn <name>(...) -> ...` tail (starting at `fn`) from a text blob.
    Returns the "rest" starting from `fn`, with whitespace collapsed.
    """
    if not signature_text or not func_name:
        return None

    # Prefer content after "最终答案"
    tail = signature_text
    for marker in ("最终答案", "FINAL", "Final", "final", "Answer", "答案"):
        pos = tail.rfind(marker)
        if pos != -1:
            tail = tail[pos:]
            break

    # Remove code fences
    tail = re.sub(r"```[a-zA-Z]*", "", tail)
    tail = tail.replace("```", "")

    # Find the signature start (include any prefix; we'll normalize later).
    m = re.search(rf"(?:pub\s+)?(?:unsafe\s+)?(?:extern\s+\"C\"\s+)?fn\s+{re.escape(func_name)}\b", tail)
    if not m:
        # Fallback: find `fn <name>`
        m = re.search(rf"\bfn\s+{re.escape(func_name)}\b", tail)
        if not m:
            return None

    start = m.start()
    # End at first '{' if present; otherwise, end at first blank line or end.
    brace = tail.find("{", start)
    if brace != -1:
        raw = tail[start:brace]
    else:
        raw = tail[start:]
        # Trim after the first double newline (common separation)
        nn = raw.find("\n\n")
        if nn != -1:
            raw = raw[:nn]

    raw = raw.strip()
    raw = _collapse_ws(raw)

    # Ensure we return from the "fn" keyword
    m2 = re.search(rf"\bfn\s+{re.escape(func_name)}\b", raw)
    if not m2:
        return None
    return raw[m2.start() :].strip()


@dataclass
class SignatureRefineResult:
    ok: bool
    refined_signature: Optional[str]
    prompt_chars: int
    prompt_tokens: int
    response_text: str
    error: Optional[str] = None


class LLMSignatureRefiner:
    def __init__(
        self,
        *,
        types_rs_path: Optional[Path],
        # 0 means "no cap" (still a slice, but not truncated by chars)
        max_types_slice_chars: int = 0,
        enabled: bool = True,
        print_prompt_len: bool = True,
    ):
        self.enabled = enabled
        self.max_types_slice_chars = max(0, int(max_types_slice_chars))
        self.print_prompt_len = print_prompt_len

        self._registry: Optional[TypesRsRegistry] = None
        if types_rs_path and types_rs_path.exists():
            try:
                self._registry = TypesRsRegistry.from_types_rs(types_rs_path)
            except Exception:
                self._registry = None

    def _build_types_slice(self, *, c_signature: str, candidate_rust_signature: str) -> str:
        if not self._registry:
            return ""

        seeds: Set[str] = set()
        # Rust-side seeds (crate::types::X)
        seeds.update(extract_types_from_rust_signature(candidate_rust_signature or ""))
        # Also seed by C identifiers that happen to exist as bindgen symbol names (typedef/struct aliases)
        seeds.update(self._registry.filter_existing(extract_identifiers(c_signature or "")))

        if not seeds:
            return ""

        entries = self._registry.slice_symbols(seeds)
        if not entries:
            return ""

        out_parts = ["// === types.rs slice (bindgen) ===\n"]
        used = len(out_parts[0])
        for e in entries:
            block = (e.code.rstrip() + "\n\n")
            if self.max_types_slice_chars and used + len(block) > self.max_types_slice_chars:
                break
            out_parts.append(block)
            used += len(block)

        return "".join(out_parts).rstrip() + "\n"

    def refine(
        self,
        *,
        func_name: str,
        c_signature: str,
        candidate_rust_signature: str,
        is_static: bool,
    ) -> SignatureRefineResult:
        if not self.enabled:
            return SignatureRefineResult(
                ok=False,
                refined_signature=None,
                prompt_chars=0,
                prompt_tokens=0,
                response_text="",
                error="disabled",
            )

        types_slice = self._build_types_slice(
            c_signature=c_signature,
            candidate_rust_signature=candidate_rust_signature,
        )
        types_slice_chars = len(types_slice or "")

        # NOTE:
        # - `_extract_fn_rest()` always returns a string starting with `fn <name>(...)`.
        # - For non-static functions, we want `pub extern "C" fn ...` (prefix + fn_rest).
        # - For static functions, we want just `fn ...` (do NOT prefix again), otherwise we'd get `fn fn ...`.
        expected_prefix_for_prompt = "fn" if is_static else 'pub extern "C" fn'
        expected_prefix_for_output = "" if is_static else 'pub extern "C"'

        system_prompt = (
            "You are a C→Rust FFI signature expert. "
            "Your job is to review a candidate Rust signature and output a corrected Rust signature if needed."
        )

        user_prompt = f"""你将看到一个 C 函数签名，以及一个“基于规则生成的 Rust FFI 签名候选”。

你的任务：判断候选签名是否可能有类型/ABI 错误，并给出最终的 Rust 签名。

## 约束（必须遵守）
1) 函数名必须是 `{func_name}`
2) 参数个数必须与候选签名一致（不要增删参数）
3) 不要引入 Rust 专有类型（String/Vec/...），只能用 FFI 安全类型和 `crate::types::*`
4) 输出必须包含两部分：
   - “简要理由：”后面用 3-6 条要点说明关键依据（不要输出冗长推理过程）
   - “最终答案：”后面只给出 *一行* Rust 签名（不要函数体）
   5) `{expected_prefix_for_prompt}` 前缀规则：
  	   - 如果是 static 函数：用 `fn`
  	   - 如果是非 static：用 `pub extern \"C\" fn`

## C 签名
```c
{c_signature.strip()}
```

## 候选 Rust 签名（规则引擎输出）
```rust
{candidate_rust_signature.strip()}
```
"""

        if types_slice.strip():
            user_prompt += f"""
## 相关类型上下文（从 bindgen 生成的 types.rs 切片，已做筛选）
```rust
{types_slice.rstrip()}
```
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_chars = sum(len(m.get("content") or "") for m in messages)

        try:
            resp = generation(messages, return_usage=True)
            if isinstance(resp, dict):
                content = resp.get("content") or ""
                usage = resp.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
            else:
                content = str(resp)
                prompt_tokens = 0

            if self.print_prompt_len:
                # Char length is deterministic; prompt_tokens depends on backend tokenization.
                print(
                    f"  ℹ 签名复核: {func_name} prompt_chars={prompt_chars}, types_slice_chars={types_slice_chars}"
                    + (f", prompt_tokens={prompt_tokens}" if prompt_tokens else "")
                )

            fn_rest = _extract_fn_rest(content, func_name)
            if not fn_rest:
                return SignatureRefineResult(
                    ok=False,
                    refined_signature=None,
                    prompt_chars=prompt_chars,
                    prompt_tokens=prompt_tokens,
                    response_text=content,
                    error="cannot_parse_signature",
                )

            refined = f"{expected_prefix_for_output} {fn_rest}" if expected_prefix_for_output else fn_rest
            refined = _collapse_ws(refined)

            # Sanity: must contain "fn <name>("
            if not re.search(rf"\bfn\s+{re.escape(func_name)}\s*\(", refined):
                return SignatureRefineResult(
                    ok=False,
                    refined_signature=None,
                    prompt_chars=prompt_chars,
                    prompt_tokens=prompt_tokens,
                    response_text=content,
                    error="invalid_refined_signature",
                )

            return SignatureRefineResult(
                ok=True,
                refined_signature=refined,
                prompt_chars=prompt_chars,
                prompt_tokens=prompt_tokens,
                response_text=content,
                error=None,
            )
        except Exception as e:
            return SignatureRefineResult(
                ok=False,
                refined_signature=None,
                prompt_chars=prompt_chars,
                prompt_tokens=0,
                response_text="",
                error=str(e),
            )


def build_refiner_from_env(*, types_rs_path: Optional[Path]) -> Optional[LLMSignatureRefiner]:
    """
    Convenience factory (used by skeleton_builder).

    Env toggles:
    - C2R_LLM_REFINE_SIGNATURES=0 to disable (enabled by default)
    - C2R_LLM_REFINE_SIGNATURES_MAX_TYPES_CHARS (default 0 = no cap)
    - C2R_LLM_REFINE_SIGNATURES_PRINT_PROMPT_LEN (default true)
    """
    if not _env_bool("C2R_LLM_REFINE_SIGNATURES", True):
        return None

    try:
        max_types = int(os.environ.get("C2R_LLM_REFINE_SIGNATURES_MAX_TYPES_CHARS", "0"))
    except Exception:
        max_types = 0
    print_len = _env_bool("C2R_LLM_REFINE_SIGNATURES_PRINT_PROMPT_LEN", True)

    return LLMSignatureRefiner(
        types_rs_path=types_rs_path,
        max_types_slice_chars=max_types,
        enabled=True,
        print_prompt_len=print_len,
    )
