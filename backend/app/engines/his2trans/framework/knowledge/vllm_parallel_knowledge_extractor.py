#!/usr/bin/env python3
"""
Parallel vLLM-based "translation knowledge" extractor.

Goal
----
Given (C code, Rust code) pairs, call an OpenAI-compatible vLLM endpoint in parallel
to extract:
- Structural fragments ("Partial"/"Full")
- API mappings ("API_Mapping")

The JSON schema and calling pattern intentionally mirrors:
  <his2trans>/rag_builder/qwen3_coder/8_run_parallel_vllm_client.py

So that the output can be merged into the existing RAG knowledge_base.json format and
used without changing downstream consumers.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from openai import OpenAI


PROMPT_TEMPLATE = """
You are an expert Data Curator for C-to-Rust Code Translation. Your goal is to extract training data and API mappings.

================================================================================
SECTION 1: OBJECTIVES
================================================================================

**Two Goals:**
1. **Structure Learning:** Find structurally similar code (Full/Partial)
2. **Dictionary Mining:** Extract API mappings between C and Rust

**Granularity:** Don't limit to whole functions. Extract specific patterns like:
- C `for` loop → Rust iterator chain
- C `malloc` pattern → Rust `Box::new`

**Value Test:** "Would this help a developer translate this specific C snippet to Rust?"

================================================================================
SECTION 2: PRE-JUDGMENT FILTERS (Execute in order, stop at first None)
================================================================================

**Filter 1: Entity Name Check (Affects Full/Partial only)**
- Names must refer to the SAME CONCEPT for Full/Partial classification
- ✅ Allowed: `GetPid`↔`get_pid`, `Monster`↔`MonsterData`, `StatBuilder`↔`new`, `CreateStat`↔`pack`
- ❌ Blocked for Full/Partial: `LLVMRustXxx` vs `f`, `MonsterBuilder` vs `PlayerInputBuilder`
- **Note:** Even if names don't match, still check for API mappings!
- **Negative Example (Hallucination Check):**
  - C: `prev_stack` (Memory/Stack logic) vs Rust: `record_exports` (Data export)
  - Even if structure looks similar (both have loops/ifs) -> **None** (Domain Mismatch).

**Filter 2: Empty/Trivial Code → None**
- **C empty:** `void foo() {{}}`, `T() = default;`, `~T() = default;`, function body is only `{{}}`
- **Rust empty:** `fn foo() {{}}`, `fn foo() {{ }}`, `fn foo(&self) {{}}`, single-line with no logic
- **Examples of empty/trivial (ALL → None):**
  - `void foo() {{}}` vs `fn foo() {{}}` → None
  - `T() = default;` vs `fn default() -> Self {{...}}` → None
  - `int get() {{ return x; }}` vs `fn get(&self) {{}}` → None (Rust side empty!)
  - Any function with body `{{}}` or `{{ }}` → None
- One side empty, other has logic → None
- **No exceptions for empty code**

**Filter 3: FFI Wrapper → None**
- IS FFI: `libc::read(...)`, `rustrt::rust_uv_xxx(...)`, body is ONLY one FFI call
- NOT FFI: Rust code with actual logic, even if calling similar-named functions

**Filter 4: Semantic Domain Mismatch (CRITICAL) → None**
- **Domain Mismatch:** Memory Mgmt (linked lists, pointers) vs Logging/UI/Map Logic.
- **Example:** C `node = node->next` (pointer) vs Rust `map.insert(k, v)` (hashmap).
- **Generic Structure:** "Both use if/else" is NOT enough. The *content* must match.
- If domains are totally different, stop here -> **None**.

**Filter 5: Empty Structs → None**
- Both structs have no fields → None
- One has fields, other doesn't → None

**Filter 6: Definition vs Usage Asymmetry (CRITICAL) → None**
- **Definition <-> Usage Mismatch:** Do NOT match a function DEFINITION with a function CALL.
  - C: `void foo() { ...body... }` (Definition)
  - Rust: `foo();` (Call/Usage)
  - **Result:** **None**. (We want to translate the code, not learn how to call it)
- **Implementation <-> Test Wrapper Mismatch:**
  - C: `NewVector() { ... }` (Implementation/Factory)
  - Rust: `test_create() { NewRowEntity(); }` (Test Wrapper calling the function)
  - **Result:** **None**.
- **Exception:** Test <-> Test (Logic mapping) is OK. Impl <-> Impl is OK.

================================================================================
SECTION 3: CLASSIFICATION (Four independent dimensions, can coexist)
================================================================================

**IMPORTANT: Extract ALL knowledge from a single code pair!**
- One pair can have: is_full=false, is_partial=true, has_api_mappings=true (multiple APIs)
- `structural_fragments`: Can contain MULTIPLE matching fragments
- `api_mappings`: Can contain MULTIPLE API correspondences (extract ALL you find!)
- Even if not Full, you should still extract Partial fragments and API mappings

**3.1 Full (`is_full: true`)**
- High structural similarity as a whole
- Names refer to same concept + has real logic
- Allowed differences: naming style, type syntax (`int`→`i32`), Rust idioms (`Option`, `Result`, `?`)

**⚠️ Full BLOCKERS (if any of these, CANNOT be Full, but CAN still extract Partial/API/Algo):**

1. **Code Length Mismatch:** If one side is >3x longer than the other → NOT Full
   - `FinishMonsterBuffer` (4 lines) vs `create_serialized_example` (50 lines) → NOT Full
   - Reason: Vastly different scope, even if one contains the other
   - **Action:** Mark `is_full=false`, but check for `is_partial` or `has_api_mappings`

2. **Implementation vs Test Mismatch:** One is implementation, other is test → NOT Full
   - C: `BuildNotifyData(...)` vs Rust: `ut_notify_data_creation()` → NOT Full
   - C: `RecordRequestTask(...)` vs Rust: `test_insert_task()` → NOT Full
   - **Detect by:** Name starts with `test_`, `ut_`, `Test`, `TEST`, or file path contains `/test/`
   - **Action:** Mark `is_full=false`.
   - **CRITICAL:** If it's just the test *calling* the implementation (Def vs Usage), mark **None** (caught by Filter 6). Only extract if there is symmetric logic (e.g. Test vs Test).

3. **Semantic & Domain Consistency (CRITICAL)**:
   - **Data Structure Consistency**: Do NOT match fundamentally different data structures unless they are standard equivalents.
     - OK: `std::vector` <-> `Vec`, `std::string` <-> `String`.
     - FORBIDDEN: `LinkedList Pointer` (`node->prev`) <-> `HashMap` (`map.insert`).
   - **Operation Domain**: Do NOT match low-level memory management with high-level business logic.
     - FORBIDDEN: `free(ptr)` <-> `log_info(...)`.
     - FORBIDDEN: Pointer arithmetic <-> UI updates / Config parsing.
   - **No Generic Structural Matching**: Do NOT classify as Partial/API just because both sides use `if/else`, `match`, or `for-loops`.
     - The **Condition** and **Action** must be semantically equivalent.
     - *Bad Example*: C checks stack size (`if (stk->is_big)`), Rust checks Option (`if let Some(...)`). -> **None**.

4. **Empty/Trivial on Either Side:** → NOT Full
   - `fn net_capability_changed(&self, ...) {{}}` → Rust is empty → NOT Full

**3.2 Partial (`is_partial: true`)**
- Some blocks match structurally, not the whole code
- Extract matching portions in `structural_fragments`

**3.3 API Mappings (`has_api_mappings: true`)**

┌─────────────────────────────────────────────────────────────────────────────┐
│ KEY IDEA: Most valuable knowledge = API translation mappings                │
│ Focus on WHAT the code DOES, not what it's CALLED.                          │
│ If C does X and Rust does X with different API names → EXTRACT the mapping  │
└─────────────────────────────────────────────────────────────────────────────┘

**How to identify mappings (even with different names):**
1. Same logical position in control flow
2. Same operation type (read/write/create/delete/query)
3. Same data flow (similar input source, similar output destination)
4. Same error handling pattern
5. Surrounding context does the same thing

**Examples of DIFFERENT names, SAME function (EXTRACT THESE!):**
| C Code                              | Rust Code                                | Function              |
|-------------------------------------|------------------------------------------|-----------------------|
| `fbb_.AddElement<float>(...)`       | `self.fbb_.push_slot::<f32>(...)`        | Add field to builder  |
| `stack_.push_back(Value(b))`        | `self.values.push(Value::Bool(x))`       | Push to container     |
| `fbb.Finish(root)`                  | `fbb.finish(root, None)`                 | Finalize buffer       |
| `GetField<uint8_t>(VT_U8, 0)`       | `self._tab.get::<u8>(VT_U8_, Some(0))`   | Read field            |
| `sysm->ListSystemAbilities()`       | `ListSystemAbilitiesWithDumpFlag(...)`   | List abilities        |
| `REQUEST_HILOGI(...)`               | `info!(...)`                             | Logging               |
| `permissions.push_back(...)`        | `permissions.push(...)`                  | Append to collection  |
| `std::make_unique<T>()`             | `Box::new(T)`                            | Heap allocation       |

**Inference Process:**
1. Read C line: "What is this DOING?" (e.g., "adding a string to a list")
2. Read Rust line: "What is this DOING?" (e.g., "pushing a string to a vector")
3. Same action? → Extract as API mapping

**When to Extract:**
- ✅ Different names, same function → EXTRACT
- ✅ Similar names, same function → EXTRACT
- ✅ Unknown API but context shows same purpose → EXTRACT
- ❌ Similar names, different function → DO NOT extract
- ❌ Definition vs Usage mismatch (e.g., C impl vs Rust test call) → DO NOT extract (Mark as None)

**IMPORTANT:** Many APIs are from proprietary libraries (OHOS, HiSysEvent, IPC).
Lack of prior knowledge is NOT a reason to skip. Use context-based inference!

**3.4 None (all flags false)**
- Empty/trivial code
- Completely unrelated with no shared patterns

================================================================================
SECTION 4: OUTPUT FORMAT
================================================================================

Return ONLY valid JSON. No markdown.

The four flags are INDEPENDENT. Multiple can be true. All false = None.

{{
  "ffi_call_detected": "(boolean)",
  "reasoning": "(string) [FFI Check] -> [Task Analysis] -> [Similarity] -> [Knowledge Extraction]",
  
  // ===== FOUR CLASSIFICATION FLAGS (independent, can all be true) =====
  "is_full": "(boolean) High structural similarity as a whole?",
  "is_partial": "(boolean) Some blocks match structurally?",
  "has_api_mappings": "(boolean) Any API/pattern correspondences?",

  "keywords": ["(string) Technical concepts involved."],

  // ===== STRUCTURAL KNOWLEDGE =====
  // Extract MULTIPLE fragments if there are multiple matching blocks
  "structural_fragments": [
    {{
      "c_fragment": "(string) EXACT substring from C code.",
      "rust_fragment": "(string) EXACT substring from Rust code.",
      "match_type": "(string) 'full' | 'partial'",
      "description": "(string) What this block does."
    }}
    // Can have MULTIPLE entries - extract ALL matching fragments
  ],

  // ===== API KNOWLEDGE =====
  // Extract MULTIPLE mappings - one entry per API correspondence
  "api_mappings": [
    {{
      "c_api": "(string) C API/pattern (exact substring).",
      "rust_api": "(string) Rust API/pattern (exact substring).",
      "mapping_type": "(string) 'function' | 'method' | 'type' | 'pattern' | 'field_access'",
      "description": "(string) E.g., 'Thread creation', 'Memory allocation'"
  }}
    // Can have MULTIPLE entries - extract ALL API correspondences found
  ],

}}

**--- INPUT CODE ---**

[C Code]
{c_code}

[Rust Code]
{rust_code}
"""


@dataclass(frozen=True)
class VllmSampling:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    max_tokens: int = 4096


_WORKER_CLIENT: Optional[OpenAI] = None
_WORKER_MODEL: str = ""
_WORKER_TIMEOUT: float = 180.0
_WORKER_SAMPLING: VllmSampling = VllmSampling()


def _init_worker(base_url: str, api_key: str, model: str, timeout: float, sampling: VllmSampling) -> None:
    global _WORKER_CLIENT, _WORKER_MODEL, _WORKER_TIMEOUT, _WORKER_SAMPLING
    _WORKER_MODEL = model
    _WORKER_TIMEOUT = timeout
    _WORKER_SAMPLING = sampling
    _WORKER_CLIENT = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    # Connectivity check (fail-fast)
    _WORKER_CLIENT.models.list()


def _clean_json_text(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```json"):
        s = s[len("```json") :]
    if s.startswith("```"):
        s = s[len("```") :]
    if s.endswith("```"):
        s = s[: -len("```")]
    return s.strip()


def _process_task(task: Dict[str, Any]) -> Dict[str, Any]:
    global _WORKER_CLIENT
    if _WORKER_CLIENT is None:
        return {**task, "llm_error": "ERROR_WORKER_NOT_INITIALIZED"}

    task_id = task.get("task_id") or ""
    c_code = task.get("c_code") or ""
    rust_code = task.get("rust_code") or ""

    # Avoid `.format()`; code contains braces.
    prompt = PROMPT_TEMPLATE.replace("{c_code}", c_code).replace("{rust_code}", rust_code)
    messages = [{"role": "user", "content": prompt}]

    try:
        completion = _WORKER_CLIENT.chat.completions.create(
            model=_WORKER_MODEL,
            messages=messages,
            stop=["<|im_end|>"],
            temperature=_WORKER_SAMPLING.temperature,
            top_p=_WORKER_SAMPLING.top_p,
            max_tokens=_WORKER_SAMPLING.max_tokens,
            extra_body={
                "top_k": _WORKER_SAMPLING.top_k,
                "repetition_penalty": _WORKER_SAMPLING.repetition_penalty,
            },
        )
        content = completion.choices[0].message.content
        cleaned = _clean_json_text(content)
        llm_json = json.loads(cleaned)
        return {
            **{k: v for k, v in task.items() if k not in ("c_code", "rust_code")},
            **llm_json,
            "llm_ok": True,
            "llm_output_raw": content,
        }
    except Exception as e:
        return {
            **{k: v for k, v in task.items() if k not in ("c_code", "rust_code")},
            "llm_ok": False,
            "llm_error": f"{type(e).__name__}: {e}",
            "llm_traceback": traceback.format_exc(limit=6),
            "task_id": task_id,
        }


def load_completed_task_ids(output_jsonl: Path) -> Set[str]:
    if not output_jsonl.exists():
        return set()
    out: Set[str] = set()
    with output_jsonl.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            tid = (obj.get("task_id") or "").strip()
            if tid:
                out.add(tid)
    return out


def load_results_by_task_id(output_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    if not output_jsonl.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with output_jsonl.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            tid = (obj.get("task_id") or "").strip()
            if tid:
                out[tid] = obj
    return out


def run_parallel_vllm_knowledge_extraction(
    tasks: List[Dict[str, Any]],
    *,
    output_jsonl: Path,
    base_url: str,
    api_key: str,
    model: str,
    num_workers: int,
    timeout: float = 180.0,
    sampling: Optional[VllmSampling] = None,
    resume: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Runs vLLM extraction and writes JSONL incrementally (append-only).

    Returns a mapping task_id -> result_record.
    """
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    results = load_results_by_task_id(output_jsonl) if resume else {}
    completed = set(results.keys())

    todo = [t for t in tasks if (t.get("task_id") or "") and (t["task_id"] not in completed)]
    if not todo:
        return results

    # Align with vLLM usage elsewhere in the repo (spawn is safer for client state).
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    sampling = sampling or VllmSampling()
    num_workers = max(1, int(num_workers or 1))

    start = time.time()
    with output_jsonl.open("a", encoding="utf-8") as f_out:
        with multiprocessing.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(base_url, api_key, model, timeout, sampling),
        ) as pool:
            for rec in pool.imap_unordered(_process_task, todo):
                try:
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_out.flush()
                except Exception:
                    # If writing fails, keep going; caller can inspect stdout/stderr.
                    pass
                tid = (rec.get("task_id") or "").strip()
                if tid:
                    results[tid] = rec

    dur = time.time() - start
    if dur > 0 and todo:
        print(f"[KB-LLM] processed={len(todo)} workers={num_workers} rate={len(todo)/dur:.2f} tasks/s", file=sys.stderr)
    return results


def convert_llm_record_to_extracted_knowledge(rec: Dict[str, Any], *, source: str = "vllm") -> List[Dict[str, Any]]:
    """
    Convert vLLM JSON schema (api_mappings/structural_fragments) to the KB schema:
      knowledge_type in {"API_Mapping","Partial","Full"}.
    """
    out: List[Dict[str, Any]] = []

    for m in rec.get("api_mappings") or []:
        if not isinstance(m, dict):
            continue
        c_api = (m.get("c_api") or "").strip()
        rust_api = (m.get("rust_api") or "").strip()
        desc = (m.get("description") or "").strip()
        if not c_api or not rust_api:
            continue
        out.append(
            {
                "knowledge_type": "API_Mapping",
                "c_api": c_api,
                "rust_api": rust_api,
                "description": desc,
                "mapping_type": (m.get("mapping_type") or "").strip(),
                "source": source,
            }
        )

    for frag in rec.get("structural_fragments") or []:
        if not isinstance(frag, dict):
            continue
        c_frag = (frag.get("c_fragment") or "").strip()
        r_frag = (frag.get("rust_fragment") or "").strip()
        mtype = (frag.get("match_type") or "").strip().lower()
        desc = (frag.get("description") or "").strip()
        if not c_frag or not r_frag:
            continue
        ktype = "Full" if mtype == "full" else "Partial"
        out.append(
            {
                "knowledge_type": ktype,
                "c_fragment": c_frag,
                "rust_fragment": r_frag,
                "description": desc,
                "match_type": mtype,
                "source": source,
            }
        )

    return out
