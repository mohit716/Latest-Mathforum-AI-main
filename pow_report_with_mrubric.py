#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PoW Analytics (sqlite3 + mrubric via Ollama) -> CSV reports

Extends the prior sqlite3-only report to add AI-based "mentoring rubric" (mrubric) evaluation
for each mentor reply, using a local Ollama server over HTTP (e.g., http://localhost:11434).

Outputs:
1) pow_report.csv                                  (original A–F analytics)
2) pow_mrubric_scores.csv                          (one row per mentor reply with mrubric scores)
3) pow_mrubric_by_replies.csv                      (distribution by # mentor replies per thread)
4) pow_mrubric_by_revisions.csv                    (distribution by # student revisions per thread; threads with mentoring only)
5) pow_mrubric_by_improvement.csv                  (distribution by Δ prubric total; threads with mentoring only)
6) pow_mrubric_by_replies_cross.csv                (as 3), crossed by service, mentor, teacher, reply-time bin)
7) pow_mrubric_by_revisions_cross.csv              (as 4), crossed by service, mentor, teacher, reply-time bin)
8) pow_mrubric_by_improvement_cross.csv            (as 5), crossed by service, mentor, teacher, reply-time bin)

Zip up documents identified by each category of pow_report.csv.  If text_problem_outputs already exists, consider that a cache for all problem threads, and zip those txt/json files.  Otherwise, extract the text from the database and save them into thread_outputs to include them in the zip as well.

mrubric categories (1–3 scale; 1 = weak, 3 = strong):
- build_on_student_thinking
- specificity_actionability
- mathematical_soundness
- questioning_scaffolding
- clarity_conciseness
- affect_support
- concise
- curious
- focused_questions
- accessible
- concrete
- collaborative
- specific
- encouraging
- nice_tone
- improve_work
- empowerment_confidence
- developing_thinking
- identity_competence
- efficiency
- encouraging_engagement
- process_oriented
- modeling_progression
- promote_curiosity
- individualized
- clear_actionable
- guiding_questions

The Ollama prompt requests STRICT JSON with these fields and an overall_comments string.
Caching: evaluations are optionally cached under mrubric_cache/{response_id}.json to avoid re-scoring.

Usage:
    python pow_sqlite_report_with_mrubric.py --db mathforum.db --out pow_report.csv \
        --ollama-urls http://localhost:11434<,...> --ollama-model llama3 \
        --ignore-deleted --timeout 30 --max-concurrency 4

Replace --ollama-model llama3 with desired model, i.e. --ollama-model llama3:8b-instruct-q4_K_M

If --enable-mrubric is omitted, the script skips AI calls and only writes the original report.

If --ignore_deleted is specified, the script removes deleted records from processing

Start with the following command for each ollama port desired (i.e. 11434, ...)
OLLAMA_HOST=localhost:$PORT ollama serve & 
OLLAMA_HOST=localhost:$PORT ollama run $MODEL
... and for the desired model used with --ollama-model above i.e. mistral:7b-instruct-q4_K_M or llama3:8b-instruct-q4_K_M

Recommended indices:

CREATE INDEX IF NOT EXISTS idx_psg_group_id ON pow_pub_submission_groups(group_id);
CREATE INDEX IF NOT EXISTS idx_psg_publication_id ON pow_pub_submission_groups(publication_id);
CREATE INDEX IF NOT EXISTS idx_dm_group_role_deleted ON dir_memberships(group_id, role, deleted);
CREATE INDEX IF NOT EXISTS idx_dm_group_id ON dir_memberships(group_id);
CREATE INDEX IF NOT EXISTS idx_dm_role_deleted ON dir_memberships(role, deleted);
CREATE INDEX IF NOT EXISTS idx_du_id ON dir_users(id);
"""

import argparse
import csv
import json
import math
import os
import queue
import re
import sqlite3
import statistics
import sys
import threading
import time
import requests
import traceback
import pickle
import tempfile
import shutil
import asyncio
import aiohttp
import hashlib
from contextlib import closing
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
import zipfile
from pathlib import Path
import html
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- in-process caches (do NOT mutate returned objects) ----
MAP_CACHE_FILE = "map_cache.pkl"
FETCH_CACHE_FILE = "fetch_cache.pkl"

_MAP_CACHE: Dict[str, Any] = {}
_FETCH_CACHE: Dict[str, Any] = {}

def load_caches(disable_cache=False):
    if not disable_cache:        
        global _MAP_CACHE, _FETCH_CACHE

        if os.path.exists(MAP_CACHE_FILE):
            try:
                with open(MAP_CACHE_FILE, "rb") as f:
                    _MAP_CACHE = pickle.load(f)
                tqdm.write(f"[load_caches] Loaded _MAP_CACHE with {len(_MAP_CACHE)} entries")
            except Exception as e:
                tqdm.write(f"[load_caches] Failed to load _MAP_CACHE: {e}")

        if os.path.exists(FETCH_CACHE_FILE):
            try:
                with open(FETCH_CACHE_FILE, "rb") as f:
                    _FETCH_CACHE = pickle.load(f)
                tqdm.write(f"[load_caches] Loaded _FETCH_CACHE with {len(_FETCH_CACHE)} entries")
            except Exception as e:
                tqdm.write(f"[load_caches] Failed to load _FETCH_CACHE: {e}")

def save_caches(disable_cache=False, save_map_cache=True, save_fetch_cache=True):
    if not disable_cache:
        def atomic_pickle_write(obj, path):
            dir_name = os.path.dirname(path) or "."
            with tempfile.NamedTemporaryFile("wb", dir=dir_name, delete=False) as tmp_file:
                pickle.dump(obj, tmp_file)
                temp_name = tmp_file.name
            shutil.move(temp_name, path)  # atomic on most OSes    
            
        if save_map_cache:
            try:
                atomic_pickle_write(_MAP_CACHE, MAP_CACHE_FILE)
                tqdm.write(f"[save_caches] Saved _MAP_CACHE with {len(_MAP_CACHE)} entries")
            except Exception as e:
                tqdm.write(f"[save_caches] Failed to save _MAP_CACHE: {e}")

        if save_fetch_cache:
            try:
                atomic_pickle_write(_FETCH_CACHE, FETCH_CACHE_FILE)
                tqdm.write(f"[save_caches] Saved _FETCH_CACHE with {len(_FETCH_CACHE)} entries")
            except Exception as e:
                tqdm.write(f"[save_caches] Failed to save _FETCH_CACHE: {e}")          

def _get_map_cache(key: str):
    result = _MAP_CACHE.get(key)
    if result is not None:
        tqdm.write(f"[INFO] {key} found in map cache...")
    return result

def _set_map_cache(disable_cache: bool, key: str, value: Any):
    tqdm.write(f"[INFO] Adding {key} to map cache...")
    _MAP_CACHE[key] = value
    save_caches(disable_cache, save_map_cache=True, save_fetch_cache=False)
    return value

def _get_fetch_cache(key: str):
    result = _FETCH_CACHE.get(key)
    if result is not None:
        tqdm.write(f"[INFO] {key} found in fetch cache...")
    return result

def _set_fetch_cache(disable_cache: bool, key: str, value: Any):
    tqdm.write(f"[INFO] Adding {key} to fetch cache...")
    _FETCH_CACHE[key] = value
    save_caches(disable_cache, save_map_cache=False, save_fetch_cache=True)
    return value

def clear_caches():
    """Call if you change DB contents mid-run and need fresh reads."""
    _MAP_CACHE.clear()
    _FETCH_CACHE.clear()

# ---------- Thread text/json document generation (on demand) ----------

THREAD_EXPORT_DIR_DEFAULT = "thread_outputs"

_threads_queries_initialized = False
_threads_queries = {}

# make a zip from one aggregate row of the pow_report
def _write_single_row_zip(
    zip_dir: str,
    export_dir: str,
    zip_name: str,
    tids: List[int],
    thread_paths_map: Dict[int, Dict[str, Optional[str]]],
    compresslevel: int = 9
) -> str:
    """
    Build exactly one row ZIP atomically:
      - If the target ZIP already exists, skip and return its path.
      - Otherwise, write to a temporary file in zip_dir
      - Atomically replace <zip_dir>/<zip_name> using os.replace()

    De-dup logic:
      - Tracks each arcname added (e.g., 'thread_123.txt') in a set and skips
        any subsequent attempt to add the same arcname, preventing
        `zipfile.UserWarning: Duplicate name: ...`.

    Returns the final path on success. Raises on failure.
    """
    ensure_dir(zip_dir)
    final_path = os.path.join(zip_dir, zip_name)

    # Skip if already present (consistent with "skip if exists" policy)
    if os.path.exists(final_path):
        tqdm.write(f"[INFO] ZIP {final_path} already exists — skipping.")
        return final_path

    # Create temp file in the same directory for atomic replace
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".zip", dir=zip_dir)
    os.close(tmp_fd)  # will reopen via ZipFile

    def _maybe_add(zf: zipfile.ZipFile, fp: str, added: set) -> bool:
        """
        Attempt to add file `fp` to zip with arcname=basename(fp),
        skipping if that arcname is already present.
        Returns True if added, False if skipped or path missing.
        """
        try:
            if not fp or not os.path.exists(fp):
                return False
            arc = os.path.basename(fp)
            if arc in added:
                return False
            zf.write(fp, arcname=arc)
            added.add(arc)
            return True
        except Exception as e:
            tqdm.write(f"[build_row_zip_thread:_maybe_add] {e}")
            traceback.print_exc()
            return False

    added_names: set = set()

    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as zf:
            for tid in tqdm(
                sorted(set(int(t) for t in tids)),
                desc="Adding threads",
                unit="thread",
                leave=False,
                dynamic_ncols=True,
            ):
                base = f"thread_{tid}"

                # Prefer freshly exported artifacts when present, but
                # regardless of order we suppress duplicate arcnames.
                txt_fp = os.path.join(export_dir, f"{base}.txt")
                json_fp = os.path.join(export_dir, f"{base}.json")
                _maybe_add(zf, txt_fp, added_names)
                _maybe_add(zf, json_fp, added_names)

                # Also consider any indexed source paths (cache on disk).
                paths = thread_paths_map.get(int(tid), {}) or {}
                _maybe_add(zf, paths.get("txt"), added_names)
                _maybe_add(zf, paths.get("json"), added_names)

        # Atomically move into place (no pre-delete; avoids race-to-nothing)
        os.replace(tmp_path, final_path)
        return final_path

    except Exception as e:
        tqdm.write(f"[build_row_zip_thread] {e}")
        traceback.print_exc()
        # Best-effort cleanup of the temp file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as ce:
            tqdm.write(f"[cleanup_tmp_zip] {ce}")
            traceback.print_exc()
        raise

def build_row_zips_threaded(
    memberships: Dict[Tuple[str, str, Optional[int]], set],
    thread_paths_map: Dict[int, Dict[str, Optional[str]]],
    zip_dir: str,
    export_dir: str,
    max_workers: Optional[int] = None,
    compresslevel: int = 9
) -> None:
    """
    Build one ZIP per aggregate row using a thread pool.
    Delegates all ZIP creation to `_write_single_row_zip`, which:
      - skips if the final ZIP already exists,
      - writes to a temp file, and
      - atomically replaces into place.

    Prints progress via tqdm; exceptions include traceback for debugging clarity.
    """
    # Determine worker count
    if max_workers is None:
        try:
            cores = os.cpu_count() or 4
        except Exception:
            cores = 4
        # Keep a conservative cap unless the caller overrides
        max_workers = min(8, cores)

    ensure_dir(zip_dir)

    # Prepare tasks
    future_to_meta = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # tqdm the queueing loop
        for (section, group, key_val), tids in tqdm(
            memberships.items(),
            desc="Queueing row ZIPs",
            unit="row",
            dynamic_ncols=True,
            leave=False,
        ):
            zip_name = zip_name_for_row(section, group, key_val)
            final_path = os.path.join(zip_dir, zip_name)

            # Optionally fast-skip here (also re-checked inside _write_single_row_zip)
            if os.path.exists(final_path):
                tqdm.write(f"[build_row_zips_threaded] ZIP {final_path} already exists — skipping.")
                continue

            # Submit a job that just calls the shared helper
            fut = ex.submit(
                _write_single_row_zip,
                zip_dir=zip_dir,
                export_dir=export_dir,
                zip_name=zip_name,
                tids=list(tids),  # ensure it's serializable & stable
                thread_paths_map=thread_paths_map,
                compresslevel=compresslevel,
            )
            future_to_meta[fut] = (zip_name, len(tids))

        # Consume completions with a top-level progress bar
        for fut in tqdm(
            as_completed(future_to_meta),
            total=len(future_to_meta),
            desc="Building row ZIPs (threaded)",
            unit="zip",
            dynamic_ncols=True,
            leave=False,
        ):
            zip_name, n_threads = future_to_meta[fut]
            try:
                path = fut.result()
                tqdm.write(f"[build_row_zips_threaded] [zip] {zip_name} ({n_threads} threads) -> {path}")
            except Exception as e:
                tqdm.write(f"[build_row_zips_threaded] [zip_failed] {zip_name}: {e}")
                traceback.print_exc()

def build_thread_file_index(root_dir: str) -> Dict[int, Dict[str, str]]:
    """
    Recursively scan `root_dir` for files named exactly thread_{id}.txt or thread_{id}.json
    and return a mapping:  thread_id -> {'txt': <abs_path or None>, 'json': <abs_path or None>}.

    Robust to partial availability (e.g., only .txt exists).
    """
    index: Dict[int, Dict[str, str]] = defaultdict(lambda: {"txt": None, "json": None})

    if not root_dir or not os.path.isdir(root_dir):
        return {}

    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            m = re.match(r"^thread_(\d+)\.(txt|json)$", fn, flags=re.IGNORECASE)
            if not m:
                continue
            tid = int(m.group(1))
            ext = m.group(2).lower()
            abs_path = os.path.join(dirpath, fn)
            # Keep the first one we see; if duplicates exist, prefer the first (closest).
            if ext == "txt" and not index[tid]["txt"]:
                index[tid]["txt"] = abs_path
            elif ext == "json" and not index[tid]["json"]:
                index[tid]["json"] = abs_path

    # Convert defaultdict to normal dict
    return {tid: paths for tid, paths in index.items()}
    
def copy_existing_thread_files(thread_id: int, source_paths: Dict[str, Optional[str]], export_dir: str) -> Tuple[bool, List[str]]:
    """
    Attempt to copy thread_{id}.txt/.json from source_paths into export_dir.
    Returns (success_copied_any, copied_paths).

    success_copied_any=True if at least one of (.txt or .json) existed and was copied.
    """
    ensure_dir(export_dir)
    copied: List[str] = []
    ok_any = False

    try:
        base = f"thread_{thread_id}"
        # TXT
        src_txt = source_paths.get("txt")
        if src_txt and os.path.isfile(src_txt):
            dst_txt = os.path.join(export_dir, f"{base}.txt")
            shutil.copy2(src_txt, dst_txt)
            copied.append(dst_txt)
            ok_any = True

        # JSON
        src_json = source_paths.get("json")
        if src_json and os.path.isfile(src_json):
            dst_json = os.path.join(export_dir, f"{base}.json")
            shutil.copy2(src_json, dst_json)
            copied.append(dst_json)
            ok_any = True

    except Exception as e:
        safe_print_err("copy_existing_thread_files", e)

    return ok_any, copied    

def _init_threads_queries():
    global _threads_queries_initialized, _threads_queries
    if _threads_queries_initialized:
        return
    _threads_queries = {
        "threads": """
            SELECT 
                t.id AS thread_id, 
                z.text AS puzzle_text,
                du.first_name || ' ' || du.last_name AS mentor_name
            FROM pow_threads t
            LEFT JOIN pow_publications p ON t.publication = p.id
            LEFT JOIN pow_puzzles z ON p.puzzle = z.id
            LEFT JOIN dir_users du ON t.mentor = du.id
            WHERE t.id = ?
        """,
        "submissions": """
            SELECT
                s.id AS submission_id,
                s.thread_id AS s_thread_id,
                s.shortanswer AS s_shortanswer,
                s.longanswer AS s_longanswer,
                s.createdate AS submission_date,
                r.message AS r_message,
                r.createdate AS response_date,
                rb.strategy AS rubric_strategy,
                rb.interpretation AS rubric_interpretation,
                rb.completeness AS rubric_completeness,
                rb.clarity AS rubric_clarity,
                rb.reflection AS rubric_reflection,
                rb.accuracy AS rubric_accuracy
            FROM pow_submissions s
            LEFT JOIN pow_responses r ON r.submission_id = s.id
            LEFT JOIN pow_rubric rb ON r.rubric_id = rb.id
            WHERE s.thread_id = ?
            ORDER BY s.createdate, r.createdate
        """
    }
    _threads_queries_initialized = True

_illegal_characters_re = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def extract_base64_placeholders(field_label, text):
    if not isinstance(text, str):
        return "", []
    pattern = re.compile(r'<[^>]+?base64,([^"\'>\s]+)[^>]*>', re.IGNORECASE)
    b64_list, placeholders = [], []
    def repl(match):
        idx = len(b64_list) + 1
        b64_data = match.group(1)
        placeholder = f"[{field_label} Image {idx}]"
        b64_list.append({"field": field_label, "index": idx, "base64": b64_data})
        placeholders.append(placeholder)
        return placeholder
    new_text = pattern.sub(repl, text)
    return new_text, b64_list

def strip_html(value):
    if not isinstance(value, str):
        return ""
    text = re.sub(r'<[^>]+>', '', value)
    return html.unescape(text.strip())

def sanitize_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r'[\r\n\t]', ' ', text)
    text = re.sub(r'\\[rnt]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def _get_student_school(cursor, submission_ids):
    student_name, school_name, age = "", "", ""
    if not submission_ids:
        return {"student_name": student_name, "school_name": school_name, "age": age}

    creator_id = None
    for sid in submission_ids:
        cursor.execute("SELECT creator FROM pow_submissions WHERE id = ?", (sid,))
        row = cursor.fetchone()
        if row and row["creator"]:
            creator_id = row["creator"]; break

    if not creator_id:
        return {"student_name": student_name, "school_name": school_name, "age": age}

    cursor.execute("SELECT first_name, last_name, ageinyears FROM dir_users WHERE id = ?", (creator_id,))
    user_row = cursor.fetchone()
    if user_row:
        first = user_row["first_name"] or ""
        last = user_row["last_name"] or ""
        student_name = f"{first} {last}".strip()
        age = str(user_row["ageinyears"]) if user_row["ageinyears"] is not None else ""

    cursor.execute("""
        SELECT g.name AS school_name
        FROM dir_memberships m
        JOIN dir_groups g ON m.group_id = g.id
        WHERE m.user_id = ? 
        ORDER BY m.createdate DESC LIMIT 1
    """, (creator_id,))
    school_row = cursor.fetchone()
    if school_row:
        school_name = school_row["school_name"] or ""

    return {"student_name": student_name, "school_name": school_name, "age": age}

def _write_thread_output_files(conn: sqlite3.Connection, thread_id: int, out_dir: str) -> None:
    """
    Write thread_{id}.txt and thread_{id}.json into out_dir for the given thread.
    Safe to call repeatedly; overwrites each time to ensure freshness.
    """
    _init_threads_queries()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Thread header data
    cur.execute(_threads_queries["threads"], (thread_id,))
    th = cur.fetchone()
    if not th:
        return
    puzzle_text = (th["puzzle_text"] or "").strip()
    mentor_name = (th["mentor_name"] or "").strip()

    # Conversation rows
    cur.execute(_threads_queries["submissions"], (thread_id,))
    rows = [dict(r) for r in cur.fetchall()]
    submission_ids = [r.get("submission_id") for r in rows if r.get("submission_id")]

    student_meta = _get_student_school(cur, submission_ids)

    plain_puzzle_text = sanitize_text(strip_html(puzzle_text))
    text_lines = [f"Problem statement: {plain_puzzle_text}", ""]
    if student_meta.get("student_name"):
        text_lines.append(f"Student Name: {student_meta['student_name']}")
    if student_meta.get("age"):
        text_lines.append(f"Age: {student_meta['age']}")
    if student_meta.get("school_name"):
        text_lines.append(f"School: {student_meta['school_name']}")
    text_lines.append("")
    if mentor_name:
        text_lines.append(f"Mentor Name: {mentor_name}")
        text_lines.append("")

    json_output = {
        "thread_id": thread_id,
        "puzzle_text": plain_puzzle_text,
        "conversation": [],
        "student_name": student_meta.get("student_name",""),
        "school_name": student_meta.get("school_name",""),
        "age": student_meta.get("age",""),
        "mentor_name": mentor_name,
    }

    for idx, row in enumerate(rows):
        image_blobs = []
        s_short_text, imgs1 = extract_base64_placeholders("Short Answer", row.get("s_shortanswer", ""))
        s_long_text,  imgs2 = extract_base64_placeholders("Long Answer",  row.get("s_longanswer", ""))
        r_msg_text,   imgs3 = extract_base64_placeholders("Mentor Message", row.get("r_message", ""))

        image_blobs.extend(imgs1 + imgs2 + imgs3)

        s_short = sanitize_text(strip_html(s_short_text))
        s_long  = sanitize_text(strip_html(s_long_text))
        r_msg   = sanitize_text(strip_html(r_msg_text))

        s_date = row.get("submission_date", "")
        r_date = row.get("response_date", "")

        rubric_fields = {}
        for rubric_key in ["strategy","interpretation","accuracy","completeness","clarity","reflection"]:
            rubric_fields[rubric_key] = row.get(f"rubric_{rubric_key}", 0)

        s_date_text = f"(submitted on {s_date}) " if s_date else ""
        text_lines.append(f"Student Submission {s_date_text}Short Answer {idx+1}: {s_short}")
        text_lines.append("")
        text_lines.append(f"Student Submission {s_date_text}Long Answer {idx+1}: {s_long}")
        text_lines.append("")
        r_date_text = f"(responded on {r_date}) " if r_date else ""
        text_lines.append(f"Mentor Response {r_date_text}{idx+1}: {r_msg}" if r_msg else "Mentor Response: (No reply yet)")
        text_lines.append("")
        if image_blobs:
            for img in image_blobs:
                text_lines.append(f"Embedded image: {img['field']} Image {img['index']} {img['base64']}")
            text_lines.append("")
        if any(v is not None for v in rubric_fields.values()):
            rubric_text = "; ".join(f"{k.capitalize()}: {v}" for k,v in rubric_fields.items() if v is not None)
            text_lines.append(f"Rubric {idx+1}: {rubric_text}")
            text_lines.append("")

        json_output["conversation"].append({
            "submission_id": row.get("submission_id"),
            "submission_date": s_date,
            "short_answer": s_short,
            "long_answer": s_long,
            "response": r_msg,
            "response_date": r_date,
            "rubrics": rubric_fields,
            "images": image_blobs
        })

    ensure_dir(out_dir)
    txt_path = os.path.join(out_dir, f"thread_{thread_id}.txt")
    json_path = os.path.join(out_dir, f"thread_{thread_id}.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

def ensure_thread_documents(conn: sqlite3.Connection,
                            thread_ids: List[int],
                            export_dir: str,
                            source_index: Optional[Dict[int, Dict[str, str]]] = None
                            ) -> Dict[int, Dict[str, Optional[str]]]:
    """
    Returns a mapping thread_id -> {'txt': path, 'json': path} for ALL thread_ids.
    If a thread exists in source_index, we record its paths directly (no copy).
    If not, we generate fresh files in export_dir and record those paths.
    """
    ensure_dir(export_dir)
    thread_paths: Dict[int, Dict[str, Optional[str]]] = {}

    for tid in tqdm(thread_ids, desc="Ensuring thread documents", unit="thread", dynamic_ncols=True):
        tid_int = int(tid)
        if source_index and tid_int in source_index:
            # Use preexisting paths directly
            thread_paths[tid_int] = source_index[tid_int]
        else:
            # Generate new files into export_dir
            try:
                _write_thread_output_files(conn, tid_int, export_dir)
                thread_paths[tid_int] = {
                    "txt": os.path.join(export_dir, f"thread_{tid_int}.txt"),
                    "json": os.path.join(export_dir, f"thread_{tid_int}.json")
                }
            except Exception as e:
                safe_print_err("ensure_thread_documents", e)
                thread_paths[tid_int] = {"txt": None, "json": None}

    return thread_paths

# ------------------------------
# Helpers & analytics 
# ------------------------------

def zip_name_for_row(section: str, group: str, key: Optional[int]) -> str:
    # Filename: "<A> + <B> + key_<C>.zip" (key included to avoid collisions)
    a = sanitize_filename(section)
    b = sanitize_filename(group)
    k = f"key_{key}" if key is not None else "key_NULL"
    return f"{a} + {b} + {k}.zip"

def build_zip_for_row(zip_dir: str, zip_name: str, export_dir: str, thread_ids: List[int]) -> str:
    """
    Create a zip in zip_dir named zip_name that includes thread_{id}.txt and .json
    from export_dir for every id in thread_ids. Returns the full zip path.
    """
    ensure_dir(zip_dir)
    zip_path = os.path.join(zip_dir, zip_name)
    # Overwrite to reflect current selection
    if os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except Exception as e:
            safe_print_err("remove_zip", e)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for tid in sorted(set(int(t) for t in thread_ids)):
            base = f"thread_{tid}"
            txt_fp = os.path.join(export_dir, f"{base}.txt")
            json_fp = os.path.join(export_dir, f"{base}.json")
            # Both files are expected to exist (we generate them beforehand),
            # but add guards in case of concurrent deletions.
            if os.path.exists(txt_fp):
                zf.write(txt_fp, arcname=os.path.basename(txt_fp))
            if os.path.exists(json_fp):
                zf.write(json_fp, arcname=os.path.basename(json_fp))
    return zip_path

def ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        safe_print_err("ensure_dir", e)

def sanitize_filename(s: str, maxlen: int = 160) -> str:
    """
    Sanitize a filename by removing/normalizing path-unfriendly characters and trimming length.
    """
    if s is None:
        s = "NULL"
    s = str(s)
    s = s.replace("/", "_").replace("\\", "_").replace(":", " - ").replace("*", "_")
    s = s.replace("?", "_").replace('"', "'").replace("<", "(").replace(">", ")").replace("|", "_")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > maxlen:
        s = s[:maxlen].rstrip()
    return s or "NULL"

def extract_json(text: str) -> Optional[dict]:
    try:
        while text.count('{') != text.count('}'):
            if text.count('{') < text.count('}'):
                text = '{' + text
            else:
                text += '}'        
            
        json_match = re.search(r"\{[\s\S]*?\}", text.strip())
        if not json_match:
            safe_print_err("extract_json", Exception(f"No JSON braces found in response: {text}"))
            return None
        json_str = json_match.group(0)

        tqdm.write(f"[extract_json] JSON candidate:\n{json_str}")

        return json.loads(json_str)
    except json.JSONDecodeError as e:
        safe_print_err("json_decode_error", e)
        return None
    except Exception as e:
        safe_print_err("extract_json_unexpected_error", e)
        return None

def is_active_deleted_clause(ignore_deleted: bool, alias: str, col: str = "deleted") -> str:
    if ignore_deleted:
        return f"({alias}.{col} IS NULL OR {alias}.{col} = 0)"
    else:
        return "1"

def safe_print_err(prefix: str, e: Exception) -> None:
    tqdm.write(f"[{prefix}] {e}", file=sys.stderr)
    traceback.print_exc()

def median_safe(vals: List[float]) -> Optional[float]:
    try:
        return statistics.median(vals) if vals else None
    except Exception as e:
        safe_print_err("median", e)
        return None

def pearson_r(x: List[float], y: List[float]) -> Tuple[Optional[float], Optional[int]]:
    try:
        if not x or not y or len(x) != len(y):
            return (None, 0)
        n = len(x)
        if n < 3:
            return (None, n)
        mx = sum(x) / n
        my = sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        denx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
        deny = math.sqrt(sum((yi - my) ** 2 for yi in y))
        if denx == 0 or deny == 0:
            return (None, n)
        return (num / (denx * deny), n)
    except Exception as e:
        safe_print_err("pearson_r", e)
        return (None, None)

def pct(part: int, whole: int) -> float:
    return round((100.0 * part / whole), 2) if whole else 0.0

def fetch_threads(conn: sqlite3.Connection, ignore_deleted: bool, disable_cache: bool) -> List[dict]:
    cached = _get_fetch_cache("threads")
    if cached is not None:
        return cached

    q = f"""
        SELECT
            t.id            AS thread_id,
            t.creator       AS thread_creator,
            t.mentor        AS mentor_user_id,
            t.publication   AS publication_id
        FROM pow_threads t
        WHERE {is_active_deleted_clause(ignore_deleted, 't')}
    """
    cur = conn.cursor()
    cur.execute(q)
    rows = cur.fetchall()
    out: List[dict] = []
    for r in tqdm(rows, desc="Loading threads", unit="thread", dynamic_ncols=True):
        out.append(dict(r))
    return _set_fetch_cache(disable_cache, "threads", out)

def build_service_map(conn: sqlite3.Connection, ignore_deleted: bool, disable_cache: bool) -> Dict[int, str]:
    cached = _get_map_cache("service_map")
    if cached is not None:
        return cached

    q = f"""
        SELECT mp.newpublicationid AS publication_id, mp.servicename AS service_name
        FROM pow_migrated_puzzles mp
        WHERE {is_active_deleted_clause(ignore_deleted, 'mp')}
    """
    cur = conn.cursor()
    cur.execute(q)
    rows = cur.fetchall()
    mapping: Dict[int, str] = {}
    for pub_id, svc in tqdm(rows, desc="Building service_map", unit="row", dynamic_ncols=True):
        if pub_id is not None and pub_id not in mapping:
            mapping[int(pub_id)] = svc
    return _set_map_cache(disable_cache, "service_map", mapping)
    
def build_mentor_name_map(conn: sqlite3.Connection, disable_cache: bool) -> Dict[int, str]:
    cached = _get_map_cache("mentor_name_map")
    if cached is not None:
        return cached

    cur = conn.cursor()
    cur.execute("""
        SELECT id,
               TRIM(COALESCE(NULLIF(first_name,''),'') || ' ' || COALESCE(NULLIF(last_name,''),'')) AS full_name,
               user_name
        FROM dir_users
    """)
    rows = cur.fetchall()
    m: Dict[int, str] = {}
    for uid, full, uname in tqdm(rows, desc="Building mentor_name_map", unit="user", dynamic_ncols=True):
        if uid is None:
            continue
        name = (full or "").strip() or (uname or f"user_{uid}")
        m[int(uid)] = name
    return _set_map_cache(disable_cache, "mentor_name_map", m)

def build_teacher_map(conn: sqlite3.Connection, disable_cache: bool) -> Dict[int, Tuple[int, str]]:
    """
    Return a mapping: publication_id -> (teacher_user_id, teacher_full_name)

    Memory-friendly iterative version:
      - Iterate over publication_ids one at a time (no massive intermediate joins).
      - For each publication, query the minimal teacher user_id (per your rule).
      - Then look up the name for that user_id.
      - tqdm progress bar uses a lightweight pre-count of distinct publication_ids.

    Notes:
      * Progress 'total' counts all publications in pow_pub_submission_groups
        (even those without a qualifying teacher). This keeps pre-count fast.
      * Ensure helpful indexes exist for good performance (see docstring).
    """
    cached = _get_map_cache("teacher_map")
    if cached is not None:
        return cached

    teacher_role_ids = (4, 7, 13)

    try:
        with closing(conn.cursor()) as cur_count:
            # Fast(ish) upper bound for tqdm; avoids heavy joins.
            cur_count.execute(
                """
                SELECT COUNT(DISTINCT publication_id)
                FROM pow_pub_submission_groups
                WHERE publication_id IS NOT NULL
                """
            )
            
            # Or, for an exact count
            # cur_count.execute(f"""
            #     SELECT COUNT(DISTINCT psg.publication_id)
            #     FROM pow_pub_submission_groups psg
            #     JOIN dir_memberships dm ON dm.group_id = psg.group_id
            #     WHERE (dm.deleted IS NULL OR dm.deleted = 0)
            #       AND dm.role IN ({",".join(str(r) for r in teacher_role_ids)})
            #       AND psg.publication_id IS NOT NULL
            #       AND dm.user_id IS NOT NULL
            # """)
            
            total_pubs = cur_count.fetchone()[0] or 0

        # Prepared statements for the per-publication lookups
        min_teacher_sql = f"""
            SELECT MIN(dm.user_id)
            FROM pow_pub_submission_groups psg
            JOIN dir_memberships dm ON dm.group_id = psg.group_id
            WHERE (dm.deleted IS NULL OR dm.deleted = 0)
              AND dm.role IN ({",".join(str(r) for r in teacher_role_ids)})
              AND psg.publication_id = ?
              AND dm.user_id IS NOT NULL
        """

        name_sql = """
            SELECT first_name, last_name, user_name
            FROM dir_users
            WHERE id = ?
        """

        teacher_map: Dict[int, Tuple[int, str]] = {}

        with closing(conn.cursor()) as cur_pub, closing(conn.cursor()) as cur_detail:
            # Stream all publication_ids without DISTINCT or joins to keep memory low.
            cur_pub.execute(
                """
                SELECT publication_id
                FROM pow_pub_submission_groups
                WHERE publication_id IS NOT NULL
                GROUP BY publication_id
                """
            )
            # NOTE: GROUP BY publication_id is usually implemented via sort/aggregate in SQLite.
            # If that becomes memory-heavy, you can: (1) drop GROUP BY and accept duplicates
            # with a seen-set (might be big), or (2) create an index on publication_id and
            # rely on DISTINCT/GROUP BY reading via the index. The index is preferred.

            for (pub_id,) in tqdm(
                cur_pub, desc="Building teacher_map (iterative)", unit="pub", total=total_pubs, dynamic_ncols=True
            ):
                # 1) Get the minimal teacher user_id for this publication (if any)
                cur_detail.execute(min_teacher_sql, (pub_id,))
                row = cur_detail.fetchone()
                if not row or row[0] is None:
                    continue  # No qualifying teacher for this publication

                user_id = int(row[0])

                # 2) Fetch the user's display name
                cur_detail.execute(name_sql, (user_id,))
                name_row = cur_detail.fetchone()
                if name_row:
                    first, last, uname = name_row
                else:
                    first = last = None
                    uname = None

                full_name = f"{(first or '').strip()} {(last or '').strip()}".strip()
                if not full_name:
                    full_name = (uname or f"user_{user_id}").strip()

                teacher_map[int(pub_id)] = (user_id, full_name)

        return _set_map_cache(disable_cache, "teacher_map", teacher_map)

    except Exception as e:
        tqdm.write(f"[build_teacher_map] Exception: {e}")
        traceback.print_exc()
        # In case of error, fail gracefully with an empty mapping (or re-raise if you prefer)
        return {}

def build_school_map(conn: sqlite3.Connection, ignore_deleted: bool, disable_cache: bool) -> Dict[int, str]:
    cached = _get_map_cache("school_map")
    if cached is not None:
        return cached

    q = f"""
        SELECT psg.publication_id, g.name
        FROM pow_pub_submission_groups psg
        JOIN dir_groups g ON g.id = psg.group_id
        WHERE {is_active_deleted_clause(ignore_deleted, 'g')}
    """
    cur = conn.cursor()
    cur.execute(q)
    rows = cur.fetchall()
    tmp: Dict[int, List[str]] = defaultdict(list)
    for pub_id, gname in tqdm(rows, desc="Building school_map", unit="row", dynamic_ncols=True):
        if pub_id is None:
            continue
        name = (gname or "").strip()
        if name:
            tmp[int(pub_id)].append(name)
    school_map: Dict[int, str] = {}
    for pub, names in tmp.items():
        school_map[pub] = sorted(set(names))[0]
    return _set_map_cache(disable_cache, "school_map", school_map)

def fetch_submission_counts_by_thread_creator(conn: sqlite3.Connection, ignore_deleted: bool, disable_cache: bool) -> Dict[int, int]:
    cached = _get_fetch_cache("subs_by_creator")
    if cached is not None:
        return cached

    q = f"""
        SELECT s.thread_id, t.creator, COUNT(*) AS cnt
        FROM pow_submissions s
        JOIN pow_threads t ON t.id = s.thread_id
        WHERE {is_active_deleted_clause(ignore_deleted, 's')}
          AND {is_active_deleted_clause(ignore_deleted, 't')}
          AND s.creator = t.creator
        GROUP BY s.thread_id, t.creator
    """
    cur = conn.cursor()
    cur.execute(q)
    rows = cur.fetchall()
    result: Dict[int, int] = {}
    for thread_id, _creator, cnt in tqdm(rows, desc="Building subs_by_creator", unit="thread", dynamic_ncols=True):
        result[int(thread_id)] = int(cnt)
    return _set_fetch_cache(disable_cache, "subs_by_creator", result)

def fetch_mentor_reply_counts(conn: sqlite3.Connection, ignore_deleted: bool, disable_cache: bool) -> Dict[int, int]:
    cached = _get_fetch_cache("replies_by_thread")
    if cached is not None:
        return cached

    q = f"""
        SELECT s.thread_id, COUNT(r.id) AS cnt
        FROM pow_responses r
        JOIN pow_submissions s ON s.id = r.submission_id
        WHERE {is_active_deleted_clause(ignore_deleted, 'r')}
          AND {is_active_deleted_clause(ignore_deleted, 's')}
        GROUP BY s.thread_id
    """
    cur = conn.cursor()
    cur.execute(q)
    rows = cur.fetchall()
    result: Dict[int, int] = {}
    for thread_id, cnt in tqdm(rows, desc="Building replies_by_thread", unit="thread", dynamic_ncols=True):
        result[int(thread_id)] = int(cnt)
    return _set_fetch_cache(disable_cache, "replies_by_thread", result)

def fetch_scored_deltas(conn: sqlite3.Connection, ignore_deleted: bool, disable_cache: bool) -> Dict[int, Dict[str, float]]:
    cached = _get_fetch_cache("deltas_by_thread")
    if cached is not None:
        return cached

    q = f"""
        SELECT
            s.thread_id AS thread_id,
            r.createdate AS response_date,
            COALESCE(rb.strategy,0),
            COALESCE(rb.interpretation,0),
            COALESCE(rb.accuracy,0),
            COALESCE(rb.completeness,0),
            COALESCE(rb.clarity,0),
            COALESCE(rb.reflection,0)
        FROM pow_responses r
        JOIN pow_submissions s ON s.id = r.submission_id
        JOIN pow_rubric rb ON rb.id = r.rubric_id
        WHERE r.rubric_id IS NOT NULL
          AND {is_active_deleted_clause(ignore_deleted, 'r')}
          AND {is_active_deleted_clause(ignore_deleted, 's')}
        ORDER BY s.thread_id, r.createdate
    """
    cur = conn.cursor()
    cur.execute(q)
    rows = cur.fetchall()

    per_thread: Dict[int, List[Tuple[str, float, float, float, float, float, float]]] = defaultdict(list)
    for thread_id, response_date, strategy, interpretation, accuracy, completeness, clarity, reflection in tqdm(
        rows, desc="Accumulating scored rubric rows", unit="row", dynamic_ncols=True
    ):
        per_thread[int(thread_id)].append((
            response_date, float(strategy), float(interpretation),
            float(accuracy), float(completeness), float(clarity), float(reflection)
        ))

    deltas: Dict[int, Dict[str, float]] = {}
    for tid, items in tqdm(per_thread.items(), desc="Computing rubric deltas per thread", unit="thread", dynamic_ncols=True):
        if not items:
            continue
        first = items[0]
        last  = items[-1]
        diffs = [last[i] - first[i] for i in range(1, 7)]
        total_first = sum(first[1:7])
        total_last  = sum(last[1:7])
        deltas[tid] = {
            "delta_total": total_last - total_first,
            "delta_strategy": diffs[0],
            "delta_interpretation": diffs[1],
            "delta_accuracy": diffs[2],
            "delta_completeness": diffs[3],
            "delta_clarity": diffs[4],
            "delta_reflection": diffs[5],
        }
    return _set_fetch_cache(disable_cache, "deltas_by_thread", deltas)

def fetch_reply_seconds_by_thread(conn: sqlite3.Connection, ignore_deleted: bool, disable_cache: bool) -> Dict[int, List[float]]:
    cached = _get_fetch_cache("reply_secs_map")
    if cached is not None:
        return cached

    q = f"""
        SELECT
            s.thread_id,
            (
                julianday(r.createdate / 1000, 'unixepoch') -
                julianday(s.createdate / 1000, 'unixepoch')
            ) * 86400.0 AS reply_seconds
        FROM pow_responses r
        JOIN pow_submissions s ON s.id = r.submission_id
        WHERE {is_active_deleted_clause(ignore_deleted, 'r')}
          AND {is_active_deleted_clause(ignore_deleted, 's')}
          AND r.createdate IS NOT NULL
          AND s.createdate IS NOT NULL
    """
    cur = conn.cursor()
    cur.execute(q)
    rows = cur.fetchall()
    data: Dict[int, List[float]] = defaultdict(list)
    for thread_id, reply_seconds in tqdm(rows, desc="Building reply_secs_map", unit="row", dynamic_ncols=True):
        if thread_id is not None and reply_seconds is not None:
            data[int(thread_id)].append(float(reply_seconds))
    return _set_fetch_cache(disable_cache, "reply_secs_map", data)

def mrubric_category_breakdown_by(mrubric_rows: List[dict], group_key: str) -> List[dict]:
    """
    Build mrubric category distributions grouped by a single factor (mentor_label or school_name).

    Output rows have:
      group_key, group_value, category, score_level, count, percent

    Notes:
      * Uses MRUBRIC_CATEGORIES (1–3).
      * Skips rows where score is missing for a category.
      * Treats empty/None group values as "NULL".
    """
    from collections import defaultdict, Counter

    # nested[group_value][category][score_level] -> count
    nested: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    totals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in mrubric_rows:
        gval = r.get(group_key)
        if gval is None or (isinstance(gval, str) and gval.strip() == ""):
            gval = "NULL"

        for cat in MRUBRIC_CATEGORIES:
            lvl = r.get(cat)
            if lvl is None:
                continue
            lvl = int(lvl)
            nested[str(gval)][cat][lvl] += 1
            totals[str(gval)][cat] += 1

    out: List[dict] = []
    for gval, cat_map in nested.items():
        for cat, counter in cat_map.items():
            total = totals[gval][cat]
            for lvl in sorted(counter.keys()):
                cnt = int(counter[lvl])
                out.append({
                    "group_key": group_key,
                    "group_value": gval,
                    "category": cat,
                    "score_level": int(lvl),
                    "count": cnt,
                    "percent": pct(cnt, total),
                })
    return out

def base_report(conn: sqlite3.Connection, out_csv: str, ignore_deleted: bool) -> None:
    """
    Writes pow_report.csv AND, for each distribution row, creates a ZIP containing
    the thread text/json documents for exactly the threads counted on that row.

    ZIP output directory and thread export directory are controlled via args:
      --zip-output-dir (default: report_thread_zips)
      --thread-export-dir (default: thread_outputs)
      --disable-row-zips  (opt-out)
    """
    try:
        cur = conn.cursor()

        tqdm.write("[INFO] Fetching threads (fetch_threads)...")
        t0 = time.time()
        threads = fetch_threads(conn, args.ignore_deleted, args.pickle_disable_cache)
        tqdm.write(f"[INFO] Retrieved {len(threads)} threads. ({time.time() - t0:.2f}s)")

        tqdm.write("[INFO] Building service map (build_service_map)...")
        t0 = time.time()
        service_map = build_service_map(conn, args.ignore_deleted, args.pickle_disable_cache)
        tqdm.write(f"[INFO] build_service_map: {len(service_map)} entries. ({time.time() - t0:.2f}s)")

        tqdm.write("[INFO] Building teacher map (build_teacher_map)...")
        t0 = time.time()
        teacher_map = build_teacher_map(conn, args.pickle_disable_cache)
        tqdm.write(f"[INFO] build_teacher_map: {len(teacher_map)} entries. ({time.time() - t0:.2f}s)")

        tqdm.write("[INFO] Building school map (build_school_map)...")
        t0 = time.time()
        school_map = build_school_map(conn, ignore_deleted, args.pickle_disable_cache)
        tqdm.write(f"[INFO] build_school_map: {len(school_map)} entries. ({time.time() - t0:.2f}s)")

        tqdm.write("[INFO] Fetching submission counts by thread creator (fetch_submission_counts_by_thread_creator)...")
        t0 = time.time()
        subs_by_creator = fetch_submission_counts_by_thread_creator(conn, ignore_deleted, args.pickle_disable_cache)
        tqdm.write(f"[INFO] fetch_submission_counts_by_thread_creator: {len(subs_by_creator)} threads found. ({time.time() - t0:.2f}s)")

        tqdm.write("[INFO] Fetching mentor reply counts (fetch_mentor_reply_counts)...")
        t0 = time.time()
        replies_by_thread = fetch_mentor_reply_counts(conn, ignore_deleted, args.pickle_disable_cache)
        tqdm.write(f"[INFO] fetch_mentor_reply_counts: {len(replies_by_thread)} threads found. ({time.time() - t0:.2f}s)")

        tqdm.write("[INFO] Fetching scored rubric deltas (fetch_scored_deltas)...")
        t0 = time.time()
        deltas_by_thread = fetch_scored_deltas(conn, ignore_deleted, args.pickle_disable_cache)
        tqdm.write(f"[INFO] fetch_scored_deltas: {len(deltas_by_thread)} threads found. ({time.time() - t0:.2f}s)")

        tqdm.write("[INFO] Fetching reply time lists (fetch_reply_seconds_by_thread)...")
        t0 = time.time()
        reply_secs_map = fetch_reply_seconds_by_thread(conn, ignore_deleted, args.pickle_disable_cache)
        tqdm.write(f"[INFO] fetch_reply_seconds_by_thread: {len(reply_secs_map)} threads found. ({time.time() - t0:.2f}s)")

        tqdm.write("[INFO] Fetching mentor names (build_mentor_name_map)...")
        t0 = time.time()
        mentor_name_map = build_mentor_name_map(conn, args.pickle_disable_cache)
        tqdm.write(f"[INFO] build_mentor_name_map: {len(mentor_name_map)} users found. ({time.time() - t0:.2f}s)")
    except Exception as e:
        safe_print_err("fetch_base", e)
        raise

    # Enrich threads with computed features
    for t in tqdm(threads, desc="Computing thread features", unit="thread", dynamic_ncols=True):
        tid = int(t["thread_id"])
        t["mentor_reply_count"] = int(replies_by_thread.get(tid, 0))
        sub_cnt = int(subs_by_creator.get(tid, 0))
        t["student_revisions"] = max(sub_cnt - 1, 0)
        d = deltas_by_thread.get(tid, {})
        for k in ("delta_total","delta_strategy","delta_interpretation","delta_accuracy","delta_completeness","delta_clarity","delta_reflection"):
            t[k] = d.get(k)

        mid = t.get("mentor_user_id")
        mname = mentor_name_map.get(mid)
        t["mentor_name"] = mname
        t["mentor_label"] = f"{mid} - {mname}" if (mid is not None and mname) else (str(mid) if mid is not None else "NULL")

        pub_id = t.get("publication_id")
        t["service_name"] = service_map.get(pub_id)
        t["school_name"] = school_map.get(pub_id) or "NULL"

        times = reply_secs_map.get(tid, [])
        t["reply_seconds_median"] = median_safe(times)
        t["reply_seconds_mean"] = (sum(times)/len(times) if times else None)

    # Records (CSV) and membership map for ZIPs:
    #   key: (section, group, key) -> set(thread_ids)
    records: List[Dict[str, Any]] = []
    memberships: Dict[Tuple[str,str,Optional[int]], set] = defaultdict(set)

    def _record_members(section: str, group: str, key_val: Optional[int], thread_ids: List[int]) -> None:
        memberships[(section, str(group), key_val)].update(thread_ids)

    def add_distribution(section: str, group: str, counts: Counter, membership_lists: Dict[int, List[int]]) -> None:
        total = sum(counts.values())
        for key in sorted(counts.keys()):
            c = counts[key]
            records.append({
                "section": section,
                "group": group,
                "key": int(key),
                "count": int(c),
                "percent": pct(c, total),
            })
            # capture membership for this row
            tids = membership_lists.get(key, [])
            if tids:
                _record_members(section, group, int(key), tids)

    def add_metric(section: str, group: str, metric: str, value: Any) -> None:
        records.append({"section": section, "group": group, "metric": metric, "value": value})

    # A. mentor_replies_distribution (ALL)
    mentor_counts = Counter(int(t["mentor_reply_count"]) for t in threads)
    a_membership: Dict[int, List[int]] = defaultdict(list)
    for t in threads:
        a_membership[int(t["mentor_reply_count"])].append(int(t["thread_id"]))
    add_distribution("A. mentor_replies_distribution", "ALL", mentor_counts, a_membership)

    # B. student_revisions_distribution (ALL)
    rev_counts = Counter(int(t["student_revisions"]) for t in threads)
    b_membership: Dict[int, List[int]] = defaultdict(list)
    for t in threads:
        b_membership[int(t["student_revisions"])].append(int(t["thread_id"]))
    add_distribution("B. student_revisions_distribution", "ALL", rev_counts, b_membership)

    # C. scored thread counts (helper)
    try:
        q_scored_counts = f"""
            SELECT s.thread_id, COUNT(*) AS c
            FROM pow_responses r
            JOIN pow_submissions s ON s.id = r.submission_id
            WHERE r.rubric_id IS NOT NULL AND {is_active_deleted_clause(ignore_deleted, 'r')} AND {is_active_deleted_clause(ignore_deleted, 's')}
            GROUP BY s.thread_id
        """
        cur.execute(q_scored_counts)
        scored_count_map = {int(tid): int(c) for tid, c in cur.fetchall()}
    except Exception as e:
        safe_print_err("scored_counts", e)
        scored_count_map = {}

    threads_2plus_scored = [tid for tid, c in scored_count_map.items() if c >= 2]
    add_metric("C. scored_threads_summary", "ALL", "threads_with_2plus_scored_responses", len(threads_2plus_scored))

    # C. total_score_improvement_distribution (ALL; only threads with >=2 scored responses)
    eligible_tids = {tid for tid, c in scored_count_map.items() if c >= 2}
    delta_totals = [t["delta_total"] for t in threads if t.get("delta_total") is not None and t["thread_id"] in eligible_tids]
    c_membership: Dict[int, List[int]] = defaultdict(list)
    for t in threads:
        if t.get("delta_total") is None or t["thread_id"] not in eligible_tids:
            continue
        key_bin = int(round(t["delta_total"]))
        c_membership[key_bin].append(int(t["thread_id"]))
    delta_bins = Counter(int(round(dt)) for dt in delta_totals)
    add_distribution("C. total_score_improvement_distribution", "ALL", delta_bins, c_membership)

    cat_fields = [
        "delta_strategy",
        "delta_interpretation",
        "delta_accuracy",
        "delta_completeness",
        "delta_clarity",
        "delta_reflection",
    ]
    # Per-category rubric deltas (overall)
    for cat in cat_fields:
        cat_vals = [t.get(cat) for t in threads if (t.get(cat) is not None) and (t["thread_id"] in eligible_tids)]
        if not cat_vals:
            continue
        cat_bins = Counter(int(round(v)) for v in cat_vals)
        cat_member: Dict[int, List[int]] = defaultdict(list)
        for t in threads:
            v = t.get(cat)
            if v is None or t["thread_id"] not in eligible_tids:
                continue
            cat_member[int(round(v))].append(int(t["thread_id"]))
        add_distribution(f"C. {cat}_improvement_distribution", "ALL", cat_bins, cat_member)

    if delta_totals:
        mean_dt = round(sum(delta_totals)/len(delta_totals), 3)
        med_dt = round(statistics.median(delta_totals), 3)
    else:
        mean_dt = None
        med_dt = None
    add_metric("C. total_score_improvement_summary", "ALL", "mean_delta_total", mean_dt)
    add_metric("C. total_score_improvement_summary", "ALL", "median_delta_total", med_dt)

    pos_counts = {c: sum(1 for t in threads if t.get(c) is not None and t[c] > 0) for c in cat_fields}
    total_pos = sum(pos_counts.values())
    for c in cat_fields:
        records.append({
            "section": "C. category_positive_improvement_share",
            "group": "ALL",
            "category": c.replace("delta_",""),
            "count": int(pos_counts[c]),
            "percent": pct(pos_counts[c], total_pos)
        })

    # D. correlations (metrics; no zips)
    x_replies = [float(t["mentor_reply_count"]) for t in threads if (t.get("delta_total") is not None) and (t["thread_id"] in eligible_tids)]
    y_delta   = [float(t["delta_total"]) for t in threads if (t.get("delta_total") is not None) and (t["thread_id"] in eligible_tids)]
    r_replies, n_replies = pearson_r(x_replies, y_delta)
    records.append({"section": "D. correlation_replies_vs_score_change", "group": "ALL", "metric": "pearson_r", "value": None if r_replies is None else round(r_replies,4)})
    records.append({"section": "D. correlation_replies_vs_score_change", "group": "ALL", "metric": "n_pairs", "value": n_replies})

    x_revisions = [float(t["student_revisions"]) for t in threads if (t.get("delta_total") is not None) and (t["thread_id"] in eligible_tids)]
    r_revisions, n_revisions = pearson_r(x_revisions, y_delta)
    records.append({"section": "D. correlation_revisions_vs_score_change", "group": "ALL", "metric": "pearson_r", "value": None if r_revisions is None else round(r_revisions,4)})
    records.append({"section": "D. correlation_revisions_vs_score_change", "group": "ALL", "metric": "n_pairs", "value": n_revisions})

    # Helper to aggregate & capture memberships for E-* breakdowns
    def add_breakdown(section_prefix: str, group_key: str, value_accessor, bin_int: bool = False):
        buckets: Dict[Any, Counter] = defaultdict(Counter)
        bucket_members: Dict[Tuple[Any,int], List[int]] = defaultdict(list)  # (group_val, key_bin)->[tid]
        for t in threads:
            gval = t.get(group_key)
            if group_key in ("service_name",) and (gval is None or gval == ""):
                gval = "NULL"
            if group_key in ("mentor_user_id","teacher_user_id") and gval is None:
                gval = "NULL"
            val = value_accessor(t)
            if val is None:
                continue
            key_bin = int(round(val)) if bin_int else val
            buckets[gval][key_bin] += 1
            bucket_members[(gval, key_bin)].append(int(t["thread_id"]))

        for g, counts in buckets.items():
            total = sum(counts.values())
            for key in sorted(counts.keys(), key=lambda x: (x is None, x)):
                c = counts[key]
                records.append({
                    "section": section_prefix,
                    "group": str(g),
                    "key": int(key),
                    "count": int(c),
                    "percent": pct(c, total),
                })
                _record_members(section_prefix, str(g), int(key), bucket_members[(g, key)])

    add_breakdown("E1. mentor_replies_by_service", "service_name", lambda t: t["mentor_reply_count"], bin_int=False)
    add_breakdown("E2. student_revisions_by_service", "service_name", lambda t: t["student_revisions"], bin_int=False)
    add_breakdown("E3. total_score_improvement_by_service", "service_name", lambda t: t.get("delta_total"), bin_int=True)

    add_breakdown("E4. mentor_replies_by_mentor", "mentor_label", lambda t: t["mentor_reply_count"], bin_int=False)
    add_breakdown("E5. student_revisions_by_mentor", "mentor_label", lambda t: t["student_revisions"], bin_int=False)
    add_breakdown("E6. total_score_improvement_by_mentor", "mentor_label", lambda t: t.get("delta_total"), bin_int=True)

    add_breakdown("E7. mentor_replies_by_school", "school_name", lambda t: t["mentor_reply_count"], bin_int=False)
    add_breakdown("E8. student_revisions_by_school", "school_name", lambda t: t["student_revisions"], bin_int=False)
    add_breakdown("E9. total_score_improvement_by_school", "school_name", lambda t: t.get("delta_total"), bin_int=True)

    # Per-category rubric delta breakdowns by service/mentor/school
    for cat in cat_fields:
        add_breakdown(f"E3_cat.{cat}_by_service", "service_name", lambda t, c=cat: t.get(c), bin_int=True)
        add_breakdown(f"E6_cat.{cat}_by_mentor", "mentor_label", lambda t, c=cat: t.get(c), bin_int=True)
        add_breakdown(f"E9_cat.{cat}_by_school", "school_name", lambda t, c=cat: t.get(c), bin_int=True)

    # F. time_to_reply_median_hours_distribution (ALL)
    ttr_hours_counts = Counter()
    ttr_members: Dict[int, List[int]] = defaultdict(list)
    x_ttr, y_ttr = [], []
    for t in threads:
        med = t.get("reply_seconds_median")
        if med is not None:
            hours = int(round(med / 3600.0))
            ttr_hours_counts[hours] += 1
            ttr_members[hours].append(int(t["thread_id"]))
            if t.get("delta_total") is not None:
                x_ttr.append(float(med))
                y_ttr.append(float(t["delta_total"]))
    for key in sorted(ttr_hours_counts.keys()):
        c = ttr_hours_counts[key]
        records.append({
            "section": "F. time_to_reply_median_hours_distribution",
            "group": "ALL",
            "key": int(key),
            "count": int(c),
            "percent": pct(c, sum(ttr_hours_counts.values()))
        })
        _record_members("F. time_to_reply_median_hours_distribution", "ALL", int(key), ttr_members[key])

    r_ttr, n_ttr = pearson_r(x_ttr, y_ttr)
    records.append({"section": "F. correlation_reply_time_vs_score_change", "group": "ALL", "metric": "pearson_r", "value": None if r_ttr is None else round(r_ttr,4)})
    records.append({"section": "F. correlation_reply_time_vs_score_change", "group": "ALL", "metric": "n_pairs", "value": n_ttr})

    # Write CSV
    cols = ["section","group","key","count","percent","metric","value","category"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow(r)

    # Build ZIPs (if enabled)
    if not getattr(args, "disable_row_zips", False):
        zip_dir = getattr(args, "zip_output_dir", "report_thread_zips")
        export_dir = getattr(args, "thread_export_dir", THREAD_EXPORT_DIR_DEFAULT)

        # We only need to generate documents for the union of all thread IDs we reference
        all_needed_tids: List[int] = sorted({tid for tids in memberships.values() for tid in tids})
        tqdm.write(f"[INFO] Preparing thread documents for {len(all_needed_tids)} referenced threads...")

        # Optional fast path: reuse pre-existing thread artifacts if available.
        source_index = {}
        try:
            if args.thread_problem_outputs_dir and os.path.isdir(args.thread_problem_outputs_dir):
                tqdm.write(f"[INFO] Indexing existing thread artifacts under: {args.thread_problem_outputs_dir}")
                t0 = time.time()
                source_index = build_thread_file_index(args.thread_problem_outputs_dir) or {}
                tqdm.write(f"[INFO] Indexed {len(source_index)} thread IDs from disk. ({time.time() - t0:.2f}s)")
        except Exception as e:
            safe_print_err("build_thread_file_index", e)
            source_index = {}

        # Ensure artifacts and collect their paths
        thread_paths_map = ensure_thread_documents(
            conn,
            all_needed_tids,
            export_dir=export_dir,
            source_index=source_index
        )

        # --- write in parallel ---
        try:
            build_row_zips_threaded(
                memberships=memberships,
                thread_paths_map=thread_paths_map,
                zip_dir=zip_dir,
                export_dir=export_dir,
                max_workers=min(args.max_zip_workers, (os.cpu_count() or 4)),
                compresslevel=args.zip_compression_level            
            )
        except Exception as e:
            safe_print_err("build_row_zips_threaded", e)

# ------------------------------
# mrubric evaluation
# ------------------------------

MRUBRIC_CATEGORIES = [
    "build_on_student_thinking",
    "specificity_actionability",
    "mathematical_soundness",
    "questioning_scaffolding",
    "clarity_conciseness",
    "affect_support",
    "concise",
    "curious",
    "focused_questions",
    "accessible",
    "concrete",
    "collaborative",
    "specific",
    "encouraging",
    "nice_tone",
    "improve_work",
    "empowerment_confidence",
    "developing_thinking",
    "identity_competence",
    "efficiency",
    "encouraging_engagement",
    "process_oriented",
    "modeling_progression",
    "promote_curiosity",
    "individualized",
    "clear_actionable",
    "guiding_questions"
]

def fetch_replies_with_context(conn: sqlite3.Connection, ignore_deleted: bool, disable_cache: bool) -> List[dict]:
    """
    Pull mentor replies with enough context to score them:
    - response_id, thread_id, mentor_user_id, response.createdate
    - mentor message text
    - the most recent submission (shortanswer/longanswer, createdate) for that response
    - problem text (puzzles.text) for thread's publication
    - thread features for later grouping (mentor_reply_count, student_revisions, delta_total, service, teacher, reply_time)
    """
    cur = conn.cursor()

    # Base thread features (reuse base computations)
    threads = fetch_threads(conn, ignore_deleted, disable_cache)
    service_map = build_service_map(conn, ignore_deleted, disable_cache)
    teacher_map = build_teacher_map(conn, disable_cache)
    subs_by_creator = fetch_submission_counts_by_thread_creator(conn, ignore_deleted, disable_cache)
    replies_by_thread = fetch_mentor_reply_counts(conn, ignore_deleted, disable_cache)
    deltas_by_thread = fetch_scored_deltas(conn, ignore_deleted, disable_cache)
    reply_secs_map = fetch_reply_seconds_by_thread(conn, ignore_deleted, disable_cache)
    school_map = build_school_map(conn, ignore_deleted, disable_cache)

    thread_feat = {}
    for t in threads:
        tid = int(t["thread_id"])
        feat = {
            "thread_id": tid,
            "mentor_user_id": t.get("mentor_user_id"),
            "publication_id": t.get("publication_id"),
            "mentor_reply_count": int(replies_by_thread.get(tid, 0)),
            "student_revisions": max(int(subs_by_creator.get(tid, 0)) - 1, 0),
            "service_name": service_map.get(t.get("publication_id")),
            "teacher_user_id": teacher_map.get(t.get("publication_id"))[0] if isinstance(teacher_map.get(t.get("publication_id")), tuple) else teacher_map.get(t.get("publication_id")),
            "teacher_name": teacher_map.get(t.get("publication_id"))[1] if isinstance(teacher_map.get(t.get("publication_id")), tuple) else teacher_map.get(t.get("publication_id")),
            "school_name": school_map.get(t.get("publication_id")),
        }
        d = deltas_by_thread.get(tid, {})
        feat["delta_total"] = d.get("delta_total")
        times = reply_secs_map.get(tid, [])
        feat["reply_seconds_median"] = median_safe(times)
        thread_feat[tid] = feat

    # Replies + nearest prior submission and problem text
    q = f"""
        SELECT
            r.id AS response_id,
            r.message AS mentor_message,
            r.createdate AS response_date,
            s.id AS submission_id,
            s.thread_id AS thread_id,
            s.createdate AS submission_date,
            s.shortanswer AS s_short,
            s.longanswer  AS s_long,
            t.publication AS publication_id
        FROM pow_responses r
        JOIN pow_submissions s ON s.id = r.submission_id
        JOIN pow_threads t ON t.id = s.thread_id
        WHERE {is_active_deleted_clause(ignore_deleted, 'r')} AND {is_active_deleted_clause(ignore_deleted, 's')} AND {is_active_deleted_clause(ignore_deleted, 't')}
        ORDER BY s.thread_id, r.createdate
    """
    cur.execute(q)
    rows = cur.fetchall()

    # Problem text by publication
    q_prob = """
        SELECT p.id AS publication_id, z.text AS puzzle_text
        FROM pow_publications p
        LEFT JOIN pow_puzzles z ON p.puzzle = z.id
    """
    cur.execute(q_prob)
    prob_map = {int(pub_id): (ptext if ptext is not None else "") for pub_id, ptext in cur.fetchall() if pub_id is not None}

    replies: List[dict] = []
    mentor_name_map = build_mentor_name_map(conn, disable_cache) 
    
    for r_id, mentor_msg, r_date, s_id, thread_id, s_date, s_short, s_long, pub_id in tqdm(rows, desc="Building reply context", unit="reply", dynamic_ncols=True):
        tid = int(thread_id)
        feat = thread_feat.get(tid, {})
        mid = feat.get("mentor_user_id")
        mname = mentor_name_map.get(mid)
        mlabel = f"{mid} - {mname}" if (mid is not None and mname) else (str(mid) if mid is not None else "NULL")

        replies.append({
            "response_id": int(r_id),
            "thread_id": tid,
            "mentor_user_id": mid,
            "mentor_name": mname,
            "mentor_label": mlabel,
            "response_date": r_date,
            "mentor_message": mentor_msg or "",
            "submission_id": int(s_id),
            "submission_date": s_date,
            "s_short": s_short or "",
            "s_long": s_long or "",
            "publication_id": pub_id,
            "puzzle_text": prob_map.get(pub_id, ""),
            # thread features
            "mentor_reply_count": feat.get("mentor_reply_count", 0),
            "student_revisions": feat.get("student_revisions", 0),
            "delta_total": feat.get("delta_total"),
            "service_name": feat.get("service_name"),
            "teacher_user_id": feat.get("teacher_user_id"),
            "reply_seconds_median": feat.get("reply_seconds_median"),
            "school_name": feat.get("school_name"),
        })
    return replies

def build_ollama_prompt(puzzle_text: str, student_short: str, student_long: str, mentor_message: str) -> str:
    json_keys_str = ",\n  ".join([f'"{k}": 1|2|3' for k in MRUBRIC_CATEGORIES])
    json_keys_str += ',\n  "overall_comments": "very brief reasoned summary (<=30 words)"'
    
    return f"""You are evaluating a math mentoring reply. Read the problem, the student's submission, and the mentor's message.
Score the mentor message on six categories with integers 1–3 (1=weak, 2=adequate, 3=strong). Return STRICT JSON only.  Return a STRICT JSON object with this exact format, including the opening and closing curly braces. Do not omit or add any text outside this object.  

Problem:
{puzzle_text}

Student submission (short answer):
{student_short}

Student submission (long answer):
{student_long}

Mentor message:
{mentor_message}

Return JSON with keys:
{{
  {json_keys_str}
}}

Only return the JSON object; no extra text."""

def call_ollama(
    ollama_url: str,
    model: str,
    prompt: str,
    timeout: int,
    num_thread: Optional[int] = None,
    num_batch: Optional[int] = None,
    num_ctx: Optional[int] = None,
    num_gpu_layers: Optional[int] = None,
    keep_alive: Optional[str] = None,
    log_dir: str = "ollama_logs",
    max_retries: int = 3
) -> Optional[dict]:
    ensure_cache_dir(log_dir)

    url = ollama_url.rstrip("/") + "/api/chat"
    options = {}
    if num_thread is not None:
        options["num_thread"] = int(num_thread)
    if num_batch is not None:
        options["num_batch"] = int(num_batch)
    if num_ctx is not None:
        options["num_ctx"] = int(num_ctx)
    if num_gpu_layers is not None:
        options["gpu_layers"] = int(num_gpu_layers)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": options
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive  # e.g., "15m"

    for attempt in range(1, max_retries + 1):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_filename = os.path.join(log_dir, f"ollama_{timestamp}_{threading.get_ident()}_attempt{attempt}.log")
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("message", {}).get("content") or data.get("response", "")

            with open(log_filename, "w", encoding="utf-8") as log_file:
                log_file.write(f"Prompt:\n{prompt}\n\n")
                log_file.write(f"Response:\n{text}\n")

            result_json = extract_json(text)
            if result_json:
                return result_json
            else:
                safe_print_err("OLLAMA_JSON_PARSE_ERROR", Exception(f"No JSON found. Attempt {attempt}"))
        except Exception as e:
            safe_print_err(f"ollama_call_exception_attempt_{attempt}", e)
            try:
                with open(log_filename, "w", encoding="utf-8") as log_file:
                    log_file.write(f"Prompt:\n{prompt}\n\nException:\n{str(e)}\n")
            except Exception as log_err:
                safe_print_err("log_file_write", log_err)

        if attempt < max_retries:
            sleep_time = 2 ** (attempt - 1)
            tqdm.write(f"[call_ollama] Retry {attempt}/{max_retries} failed. Sleeping {sleep_time}s before next attempt...")
            time.sleep(sleep_time)

    tqdm.write(f"[call_ollama] All {max_retries} attempts failed.")
    return None

def ensure_cache_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        safe_print_err("cache_dir", e)

def cache_path(cache_dir: str, response_id: int) -> str:
    return os.path.join(cache_dir, f"{response_id}.json")

def load_cached(cache_dir: str, response_id: int, ollama_disable_cache: bool = False) -> Optional[dict]:
    if ollama_disable_cache:
        return None
    try:
        p = cache_path(cache_dir, response_id)
        if os.path.exists(p):
            tqdm.write(f"[INFO] Loaded from Ollama cache: {p}")
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    except Exception as e:
        safe_print_err("cache_load", e)
        return None

def save_cached(cache_dir: str, response_id: int, obj: dict, ollama_disable_cache: bool = False) -> None:
    if ollama_disable_cache:
        return
    try:
        p = cache_path(cache_dir, response_id)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        safe_print_err("cache_save", e)

async def call_ollama_async(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    timeout: int,
    log_dir: str,
    semaphore: asyncio.Semaphore,
    ollama_disable_logs: bool = False,
    max_retries: int = 3,
    connect_timeout: int = 10,
    backoff_base: float = 0.75,
    num_thread: Optional[int] = None,
    num_batch: Optional[int] = None,
    num_ctx: Optional[int] = None,
    num_gpu_layers: Optional[int] = None,
    keep_alive: Optional[str] = "15m",
    stream: bool = True,
) -> Optional[dict]:
    """
    Uses streaming by default to avoid long silent read periods that trigger SocketTimeoutError.
    Accumulates streamed JSON chunks from Ollama and reconstructs the final text, then extracts JSON.
    """
    endpoint = url.rstrip("/") + "/api/chat"

    options = {}
    if num_thread is not None:
        options["num_thread"] = int(num_thread)
    if num_batch is not None:
        options["num_batch"] = int(num_batch)
    if num_ctx is not None:
        options["num_ctx"] = int(num_ctx)
    if num_gpu_layers is not None:
        options["gpu_layers"] = int(num_gpu_layers)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": bool(stream),
        "options": options
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    # Read timeout is per "no-bytes" period; keep it reasonably high if not streaming.
    req_timeout = aiohttp.ClientTimeout(
        total=None,
        sock_connect=connect_timeout,
        sock_read=timeout,
    )

    await semaphore.acquire()
    try:
        attempt = 0
        while True:
            attempt += 1
            try:
                async with session.post(endpoint, json=payload, timeout=req_timeout) as resp:
                    if resp.status in (408, 409, 425, 429) or 500 <= resp.status <= 599:
                        body = await resp.text()
                        safe_print_err(
                            f"call_ollama_async_http_{resp.status}",
                            Exception(f"Endpoint={endpoint} Status={resp.status} Body={body[:400]} ...")
                        )
                        if attempt < max_retries:
                            sleep_s = (2 ** (attempt - 1)) * backoff_base + (0.25 * (attempt - 1))
                            await asyncio.sleep(sleep_s)
                            continue
                        resp.raise_for_status()

                    resp.raise_for_status()

                    if stream:
                        # Ollama stream returns JSON lines with token deltas.
                        # We reconstruct final text from either message.content or response field.
                        text_parts: List[str] = []
                        async for raw in resp.content:
                            try:
                                line = raw.decode("utf-8", errors="ignore").strip()
                                if not line:
                                    continue
                                # Some builds prefix with "data: {...}"
                                if line.startswith("data:"):
                                    line = line[5:].strip()
                                # Not every line is JSON; guard parse.
                                if line and line[0] == "{":
                                    chunk = json.loads(line)
                                    seg = (chunk.get("message", {}) or {}).get("content")
                                    if not seg:
                                        seg = chunk.get("response", "")
                                    if seg:
                                        text_parts.append(seg)
                            except Exception as pe:
                                # Non-fatal; keep reading. Log parse issues.
                                safe_print_err("ollama_stream_parse", pe)

                        text = "".join(text_parts)
                    else:
                        # Non-stream: entire JSON at end
                        data = await resp.json()
                        text = (data.get("message", {}) or {}).get("content") or data.get("response", "") or ""

                    result_json = extract_json(text)

                    if not ollama_disable_logs:
                        try:
                            os.makedirs(log_dir, exist_ok=True)
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            fn = os.path.join(
                                log_dir,
                                f"ollama_async_{timestamp}_{hashlib.md5(prompt.encode()).hexdigest()}_attempt{attempt}.log"
                            )
                            with open(fn, "w", encoding="utf-8") as f:
                                f.write(f"Endpoint: {endpoint}\nModel: {model}\nOptions: {options}\nStream: {stream}\n")
                                f.write(f"Prompt:\n{prompt}\n\nResponse text ({len(text)} chars):\n{text}\n")
                        except Exception as log_e:
                            safe_print_err("call_ollama_async_log", log_e)

                    if result_json:
                        return result_json

                    # If we got here, we failed to parse JSON. Retry a couple of times.
                    safe_print_err("OLLAMA_JSON_PARSE_ERROR", Exception("No JSON in response"))
                    if attempt < max_retries:
                        sleep_s = (2 ** (attempt - 1)) * backoff_base
                        await asyncio.sleep(sleep_s)
                        continue
                    return None

            except (asyncio.TimeoutError, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, aiohttp.ServerTimeoutError, aiohttp.ClientOSError) as e:
                # Explicitly include SocketTimeoutError subclasses
                safe_print_err("call_ollama_async_timeout_or_conn", e)
                if attempt < max_retries:
                    sleep_s = (2 ** (attempt - 1)) * backoff_base
                    await asyncio.sleep(sleep_s)
                    continue
                return None
            except Exception as e:
                safe_print_err("call_ollama_async_unexpected", e)
                return None
    finally:
        semaphore.release()

async def score_replies_with_mrubric_async(
    replies: List[dict],
    ollama_urls: List[str],
    model: str,
    timeout: int,
    max_conc: int,
    cache_dir: str,
    log_dir: str = "ollama_logs",
    ollama_disable_cache: bool = False,
    ollama_disable_logs: bool = False    
) -> List[dict]:
    if not ollama_disable_cache:
        ensure_cache_dir(cache_dir)
    if not ollama_disable_logs:
        ensure_cache_dir(log_dir)

    results: List[dict] = []
    semaphore = asyncio.Semaphore(max_conc)
    tasks = []

    pbar = tqdm(total=len(replies), desc="Scoring replies", unit="reply", dynamic_ncols=True)

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=2 * len(ollama_urls), ttl_dns_cache=300)

    async with aiohttp.ClientSession(connector=connector) as session:
        for idx, r in enumerate(replies):
            rid = r["response_id"]
            cached = load_cached(cache_dir, rid, ollama_disable_cache=ollama_disable_cache)
            if cached:
                row = {
                    "response_id": rid,
                    "thread_id": r["thread_id"],
                    "mentor_user_id": r.get("mentor_user_id"),
                    "mentor_name": r.get("mentor_name"),
                    "mentor_label": r.get("mentor_label"),
                    "response_date": r.get("response_date"),
                    "mentor_reply_count": r.get("mentor_reply_count", 0),
                    "student_revisions": r.get("student_revisions", 0),
                    "delta_total": r.get("delta_total"),
                    "service_name": r.get("service_name"),
                    "teacher_user_id": r.get("teacher_user_id"),
                    "reply_seconds_median": r.get("reply_seconds_median"),
                    "school_name": r.get("school_name")
                }
                for k in MRUBRIC_CATEGORIES:
                    row[k] = cached.get(k)
                row["overall_comments"] = cached.get("overall_comments")
                results.append(row)
                pbar.update(1)
                continue

            prompt = build_ollama_prompt(
                r.get("puzzle_text", ""),
                r.get("s_short", ""),
                r.get("s_long", ""),
                r.get("mentor_message", "")
            )

            async def process(
                r=r,
                rid=rid,
                prompt=prompt,
                url=ollama_urls[idx % len(ollama_urls)]
            ):
                try:
                    scored = await call_ollama_async(
                        session=session,
                        url=url,
                        model=model,
                        prompt=prompt,
                        timeout=timeout,
                        log_dir=log_dir,
                        semaphore=semaphore,
                        ollama_disable_logs=ollama_disable_logs,
                        max_retries=3,
                        connect_timeout=10,
                        backoff_base=0.75,
                        num_thread=args.ollama_num_threads if 'args' in globals() else None,
                        num_batch=args.ollama_num_batch if 'args' in globals() else None,
                        num_ctx=args.ollama_num_ctx if 'args' in globals() else None,
                        num_gpu_layers=args.ollama_gpu_layers if 'args' in globals() else None,
                        keep_alive=args.ollama_keepalive if 'args' in globals() else "15m",
                        stream=True,  
                    )
                    if not scored:
                        return  # skip on error

                    save_cached(cache_dir, rid, scored, ollama_disable_cache=ollama_disable_cache)
                    row = {
                        "response_id": rid,
                        "thread_id": r["thread_id"],
                        "mentor_user_id": r.get("mentor_user_id"),
                        "mentor_name": r.get("mentor_name"),
                        "mentor_label": r.get("mentor_label"),
                        "response_date": r.get("response_date"),
                        "mentor_reply_count": r.get("mentor_reply_count", 0),
                        "student_revisions": r.get("student_revisions", 0),
                        "delta_total": r.get("delta_total"),
                        "service_name": r.get("service_name"),
                        "teacher_user_id": r.get("teacher_user_id"),
                        "reply_seconds_median": r.get("reply_seconds_median"),
                        "school_name": r.get("school_name"),
                    }
                    for k in MRUBRIC_CATEGORIES:
                        try:
                            row[k] = int(scored.get(k)) if scored.get(k) is not None else None
                        except Exception:
                            row[k] = None
                    row["overall_comments"] = scored.get("overall_comments")
                    results.append(row)
                finally:
                    pbar.update(1)

            tasks.append(process())

        await asyncio.gather(*tasks)
        pbar.close()
        return results
        
# ------------------------------
# Distributions & Crossed Subsets
# ------------------------------

def write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def dist_by_bucket(mrubric_rows: List[dict], bucket_key: str, within_threads_only: bool = False) -> List[dict]:
    """
    Compute, for each bucket (e.g., mentor_reply_count), the distribution of mrubric scores by category and level.
    Returns rows with: subset, bucket_value, category, score_level, count, percent
    within_threads_only: if True, ignore replies from threads that do not meet "has mentoring" criteria (i.e., bucket may imply).
    """
    # Filter if requested: "threads with mentoring" => mentor_reply_count >= 1
    data = mrubric_rows
    if within_threads_only and bucket_key != "mentor_reply_count":
        data = [r for r in data if (r.get("mentor_reply_count") or 0) >= 1]

    # Build nested counters
    nested: Dict[Any, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    totals: Dict[Any, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in tqdm(data, desc=f"Bucket={bucket_key}", unit="reply", dynamic_ncols=True):
        bval = r.get(bucket_key)
        if bval is None:
            bval = "NULL"
        if bucket_key == "delta_total" and bval is not None and bval != "NULL":
            try:
                bval = int(round(float(bval)))
            except Exception:
                bval = "NULL"
        for cat in MRUBRIC_CATEGORIES:
            lvl = r.get(cat)
            if lvl is None:
                continue
            nested[bval][cat][int(lvl)] += 1
            totals[bval][cat] += 1

    rows: List[dict] = []
    for bval, cat_map in nested.items():
        for cat, counter in cat_map.items():
            total = totals[bval][cat]
            for lvl in sorted(counter.keys()):
                cnt = counter[lvl]
                rows.append({
                    "subset": bucket_key,
                    "bucket_value": bval,
                    "category": cat,
                    "score_level": int(lvl),
                    "count": int(cnt),
                    "percent": pct(cnt, total),
                })
    return rows

def crossed_dist(mrubric_rows: List[dict], bucket_key: str, factor_key: str, within_threads_only: bool = False) -> List[dict]:
    """
    Crossed distributions: for each factor value (service/mentor/teacher/reply_time_bin),
    compute mrubric distributions by the bucket.
    Returns rows with: subset, bucket_value, factor, factor_value, category, score_level, count, percent
    """
    # Build a reply_time_bin if needed
    rows = mrubric_rows
    if factor_key == "reply_time_bin":
        tmp = []
        for r in tqdm(rows, desc=f"Processing crossed dist: {bucket_key} x {factor_key}", unit="reply", dynamic_ncols=True):
            v = r.get("reply_seconds_median")
            if v is None:
                binv = "NULL"
            else:
                binv = int(round(float(v) / 3600.0))  # hours
            rr = dict(r)
            rr["reply_time_bin"] = binv
            tmp.append(rr)
        rows = tmp

    if within_threads_only and bucket_key != "mentor_reply_count":
        rows = [r for r in rows if (r.get("mentor_reply_count") or 0) >= 1]

    nested: Dict[Any, Dict[Any, Dict[str, Counter]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    totals: Dict[Any, Dict[Any, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for r in rows:
        bval = r.get(bucket_key)
        if bval is None:
            bval = "NULL"
        if bucket_key == "delta_total" and bval is not None and bval != "NULL":
            try:
                bval = int(round(float(bval)))
            except Exception:
                bval = "NULL"

        fval = r.get(factor_key)
        if fval is None or (isinstance(fval, str) and fval.strip() == ""):
            fval = "NULL"
        for cat in MRUBRIC_CATEGORIES:
            lvl = r.get(cat)
            if lvl is None:
                continue
            nested[bval][fval][cat][int(lvl)] += 1
            totals[bval][fval][cat] += 1

    out: List[dict] = []
    for bval, f_map in nested.items():
        for fval, cat_map in f_map.items():
            for cat, counter in cat_map.items():
                total = totals[bval][fval][cat]
                for lvl in sorted(counter.keys()):
                    cnt = counter[lvl]
                    out.append({
                        "subset": bucket_key,
                        "bucket_value": bval,
                        "factor": factor_key,
                        "factor_value": fval,
                        "category": cat,
                        "score_level": int(lvl),
                        "count": int(cnt),
                        "percent": pct(cnt, total),
                    })
    return out

# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate PoW analytics CSVs with mrubric via Ollama.")
    parser.add_argument("--db", default="mathforum.db", help="Path to mathforum.db")
    parser.add_argument("--out", default="pow_report.csv", help="Original report CSV path (A–F)")

    # NEW: control ZIPs and thread document dirs
    parser.add_argument("--zip-output-dir", default="report_thread_zips", help="Directory to write per-row ZIP bundles")
    parser.add_argument("--thread-export-dir", default=THREAD_EXPORT_DIR_DEFAULT, help="Directory to write thread_{id}.txt/.json")
    parser.add_argument("--disable-row-zips", action="store_true", help="Disable creation of per-row ZIPs during base report")
    parser.add_argument("--thread-problem-outputs-dir", default="text_problem_outputs",
                        help="If present, recursively reuse existing thread_{id}.txt/.json from this directory instead of querying the DB")
    parser.add_argument("--max-zip-workers", type=int, default=32,
                        help="Maximum number of worker threads for building ZIP files (default: 32)")
    parser.add_argument("--zip-compression-level", type=int, default=6,
                        help="ZIP compression level (0–9, default: 6)")
                        
    parser.add_argument("--disable-mrubric", dest="disable_mrubric", action="store_true", help="Disable mrubric scoring via Ollama")
    parser.add_argument("--ignore-deleted", dest="ignore_deleted", action="store_true", help="Remove deleted records from the dataset")
    parser.add_argument("--ollama-urls",
                        type=str,
                        default="http://localhost:11434",
                        help="Comma-separated list of Ollama URLs (e.g., http://localhost:11434,http://localhost:11435)"
                    )
    parser.add_argument("--ollama-model", default="llama3", help="Ollama model name/tag")
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout seconds for Ollama requests")
    parser.add_argument("--max-concurrency", type=int, default=8, help="Max concurrent Ollama requests")
    parser.add_argument("--ollama-num-threads", type=int, default=16)
    parser.add_argument("--ollama-num-batch", type=int, default=64)
    parser.add_argument("--ollama-num-ctx", type=int, default=8192)
    parser.add_argument("--ollama-gpu-layers", type=int, default=20)
    parser.add_argument("--ollama-keepalive", default="15m",
                    help='Keep model loaded between requests, e.g., "15m" or "forever"')
    parser.add_argument("--ollama-disable-cache", action="store_true", help="Disable Ollama reply cache (forces re-scoring)")
    parser.add_argument("--ollama-disable-logs", action="store_true", help="Disable Ollama logging of prompts and responses")
    parser.add_argument("--pickle-disable-cache", action="store_true", help="Disable calls to save_caches and load_caches")
    parser.add_argument("--cache-dir", default="mrubric_cache", help="Directory for per-reply JSON cache")
    parser.add_argument("--scores-csv", default="pow_mrubric_scores.csv", help="Per-reply mrubric scores CSV")
    parser.add_argument("--by-mentor-cat-csv", default="pow_mrubric_category_by_mentor.csv",
                        help="mrubric category distribution grouped by mentor (mentor_label)")
    parser.add_argument("--by-school-cat-csv", default="pow_mrubric_category_by_school.csv",
                        help="mrubric category distribution grouped by school (school_name)")
    parser.add_argument("--by-replies-csv", default="pow_mrubric_by_replies.csv", help="Distribution by # mentor replies")
    parser.add_argument("--by-revisions-csv", default="pow_mrubric_by_revisions.csv", help="Distribution by # student revisions (mentored only)")
    parser.add_argument("--by-improvement-csv", default="pow_mrubric_by_improvement.csv", help="Distribution by Δ prubric total (mentored only)")
    parser.add_argument("--by-replies-cross-csv", default="pow_mrubric_by_replies_cross.csv", help="Crossed distributions for replies bucket")
    parser.add_argument("--by-revisions-cross-csv", default="pow_mrubric_by_revisions_cross.csv", help="Crossed distributions for revisions bucket")
    parser.add_argument("--by-improvement-cross-csv", default="pow_mrubric_by_improvement_cross.csv", help="Crossed distributions for improvement bucket")
    global args
    args = parser.parse_args()

    load_caches(disable_cache=args.pickle_disable_cache)

    ollama_urls = [url.strip() for url in args.ollama_urls.split(",") if url.strip()]

    start_time = time.time()
    tqdm.write(f"[INFO] Connecting to database {args.db} ...")

    # Connect & basic pragmas
    try:
        conn = sqlite3.connect(args.db)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        for pragma in [
            "PRAGMA journal_mode=OFF;",
            "PRAGMA synchronous=OFF;",
            "PRAGMA temp_store=MEMORY;",
            "PRAGMA cache_size=100000;"
        ]:
            cur.execute(pragma)
    except Exception as e:
        safe_print_err("connect", e)
        sys.exit(1)

    tqdm.write(f"[INFO] Database connected, pragmas set. ({time.time() - start_time:.2f}s)")

    # 1) Base report (A–F) + per-row ZIPs
    try:
        if os.path.exists(args.out):
            tqdm.write(f"[INFO] Base report {args.out} already exists — skipping.")
        else:
            base_report(conn, args.out, args.ignore_deleted)
            tqdm.write(f"[info] Wrote base report to {args.out}")
    except Exception as e:
        safe_print_err("base_report", e)
        sys.exit(2)

    # 2) mrubric scoring (if enabled) — unchanged from your script
    if not args.disable_mrubric:
        try:
            replies = fetch_replies_with_context(conn, args.ignore_deleted, args.pickle_disable_cache)
            scored_rows = asyncio.run(score_replies_with_mrubric_async(
                replies=replies,
                ollama_urls=ollama_urls,
                model=args.ollama_model,
                timeout=args.timeout,
                max_conc=args.max_concurrency,
                cache_dir=args.cache_dir,
                log_dir="ollama_logs",
                ollama_disable_cache=args.ollama_disable_cache,
                ollama_disable_logs=args.ollama_disable_logs
            ))

            rows_by_mentor_cat = mrubric_category_breakdown_by(scored_rows, "mentor_label")
            rows_by_school_cat = mrubric_category_breakdown_by(scored_rows, "school_name")

            cat_fields = ["group_key", "group_value", "category", "score_level", "count", "percent"]
            write_csv(args.by_mentor_cat_csv, rows_by_mentor_cat, cat_fields)
            write_csv(args.by_school_cat_csv, rows_by_school_cat, cat_fields)
            tqdm.write(f"[info] Wrote mrubric category breakdowns to {args.by_mentor_cat_csv}, {args.by_school_cat_csv}")

            score_fields = [
                "response_id","thread_id",
                "mentor_user_id","mentor_name","mentor_label",
                "response_date",
                "mentor_reply_count","student_revisions",
                "delta_total","service_name","school_name",
                "teacher_user_id", "reply_seconds_median"
            ] + MRUBRIC_CATEGORIES + ["overall_comments"]
            write_csv(args.scores_csv, scored_rows, score_fields)
            tqdm.write(f"[info] Wrote mrubric scores to {args.scores_csv}")

            rows_by_replies   = dist_by_bucket(scored_rows, "mentor_reply_count", within_threads_only=False)
            rows_by_revisions = dist_by_bucket(scored_rows, "student_revisions", within_threads_only=True)
            rows_by_improve   = dist_by_bucket(scored_rows, "delta_total", within_threads_only=True)

            dist_fields = ["subset","bucket_value","category","score_level","count","percent"]
            write_csv(args.by_replies_csv, rows_by_replies, dist_fields)
            write_csv(args.by_revisions_csv, rows_by_revisions, dist_fields)
            write_csv(args.by_improvement_csv, rows_by_improve, dist_fields)
            tqdm.write(f"[info] Wrote mrubric distributions to {args.by_replies_csv}, {args.by_revisions_csv}, {args.by_improvement_csv}")

            factors = ["service_name","school_name","mentor_label","reply_time_bin"]
            rows_cross = []
            rows_cross_rev = []
            rows_cross_imp = []
            for f in factors:
                rows_cross.extend(crossed_dist(scored_rows, "mentor_reply_count", f, within_threads_only=False))
                rows_cross_rev.extend(crossed_dist(scored_rows, "student_revisions", f, within_threads_only=True))
                rows_cross_imp.extend(crossed_dist(scored_rows, "delta_total", f, within_threads_only=True))

            cross_fields = ["subset","bucket_value","factor","factor_value","category","score_level","count","percent"]
            write_csv(args.by_replies_cross_csv, rows_cross, cross_fields)
            write_csv(args.by_revisions_cross_csv, rows_cross_rev, cross_fields)
            write_csv(args.by_improvement_cross_csv, rows_cross_imp, cross_fields)
            tqdm.write(f"[info] Wrote mrubric crossed distributions to {args.by_replies_cross_csv}, {args.by_revisions_cross_csv}, {args.by_improvement_cross_csv}")
        except Exception as e:
            safe_print_err("mrubric", e)
            sys.exit(3)
    else:
        tqdm.write("[info] mrubric disabled; skipping Ollama evaluation.")

    tqdm.write(f"[INFO] All processing complete. Total runtime: {time.time() - start_time:.1f}s")
    save_caches(disable_cache=args.pickle_disable_cache)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        tqdm.write("[main] Interrupted by user (Ctrl+C)")
    except Exception as e:
        safe_print_err("main", e)
        sys.exit(10)