import asyncpg
import asyncio
import os
import re
import json
import html
import time
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import traceback

# Configurations
output_dir = 'problem_outputs'
os.makedirs(output_dir, exist_ok=True)

postgres_config = {
    'user': 'postgres',
    'password': 'postgres',
    'database': 'mathforum',
    'host': 'localhost',
    'port': 5432,
}

_illegal_characters_re = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

progress = None

def extract_base64_placeholders(field_label, text):
    if not isinstance(text, str):
        return "", []
        
    pattern = re.compile(r'<[^>]+?base64,([^"\'>\s]+)[^>]*>', re.IGNORECASE)
    b64_list = []
    placeholders = []

    def repl(match):
        idx = len(b64_list) + 1
        b64_data = match.group(1)
        placeholder = f"[{field_label} Image {idx}]"
        b64_list.append({"field": field_label, "index": idx, "base64": b64_data})
        placeholders.append(placeholder)
        return placeholder

    new_text = pattern.sub(repl, text)
    return new_text, b64_list
    
def clean_illegal_chars(value):
    if isinstance(value, str):
        return _illegal_characters_re.sub("", value)
    return value

def sanitize_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r'[\r\n\t]', ' ', text)
    text = re.sub(r'\\[rnt]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def strip_html(value):
    if not isinstance(value, str):
        return ""
    text = re.sub(r'<[^>]+>', '', value)
    return html.unescape(text.strip())

def write_thread_output(thread_id, puzzle_text, all_rows, output_dir, student_metadata=None, mentor_name=""):
    try:
        plain_puzzle_text = sanitize_text(strip_html(puzzle_text or ""))
        text_lines = [f"Problem statement: {plain_puzzle_text}", ""]
        
        if student_metadata:
            if student_metadata.get("student_name"):
                text_lines.append(f"Student Name: {student_metadata['student_name']}")
            if student_metadata.get("age"):
                text_lines.append(f"Age: {student_metadata['age']}")
            if student_metadata.get("school_name"):
                text_lines.append(f"School: {student_metadata['school_name']}")
            text_lines.append("")
            
        if mentor_name and len(mentor_name) > 0:
            text_lines.append(f"Mentor Name: {mentor_name}")
            text_lines.append("")
        
        json_output = {"thread_id": thread_id, "puzzle_text": plain_puzzle_text, "conversation": [],
            "student_name": "",
            "school_name": "",
            "age": "",
            "mentor_name": mentor_name
        }
        
        if student_metadata:
            json_output.update(student_metadata)

        if not all_rows:
            print(f"Thread {thread_id} has no conversation data.")
            return

        for idx, row in enumerate(sorted(all_rows, key=lambda x: x.get("submission_id", 0))):
            image_blobs = []

            s_short_raw = row.get("s_shortanswer", "")
            s_long_raw = row.get("s_longanswer", "")
            r_msg_raw = row.get("r_message", "")

            s_short_text, imgs1 = extract_base64_placeholders("Short Answer", s_short_raw)
            s_long_text, imgs2 = extract_base64_placeholders("Long Answer", s_long_raw)
            r_msg_text, imgs3 = extract_base64_placeholders("Mentor Message", r_msg_raw)

            image_blobs.extend(imgs1 + imgs2 + imgs3)

            s_short = sanitize_text(strip_html(s_short_text))
            s_long = sanitize_text(strip_html(s_long_text))
            r_msg = sanitize_text(strip_html(r_msg_text))
            
            s_id = row.get("submission_id", "")
            s_date = row.get("submission_date", "")
            r_id   = row.get("response_id", None)
            r_date = row.get("response_date", "")

            rubric_fields = ["rb_strategy", "rb_interpretation", "rb_completeness", "rb_clarity", "rb_reflection", "rb_accuracy"]
            rubrics = {k.replace("rb_", ""): row.get(k) for k in rubric_fields}
            rubrics_clean = {k: v for k, v in rubrics.items() if v is not None}

            s_date_text = f"(submitted on {s_date}) " if s_date else ""
            s_text = f"Student Submission {s_date_text}Short Answer {idx+1}: {s_short}"
            text_lines.append(s_text)
            text_lines.append("")

            l_text = f"Student Submission {s_date_text}Long Answer {idx+1}: {s_long}"
            text_lines.append(l_text)
            text_lines.append("")

            r_date_text = f"(responded on {r_date}) " if r_date else ""
            r_text = f"Mentor Response {r_date_text}{idx+1}: {r_msg}" if r_msg else "Mentor Response: (No reply yet)"
            text_lines.append(r_text)
            text_lines.append("")
            
            if image_blobs:
                for img in image_blobs:
                    text_lines.append(f"Embedded image: {img['field']} Image {img['index']} {img['base64']}")
                text_lines.append("")            
            
            if rubrics_clean:
                rubric_texts = [f"{k.capitalize()}: {v}" for k, v in rubrics_clean.items()]
                rubric_line = f"Rubric {idx+1}: " + "; ".join(rubric_texts)
                text_lines.append(rubric_line)
                text_lines.append("")

            json_output["conversation"].append({
                "submission_id": s_id,
                "submission_date": f"{s_date}",
                "short_answer": s_short,
                "long_answer": s_long,
                "response": r_msg,
                "response_id": r_id,
                "response_date":f"{r_date}",
                "rubrics": rubrics,
                "images": image_blobs
            })

        text_path = os.path.join(output_dir, f"thread_{thread_id}.txt")
        json_path = os.path.join(output_dir, f"thread_{thread_id}.json")

        with open(text_path, "w", encoding="utf-8") as f:
            f.write("\n".join(text_lines))

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"Error writing thread {thread_id}: {e}")
        traceback.print_exc()

threads_query = """
SELECT t.id AS thread_id, z.text AS puzzle_text, du.id AS mentor_id, du.first_name || ' ' || du.last_name AS mentor_name
FROM pow_threads t
LEFT JOIN pow_publications p ON t.publication = p.id
LEFT JOIN pow_puzzles z ON p.puzzle = z.id
LEFT JOIN dir_users du ON t.mentor = du.id
"""

submissions_query = """
SELECT
    s.id AS submission_id,
    s.thread_id AS s_thread_id,
    s.shortanswer AS s_shortanswer,
    s.longanswer AS s_longanswer,
    s.createdate AS submission_date,
    r.id AS response_id,
    r.message AS r_message,
    r.createdate AS response_date,
    rb.strategy AS rb_strategy,
    rb.interpretation AS rb_interpretation,
    rb.completeness AS rb_completeness,
    rb.clarity AS rb_clarity,
    rb.reflection AS rb_reflection,
    rb.accuracy AS rb_accuracy
FROM pow_submissions s
LEFT JOIN pow_responses r ON r.submission_id = s.id
LEFT JOIN pow_rubric rb ON r.rubric_id = rb.id
WHERE s.thread_id = $1
ORDER BY s.createdate, r.createdate
"""

async def fetch_all_as_dict(conn, query, params=()):
    async with conn.transaction():
        cursor = await conn.cursor(query, *params)
        while True:
            rows = await cursor.fetch(100)
            if not rows:
                break
            for row in rows:
                yield dict(row)

async def build_global_metadata_maps(conn):
    # Main data query
    query = """
        SELECT
            s.id AS submission_id,
            stu.first_name || ' ' || stu.last_name AS student_name,
            stu.ageinyears AS age,
            g.name AS school_name
        FROM pow_submissions s
        LEFT JOIN dir_users stu ON s.creator = stu.id
        LEFT JOIN dir_memberships m ON m.user_id = s.creator
        LEFT JOIN dir_groups g ON m.group_id = g.id
    """

    # Accurate count for progress bar based on the exact query
    count_query = f"SELECT COUNT(*) FROM ({query}) AS subq"

    total_rows = await conn.fetchval(count_query)

    metadata_map = {}

    async with conn.transaction():
        cursor = await conn.cursor(query)
        with tqdm(total=total_rows, desc="Mapping student metadata", unit="submission") as pbar:
            while True:
                batch = await cursor.fetch(100)
                if not batch:
                    break
                for r in batch:
                    metadata_map[r['submission_id']] = {
                        'student_name': (r['student_name'] or "").strip(),
                        'age': str(r['age']) if r['age'] is not None else '',
                        'school_name': r['school_name'] or '',
                    }
                    pbar.update(1)

    return metadata_map

async def fetch_and_write_submissions(pool, thread, executor, student_metadata_map):
    async with pool.acquire() as conn:
        rows_iter = fetch_all_as_dict(conn, submissions_query, (thread["thread_id"],))
        all_rows = []
        async for row in rows_iter:
            row['mentor_id'] = thread.get('mentor_id')
            row['mentor_name'] = thread.get('mentor_name')
            all_rows.append(row)

        puzzle_text = (thread.get("puzzle_text") or "").strip()
        
        first_sid = all_rows[0]['submission_id'] if all_rows else None
        student_metadata = student_metadata_map.get(first_sid, {})
        mentor_name = thread.get("mentor_name", "")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor,
            partial(write_thread_output, thread["thread_id"], puzzle_text, all_rows, output_dir, student_metadata, mentor_name)
        )

async def main():
    global progress

    pool = await asyncpg.create_pool(**postgres_config)
    executor = ThreadPoolExecutor(max_workers=(os.cpu_count() or 4) * 2)
    sem = asyncio.Semaphore(8)

    async with pool.acquire() as conn:
        student_metadata_map = await build_global_metadata_maps(conn)
        total_result = await conn.fetchval("SELECT COUNT(*) FROM pow_threads")
        threads = await conn.fetch(threads_query)

    progress = tqdm(total=total_result, desc="Threads completed", unit="thread")

    async def process_thread(thread):
        async with sem:
            try:
                await fetch_and_write_submissions(pool, dict(thread), executor, student_metadata_map)
            except Exception as e:
                print(f"Error processing thread {thread.get('thread_id', 'unknown')}: {e}")
                traceback.print_exc()
            finally:
                if progress:
                    progress.update(1)

    await asyncio.gather(*(process_thread(t) for t in threads))

    await pool.close()
    executor.shutdown()
    progress.close()

if __name__ == '__main__':
    asyncio.run(main())