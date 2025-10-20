import sqlite3
import os
import re
import html
import json
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed, FIRST_COMPLETED
from tqdm import tqdm
import traceback

# Configurations
db_file = 'mathforum.db'
output_dir = 'thread_outputs'
os.makedirs(output_dir, exist_ok=True)

# Precompiled regex
_illegal_characters_re = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

# Progress bar
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

def write_thread_output(thread_id, puzzle_text, conversation, student_metadata=None, mentor_name=""):
    global progress
    
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
        
        if not conversation:
            print(f"Thread {thread_id} has no conversation data.")
            return
        
        if student_metadata:
            json_output.update(student_metadata)            
            
        for idx, row in enumerate(conversation):
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
            r_id = row.get("response_id", "")
            r_date = row.get("response_date", "")

            # Rubric fields (robust handling)
            rubric_fields = {}
            for rubric_key in ["strategy", "interpretation", "accuracy", "completeness", "clarity", "reflection"]:
                raw_key = f"rubric_{rubric_key}"
                rubric_fields[rubric_key] = row.get(raw_key, 0)

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

            if any(v is not None for v in rubric_fields.values()):
                rubric_texts = [f"{k.capitalize()}: {v}" for k, v in rubric_fields.items() if v is not None]
                rubric_line = f"Rubric {idx+1}: " + "; ".join(rubric_texts)
                text_lines.append(rubric_line)
                text_lines.append("")  # Blank line after rubric

            json_output["conversation"].append({
                "submission_id": s_id,
                "submission_date": f"{s_date}",
                "short_answer": s_short,
                "long_answer": s_long,
                "response_id": r_id,
                "response": r_msg,
                "response_date": f"{r_date}",
                "rubrics": rubric_fields,
                "images": image_blobs
            })
        with open(os.path.join(output_dir, f"thread_{thread_id}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(text_lines))

        with open(os.path.join(output_dir, f"thread_{thread_id}.json"), "w", encoding="utf-8") as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)  
    except Exception as e:
        print(f"Error writing thread {thread_id}: {e}")            
        traceback.print_exc()
    
    if progress:
        progress.update(1)  
        if progress.n % 100 == 0:
            gc.collect()

threads_query = """
SELECT 
    t.id AS thread_id, 
    z.text AS puzzle_text,
    du.first_name || ' ' || du.last_name AS mentor_name
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

def stream_rows(cursor, query, params=(), batch=1000):
    cursor.execute(query, params)
    while True:
        rows = cursor.fetchmany(batch)
        if not rows:
            break
        for row in rows:
            yield dict(row)

def get_student_school(cursor, conversation):
    student_name = ""
    school_name = ""
    age = ""

    if not conversation:
        return {
            "student_name": student_name,
            "school_name": school_name,
            "age": age
        }

    submission_ids = [row.get("submission_id") for row in conversation if row.get("submission_id")]
    if not submission_ids:
        return {
            "student_name": student_name,
            "school_name": school_name,
            "age": age
        }

    creator_id = None
    for sid in submission_ids:
        cursor.execute("SELECT creator FROM pow_submissions WHERE id = ?", (sid,))
        row = cursor.fetchone()
        if row and row["creator"]:
            creator_id = row["creator"]
            break

    if not creator_id:
        return {
            "student_name": student_name,
            "school_name": school_name,
            "age": age
        }

    # Get student info
    cursor.execute("SELECT first_name, last_name, ageinyears FROM dir_users WHERE id = ?", (creator_id,))
    user_row = cursor.fetchone()
    if user_row:
        first = user_row["first_name"] or ""
        last = user_row["last_name"] or ""
        student_name = f"{first} {last}".strip()
        age = str(user_row["ageinyears"]) if user_row["ageinyears"] is not None else ""

    # Get school name
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

    return {
        "student_name": student_name,
        "school_name": school_name,
        "age": age
    }
    
def process_thread(thread, db_file):
    try:
        if not thread or not thread.get("thread_id"):
            print("Skipping invalid or empty thread record")
            return
        
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        for pragma in [
            "PRAGMA journal_mode = OFF;",
            "PRAGMA synchronous = OFF;",
            "PRAGMA temp_store = MEMORY;",
            "PRAGMA cache_size = 10000;"
        ]:
            cursor.execute(pragma)

        thread_id = thread.get("thread_id")
        if thread_id is None:
            print("Skipping thread with missing thread_id")
            return
            
        conversation = list(stream_rows(cursor, submissions_query, (thread_id,)))
        puzzle_text = (thread.get("puzzle_text") or "").strip()
        student_metadata = get_student_school(cursor, conversation)
        mentor_name = thread.get("mentor_name", "")
        write_thread_output(thread_id, puzzle_text, conversation, student_metadata, mentor_name)        
        
        conn.close()
    except Exception as e:
        print(f"Error processing thread {thread.get('thread_id', '?')}: {e}")
        traceback.print_exc()

def main():
    global progress
    
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    for pragma in [
        "PRAGMA journal_mode = OFF;",
        "PRAGMA synchronous = OFF;",
        "PRAGMA temp_store = MEMORY;",
        "PRAGMA cache_size = 10000;"
    ]:
        cursor.execute(pragma)

    cursor.execute("SELECT COUNT(*) FROM pow_threads")
    total_threads = cursor.fetchone()[0]

    threads = [thread for thread in stream_rows(cursor, threads_query) if thread and thread.get("thread_id")]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_thread, thread, db_file): thread for thread in threads}
        
        progress = tqdm(total=len(futures), desc="Threads processed", unit="thread")

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                thread = futures.pop(future)
                try:
                    future.result()
                except Exception as e:
                    thread_id = thread.get('thread_id', '?')
                    print(f"Unhandled error in thread {thread_id}: {e}", flush=True)
                    traceback.print_exc()

    conn.close()
    print(f"\nCompleted processing {total_threads} threads.")

if __name__ == '__main__':
    main()