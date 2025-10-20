import sqlite3
import math
import re
import html
import pickle
import os
import json
import unicodedata
import string
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from statistics import median
from flask import Flask, render_template_string, request, abort, url_for
from tqdm import tqdm

app = Flask(__name__)
DB_PATH = "mathforum.db"
FEEDBACK_DB = "feedback.db"  
THREADS_PER_PAGE = 50
MEM_PICKLE = "mathforum_web_viewer_cache.pkl"
THREADS_PICKLE = "mathforum_web_viewer_threads.pkl"
STUDENTS_PICKLE = "mathforum_web_viewer_students.pkl"
SCHOOLS_PICKLE = "mathforum_web_viewer_schools.pkl"
SUBMISSIONS_PICKLE = "mathforum_web_viewer_submissions.pkl"

# In-memory storage
THREADS = {}  # thread_id -> {'puzzle_text': str, 'submissions': [list of dicts]}
THREAD_INDEX = []  # list of thread_id for pagination/searching
THREAD_METRICS = {}
FEEDBACK_BY_THREAD = {}  # thread_id → list of feedback entries

# -- Templates --
INDEX_TEMPLATE = """
<!doctype html>
<title>Math Forum Threads</title>
<h1>Math Forum Threads</h1>

<form method="get">
  <p><input type="submit" value="Apply Filters"></p>
  <table border="1" cellspacing="0" cellpadding="4">
    <thead>
      <tr>
        <th>Thread ID</th>
        <th>Problem Title</th>
        <th>School</th>
        <th>Mentor</th>        
        <th># Student Submissions</th>
        <th># Mentor Responses</th>
        <th># Rubrics</th>
        <th>Median Short Answer Length</th>
        <th>Median Long Answer Length</th>
        <th>Median Mentor Response Length</th>
        <th>Has Images</th>
        <th>Leave Feedback</th>
        <th>View Feedback</th>
      </tr>
      <tr>
        <!-- Search Inputs -->
        <td><input type="text" name="thread_id" value="{{ form.thread_id }}" style="width: 90%;"></td>
        <td><input type="text" name="puzzle_text" value="{{ form.puzzle_text }}" style="width: 90%;"></td>
        <td><input type="text" name="school_name" value="{{ form.school_name }}" style="width: 90%;"></td>
        <td><input type="text" name="mentor_name" value="{{ form.mentor_name }}" style="width: 90%;"></td>        
        <td><input type="number" name="num_subs" value="{{ form.num_subs }}" min="0" style="width: 90%;"></td>
        <td><input type="number" name="num_mentor_msgs" value="{{ form.num_mentor_msgs }}" min="0" style="width: 90%;"></td>
        <td><input type="number" name="num_rubrics" value="{{ form.num_rubrics }}" min="0" style="width: 90%;"></td>
        <td><input type="number" name="median_short" value="{{ form.median_short }}" min="0" style="width: 90%;"></td>
        <td><input type="number" name="median_long" value="{{ form.median_long }}" min="0" style="width: 90%;"></td>
        <td><input type="number" name="median_mentor" value="{{ form.median_mentor }}" min="0" style="width: 90%;"></td>
        <td>
          <select name="has_images">
            <option value="" {% if not form.has_images %}selected{% endif %}>Any</option>
            <option value="yes" {% if form.has_images == 'yes' %}selected{% endif %}>Yes</option>
            <option value="no" {% if form.has_images == 'no' %}selected{% endif %}>No</option>
          </select>
        </td>
        <td></td>
        <td>
          <select name="has_feedback">
            <option value="" {% if not form.has_feedback %}selected{% endif %}>Any</option>
            <option value="yes" {% if form.has_feedback == 'yes' %}selected{% endif %}>Yes</option>
            <option value="no" {% if form.has_feedback == 'no' %}selected{% endif %}>No</option>
          </select>
        </td>
      </tr>
    </thead>
    <tbody>
      {% for t in threads %}
        <tr>
          <td><a href="{{ url_for('view_thread', thread_id=t['thread_id']) }}">{{ t['thread_id'] }}</a></td>
          <td>{{ t['puzzle_text_summary'][:160] }}{% if t['puzzle_text_summary']|length > 160 %}...{% endif %}</td>
          <td>{{ t['school_name'] }}</td>
          <td>{{ t['mentor_name'] }}</td>
          <td>{{ t['num_subs'] }}</td>
          <td>{{ t['num_mentor_msgs'] }}</td>
          <td>{{ t['num_rubrics'] }}</td>
          <td>{{ t['median_short'] }}</td>
          <td>{{ t['median_long'] }}</td>
          <td>{{ t['median_mentor'] }}</td>
          <td>{{ 'Yes' if t['has_images'] else 'No' }}</td>
          <td><a href="{{ url_for('feedback', thread_id=t['thread_id']) }}">Leave Feedback</a></td>
          <td>
          {% if t['thread_id'] in feedback_by_thread %}
            <a href="{{ url_for('view_feedback', thread_id=t['thread_id']) }}">View Feedback</a>
          {% else %}
            &mdash;
          {% endif %}
        </td>
        </tr>
      {% endfor %}
    </tbody>
  </table>
</form>

<div>
  {% if page > 1 %}
    <a href="{{ url_for('index',
                    thread_id=form.thread_id,
                    puzzle_text=form.puzzle_text,
                    num_subs=form.num_subs,
                    num_rubrics=form.num_rubrics,
                    median_short=form.median_short,
                    median_long=form.median_long,
                    median_mentor=form.median_mentor,
                    has_images=form.has_images,
                    has_feedback=form.has_feedback,
                    page=page-1) }}">Prev</a>
  {% endif %}
  Page {{ page }} of {{ max_page }}
  {% if page < max_page %}
    <a href="{{ url_for('index',
                    thread_id=form.thread_id,
                    puzzle_text=form.puzzle_text,
                    num_subs=form.num_subs,
                    num_rubrics=form.num_rubrics,
                    median_short=form.median_short,
                    median_long=form.median_long,
                    median_mentor=form.median_mentor,
                    has_images=form.has_images,
                    has_feedback=form.has_feedback,
                    page=page+1) }}">Next</a>
  {% endif %}
</div>
"""

DETAIL_TEMPLATE = """
<!doctype html>
<title>Thread {{ thread_id }}</title>
<h1>Thread {{ thread_id }}</h1>

<h2>Puzzle</h2>
<div style="white-space: pre-wrap;">{{ puzzle_text|safe }}</div>

{% if submissions %}

<h2>Student and Mentor Info</h2>
<ul>
  <li><strong>Name:</strong> {{ submissions[0]['student_name'] }}</li>
  <li><strong>School:</strong> {{ submissions[0]['student_school'] }}</li>
  <li><strong>Age:</strong> {{ submissions[0]['student_age'] }}</li>
  <li><strong>Mentor:</strong> {{ submissions[0]['mentor_name'] }}</li>
</ul>

<h2>Submissions</h2>
<ul>
  {% for sub in submissions %}
    <li><a href="#sub{{ sub['submission_id'] }}">Submission {{ sub['submission_id'] }}</a></li>
  {% endfor %}
</ul>

{% for sub in submissions %}
  <div id="sub{{ sub['submission_id'] }}" style="border:1px solid #ccc; padding:10px; margin-bottom:20px;">
    <h3>Submission ID: {{ sub['submission_id'] }}</h3>

    <h4>Rubric</h4>
    <ul>
      {% for key, val in sub['rubric'].items() %}
        <li><strong>{{ key }}:</strong> {{ val if val is not none else "N/A" }}</li>
      {% endfor %}
    </ul>

    <h4>Short Answer</h4>
    <div style="white-space: pre-wrap;">{{ sub['shortanswer']|safe }}</div>

    <h4>Long Answer</h4>
    <div style="white-space: pre-wrap;">{{ sub['longanswer']|safe }}</div>

    <h4>Mentor Message</h4>
    <div style="white-space: pre-wrap;">{{ sub['r_message']|safe }}</div>
  </div>
{% endfor %}

{% else %}
<p><em>No submissions found for this thread.</em></p>
{% endif %}

<p><a href="{{ url_for('index') }}">Back to thread list</a></p>

<p><a href="{{ url_for('feedback', thread_id=thread_id) }}">Leave Feedback</a></p>
"""

FEEDBACK_TEMPLATE = """
<!doctype html>
<title>Leave Feedback for Thread {{ thread_id }}</title>
<h1>Leave Feedback for Thread {{ thread_id }}</h1>

<form method="post">
  <label for="comments"><strong>Comments:</strong></label><br>
  <textarea name="comments" id="comments" rows="5" cols="80"></textarea><br>
  <input type="submit" value="Submit Feedback">
</form>

<hr>

<h2>Puzzle</h2>
<div style="white-space: pre-wrap;">{{ puzzle_text|safe }}</div>

{% if submissions %}
<h2>Student and Mentor Info</h2>
<ul>
  <li><strong>Name:</strong> {{ submissions[0]['student_name'] }}</li>
  <li><strong>School:</strong> {{ submissions[0]['student_school'] }}</li>
  <li><strong>Age:</strong> {{ submissions[0]['student_age'] }}</li>
  <li><strong>Mentor:</strong> {{ submissions[0]['mentor_name'] }}</li>
</ul>

<h2>Submissions</h2>
<ul>
  {% for sub in submissions %}
    <li><a href="#sub{{ sub['submission_id'] }}">Submission {{ sub['submission_id'] }}</a></li>
  {% endfor %}
</ul>

{% for sub in submissions %}
  <div id="sub{{ sub['submission_id'] }}" style="border:1px solid #ccc; padding:10px; margin-bottom:20px;">
    <h3>Submission ID: {{ sub['submission_id'] }}</h3>

    <h4>Rubric</h4>
    <ul>
      {% for key, val in sub['rubric'].items() %}
        <li><strong>{{ key }}:</strong> {{ val if val is not none else "N/A" }}</li>
      {% endfor %}
    </ul>

    <h4>Short Answer</h4>
    <div style="white-space: pre-wrap;">{{ sub['shortanswer']|safe }}</div>

    <h4>Long Answer</h4>
    <div style="white-space: pre-wrap;">{{ sub['longanswer']|safe }}</div>

    <h4>Mentor Message</h4>
    <div style="white-space: pre-wrap;">{{ sub['r_message']|safe }}</div>
  </div>
{% endfor %}
{% else %}
<p><em>No submissions available.</em></p>
{% endif %}

<p><a href="{{ url_for('index') }}">Back to thread list</a></p>
"""

FEEDBACK_VIEW_TEMPLATE = """
<!doctype html>
<title>Feedback for Thread {{ thread_id }}</title>
<h1>Feedback for Thread {{ thread_id }}</h1>

{% if feedback_list %}
  <table border="1" cellpadding="4" cellspacing="0">
    <thead>
      <tr>
        <th>Timestamp</th>
        {% for key in feedback_list[0].keys() if key not in ['thread_id', 'timestamp'] %}
          <th>{{ key }}</th>
        {% endfor %}
      </tr>
    </thead>
    <tbody>
      {% for item in feedback_list %}
        <tr>
          <td>{{ item.timestamp }}</td>
          {% for key in item.keys() if key not in ['thread_id', 'timestamp'] %}
            <td>{{ item[key] }}</td>
          {% endfor %}
        </tr>
      {% endfor %}
    </tbody>
  </table>
{% else %}
  <p>No feedback available for this thread.</p>
{% endif %}

<hr>
<h2>Puzzle</h2>
<div style="white-space: pre-wrap;">{{ puzzle_text|safe }}</div>

{% if submissions %}
<h2>Student and Mentor Info</h2>
<ul>
  <li><strong>Name:</strong> {{ submissions[0]['student_name'] }}</li>
  <li><strong>School:</strong> {{ submissions[0]['student_school'] }}</li>
  <li><strong>Age:</strong> {{ submissions[0]['student_age'] }}</li>
  <li><strong>Mentor:</strong> {{ submissions[0]['mentor_name'] }}</li>
</ul>

<h2>Submissions</h2>
<ul>
  {% for sub in submissions %}
    <li><a href="#sub{{ sub['submission_id'] }}">Submission {{ sub['submission_id'] }}</a></li>
  {% endfor %}
</ul>

{% for sub in submissions %}
  <div id="sub{{ sub['submission_id'] }}" style="border:1px solid #ccc; padding:10px; margin-bottom:20px;">
    <h3>Submission ID: {{ sub['submission_id'] }}</h3>

    <h4>Rubric</h4>
    <ul>
      {% for key, val in sub['rubric'].items() %}
        <li><strong>{{ key }}:</strong> {{ val if val is not none else "N/A" }}</li>
      {% endfor %}
    </ul>

    <h4>Short Answer</h4>
    <div style="white-space: pre-wrap;">{{ sub['shortanswer']|safe }}</div>

    <h4>Long Answer</h4>
    <div style="white-space: pre-wrap;">{{ sub['longanswer']|safe }}</div>

    <h4>Mentor Message</h4>
    <div style="white-space: pre-wrap;">{{ sub['r_message']|safe }}</div>
  </div>
{% endfor %}
{% else %}
<p><em>No submissions available.</em></p>
{% endif %}

<p><a href="{{ url_for('index') }}">Back to thread list</a></p>
"""

# -- Load Data and Image Processing Helpers -- 
def html_to_plaintext(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text(separator=' ')
    clean = re.sub(r'\s+', ' ', html.unescape(text)).strip()
    return clean
    
def remove_control_characters(text):
    return ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
    
def fix_common_mojibake(text):
    # Mapping of mojibake sequences to actual characters
    replacements = {
        "â": "—",  # em dash
        "â": "–",  # en dash
        "â¦": "…",  # ellipsis
        "â": "“",  # left double quote
        "â": "”",  # right double quote
        "â": "‘",  # left single quote
        "â": "’",  # right single quote
        "â¢": "•",  # bullet
        "Ã©": "é",
        "Ã¨": "è",
        "Ã¢": "â",
        "Ãª": "ê",
        "Ã®": "î",
        "Ã´": "ô",
        "Ã»": "û",
        "Ã¶": "ö",
        "Ã¤": "ä",
        "Ã§": "ç",
        "Â": "",     # often a stray prefix
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text

def sanitize(text):
    if not text:
        return ''
        
    text = html.unescape(text)
    text = fix_common_mojibake(text)
    text = remove_control_characters(text)
    
    # Normalize newline/whitespace variants
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('\\r\\n', '\n').replace('\\r', '\n').replace('\\n', '\n')
    text = text.replace('\\t', '\t')

    # Replace tab with 4 non-breaking spaces
    text = text.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')

    # Remove non-breaking spaces and extended Latin-1 characters
    text = text.replace('\xa0', ' ')
    text = re.sub(r'[^\x20-\x7E\n]', '', text)

    # Strip HTML tags (including <br>), since this version doesn't preserve any
    text = re.sub(r'<[^>]*?>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Normalize excess whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[ ]*\n[ ]*', '\n', text)

    # Final HTML formatting: turn newlines into <br>
    text = text.replace('\n', '<br>')

    return text.strip()
    
def sanitize_preserve_images(text):
    if not text:
        return ''
    
    text = html.unescape(text)
    text = fix_common_mojibake(text)
    text = remove_control_characters(text)

    # Normalize newline/whitespace variants
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('\\r\\n', '\n').replace('\\r', '\n').replace('\\n', '\n')
    text = text.replace('\\t', '\t')

    # Replace tab with 4 non-breaking spaces
    text = text.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')

    # Preserve <img> and <br> tags using placeholders
    img_pattern = re.compile(r'<img\s+[^>]*?src=["\']data:image/[^;]+;base64,[^"\']+["\'][^>]*?>', re.IGNORECASE)
    br_pattern = re.compile(r'<br\s*/?>', re.IGNORECASE)

    preserved_tags = []
    placeholder_img = "___IMG_TAG___"
    placeholder_br = "___BR_TAG___"

    for tag in img_pattern.findall(text):
        # Wrap image with line breaks
        wrapped = f"<br>{tag}<br>"
        preserved_tags.append((placeholder_img, wrapped))
        text = text.replace(tag, placeholder_img, 1)

    for tag in br_pattern.findall(text):
        preserved_tags.append((placeholder_br, "<br>"))  # normalize to consistent tag
        text = text.replace(tag, placeholder_br, 1)

    # Remove all other HTML
    text = re.sub(r'<[^>]*?>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Restore preserved tags
    for placeholder, tag in preserved_tags:
        text = text.replace(placeholder, tag, 1)

    # Remove extended Unicode characters outside basic ASCII + newline
    text = text.replace('\xa0', ' ')
    text = re.sub(r'[^\x20-\x7E\n]', '', text)

    # Normalize excess whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[ ]*\n[ ]*', '\n', text)

    # Convert final newlines to <br> tags
    text = text.replace('\n', '<br>')

    return text.strip()
    
def compute_thread_metrics(thread_id, thread_data):
    subs = thread_data["submissions"]
    num_subs = len(subs)
    num_rubrics = sum(1 for s in subs if any(v is not None for v in s["rubric"].values()))
    short_lens = [len(s["shortanswer"]) for s in subs]
    long_lens = [len(s["longanswer"]) for s in subs]
    mentor_lens = [len(s["r_message"]) for s in subs]
    num_mentor_msgs = sum(1 for s in subs if s["r_message"].strip())

    # Check for inline base64 images in any submission field
    base64_img_pattern = re.compile(r'data:image/[^;]+;base64,', re.IGNORECASE)
    has_images = any(
        base64_img_pattern.search(s["shortanswer"]) or
        base64_img_pattern.search(s["longanswer"]) or
        base64_img_pattern.search(s["r_message"])
        for s in subs
    )

    return {
        "thread_id": thread_id,
        "puzzle_text": thread_data["puzzle_text"],
        "puzzle_text_summary": html_to_plaintext(thread_data["puzzle_text"]),
        "mentor_name": thread_data["mentor_name"],
        "school_name": thread_data["submissions"][0]["student_school"] if thread_data["submissions"] else "Unknown",        
        "num_subs": num_subs,
        "num_rubrics": num_rubrics,
        "num_mentor_msgs": num_mentor_msgs,
        "median_short": median(short_lens) if short_lens else 0,
        "median_long": median(long_lens) if long_lens else 0,
        "median_mentor": median(mentor_lens) if mentor_lens else 0,
        "has_images": has_images,
    }    
    
# -- Data Load at Startup --
def init_feedback_db():
    if not os.path.exists(FEEDBACK_DB):
        conn = sqlite3.connect(FEEDBACK_DB)
        conn.execute("CREATE TABLE IF NOT EXISTS feedback (thread_id INTEGER, entry TEXT)")
        conn.commit()
        conn.close()

def load_feedback():
    global FEEDBACK_BY_THREAD
    FEEDBACK_BY_THREAD = {}

    if not os.path.exists(FEEDBACK_DB):
        return

    conn = sqlite3.connect(FEEDBACK_DB)
    cur = conn.cursor()

    try:
        cur.execute("SELECT entry FROM feedback")
        rows = cur.fetchall()

        for row in rows:
            try:
                entry = json.loads(row[0])
                tid = entry.get("thread_id")
                if tid is not None:
                    FEEDBACK_BY_THREAD.setdefault(tid, []).append(entry)
            except json.JSONDecodeError:
                continue
    finally:
        conn.close()

def load_data():
    global THREADS, THREAD_INDEX, THREAD_METRICS

    if os.path.exists(MEM_PICKLE):
        print("Loading cached data from pickle...")
        with open(MEM_PICKLE, "rb") as f:
            THREADS, THREAD_INDEX, THREAD_METRICS = pickle.load(f)
        print(f"Loaded {len(THREAD_INDEX)} threads from cache.")
        return

    THREADS = {}
    THREAD_INDEX = []
    THREAD_METRICS = {}

    def get_connection():
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro&cache=shared", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=OFF;")
        conn.execute("PRAGMA synchronous=OFF;")        
        return conn

    def load_threads():
        global THREADS_PICKLE
        if os.path.exists(THREADS_PICKLE):
            with open(THREADS_PICKLE, "rb") as f:
                return pickle.load(f)

        conn = get_connection()
        cur = conn.cursor()

        thread_count = cur.execute("""
            SELECT COUNT(*) FROM pow_threads t
            JOIN pow_publications p ON t.publication = p.id
            JOIN pow_puzzles z ON p.puzzle = z.id
        """).fetchone()[0]

        cur.execute("""
            SELECT t.id AS thread_id, z.text AS puzzle_text,
                   du.first_name || ' ' || du.last_name AS mentor_name
            FROM pow_threads t
            JOIN pow_publications p ON t.publication = p.id
            JOIN pow_puzzles z ON p.puzzle = z.id
            LEFT JOIN dir_users du ON t.mentor = du.id
        """)

        local_threads = {}
        local_index = []

        for row in tqdm(cur.fetchall(), total=thread_count, desc="Threads"):
            tid = row["thread_id"]
            local_threads[tid] = {
                "mentor_name": row["mentor_name"] or "Unknown",
                "puzzle_text": sanitize_preserve_images(row["puzzle_text"]),
                "submissions": []
            }
            local_index.append(tid)

        conn.close()
        with open(THREADS_PICKLE, "wb") as f:
            pickle.dump((local_threads, local_index), f)

        return local_threads, local_index


    def load_students():
        global STUDENTS_PICKLE
        if os.path.exists(STUDENTS_PICKLE):
            with open(STUDENTS_PICKLE, "rb") as f:
                return pickle.load(f)

        conn = get_connection()
        cur = conn.cursor()

        student_count = cur.execute("SELECT COUNT(*) FROM dir_users").fetchone()[0]
        cur.execute("SELECT id, first_name, last_name, ageinyears FROM dir_users")

        student_data = {}
        for row in tqdm(cur.fetchall(), total=student_count, desc="Students"):
            sid = row["id"]
            fname = row["first_name"] or ""
            lname = row["last_name"] or ""
            age = str(row["ageinyears"]) if row["ageinyears"] is not None else ""
            student_data[sid] = {
                "name": f"{fname} {lname}".strip(),
                "age": age
            }

        conn.close()
        with open(STUDENTS_PICKLE, "wb") as f:
            pickle.dump(student_data, f)

        return student_data


    def load_schools():
        global SCHOOLS_PICKLE
        if os.path.exists(SCHOOLS_PICKLE):
            with open(SCHOOLS_PICKLE, "rb") as f:
                return pickle.load(f)

        conn = get_connection()
        cur = conn.cursor()

        school_count = cur.execute("""
            SELECT COUNT(*) FROM (
                SELECT m.user_id, g.name AS school_name
                FROM dir_memberships m
                JOIN dir_groups g ON m.group_id = g.id
                WHERE (m.deleted IS NULL OR m.deleted = 0)
            )
        """).fetchone()[0]

        cur.execute("""
            SELECT m.user_id, g.name AS school_name
            FROM dir_memberships m
            JOIN dir_groups g ON m.group_id = g.id
            WHERE (m.deleted IS NULL OR m.deleted = 0)
        """)

        student_school = {}
        for row in tqdm(cur.fetchall(), total=school_count, desc="Schools"):
            uid = row["user_id"]
            if uid not in student_school:
                student_school[uid] = row["school_name"] or "Unknown"

        conn.close()
        with open(SCHOOLS_PICKLE, "wb") as f:
            pickle.dump(student_school, f)

        return student_school


    def load_submissions():
        global SUBMISSIONS_PICKLE
        if os.path.exists(SUBMISSIONS_PICKLE):
            with open(SUBMISSIONS_PICKLE, "rb") as f:
                return pickle.load(f)

        conn = get_connection()
        cur = conn.cursor()

        submission_count = cur.execute("""
            SELECT COUNT(*) FROM pow_submissions s
            LEFT JOIN pow_responses r ON r.submission_id = s.id
            LEFT JOIN pow_rubric rb ON r.rubric_id = rb.id
            WHERE s.shortanswer IS NOT NULL AND s.longanswer IS NOT NULL AND r.id IS NOT NULL
        """).fetchone()[0]

        cur.execute("""
            SELECT
                s.id AS submission_id,
                s.thread_id,
                s.shortanswer, s.longanswer,
                s.creator AS student_creator_id,
                r.message AS r_message,
                rb.strategy, rb.interpretation, rb.completeness,
                rb.clarity, rb.reflection, rb.accuracy
            FROM pow_submissions s
            LEFT JOIN pow_responses r ON r.submission_id = s.id
            LEFT JOIN pow_rubric rb ON r.rubric_id = rb.id
            WHERE s.shortanswer IS NOT NULL AND s.longanswer IS NOT NULL AND r.id IS NOT NULL
        """)

        rows = list(tqdm(cur.fetchall(), total=submission_count, desc="Submissions"))
        conn.close()
        
        sanitized_rows = []
        for row in tqdm(rows, total=submission_count, desc="Sanitizing submissions"):
            row_dict = dict(row)
            row_dict["shortanswer"] = sanitize_preserve_images(row_dict.get("shortanswer", ""))
            row_dict["longanswer"] = sanitize_preserve_images(row_dict.get("longanswer", ""))
            row_dict["r_message"] = sanitize_preserve_images(row_dict.get("r_message", ""))
            sanitized_rows.append(row_dict)

        with open(SUBMISSIONS_PICKLE, "wb") as f:
            pickle.dump(sanitized_rows, f)

        return sanitized_rows

    print("Loading data with threads...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(load_threads): "threads",
            executor.submit(load_students): "students",
            executor.submit(load_schools): "schools",
            executor.submit(load_submissions): "submissions",
        }

        results = {}
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()

    thread_data, thread_index = results["threads"]
    student_data = results["students"]
    student_school = results["schools"]
    submission_rows = results["submissions"]

    THREADS.update(thread_data)
    THREAD_INDEX.extend(thread_index)

    for row in submission_rows:
        tid = row["thread_id"]
        if tid not in THREADS:
            continue
        creator_id = row["student_creator_id"]
        student_info = student_data.get(creator_id, {})
        student_name = student_info.get("name", "Unknown")
        student_age = student_info.get("age", "")
        student_school_name = student_school.get(creator_id, "Unknown")

        THREADS[tid]["submissions"].append({
            "submission_id": row["submission_id"],
            "student_name": student_name,
            "student_school": student_school_name,
            "student_age": student_age,
            "mentor_name": THREADS[tid]["mentor_name"],
            "shortanswer": sanitize_preserve_images(row["shortanswer"]),
            "longanswer": sanitize_preserve_images(row["longanswer"]),
            "r_message": sanitize_preserve_images(row["r_message"]),
            "rubric": {
                "Strategy": row["strategy"],
                "Interpretation": row["interpretation"],
                "Completeness": row["completeness"],
                "Clarity": row["clarity"],
                "Reflection": row["reflection"],
                "Accuracy": row["accuracy"]
            }
        })

    print("Computing thread metrics...")
    for tid in tqdm(THREAD_INDEX, total=len(THREAD_INDEX), desc="Metrics"):
        THREAD_METRICS[tid] = compute_thread_metrics(tid, THREADS[tid])

    print("Saving parsed data to pickle cache...")
    with open(MEM_PICKLE, "wb") as f:
        pickle.dump((THREADS, THREAD_INDEX, THREAD_METRICS), f)

    print("Load complete.")
    
# -- Routes --
@app.route('/')
def index():
    form = request.args
    page = int(form.get("page", 1))
    offset = (page - 1) * THREADS_PER_PAGE

    # Get form values
    filters = {
        "thread_id": form.get("thread_id", "").strip(),
        "puzzle_text": form.get("puzzle_text", "").strip().lower(),
        "school_name": form.get("school_name", "").strip(),
        "mentor_name": form.get("mentor_name", "").strip(),        
        "num_subs": form.get("num_subs", "").strip(),
        "num_mentor_msgs": form.get("num_mentor_msgs", "").strip(),
        "num_rubrics": form.get("num_rubrics", "").strip(),
        "median_short": form.get("median_short", "").strip(),
        "median_long": form.get("median_long", "").strip(),
        "median_mentor": form.get("median_mentor", "").strip(),
        "has_images": form.get("has_images", "").strip().lower(),  # yes/no
        "has_feedback": form.get("has_feedback", "").strip().lower(), # yes/no
    }

    def passes_filters(row):
        if filters["thread_id"] and filters["thread_id"] not in str(row["thread_id"]):
            return False
        if filters["puzzle_text"] and filters["puzzle_text"] not in row["puzzle_text"].lower():
            return False
        if filters["school_name"] and filters["school_name"].lower() not in row.get("school_name", "").lower():
            return False
        if filters["mentor_name"] and filters["mentor_name"].lower() not in row.get("mentor_name", "").lower():
            return False            
        if filters["num_subs"]:
            try:
                if row["num_subs"] < int(filters["num_subs"]):
                    return False
            except ValueError:
                return False
        if filters["num_mentor_msgs"]:
            try:
                if row["num_mentor_msgs"] < int(filters["num_mentor_msgs"]):
                    return False
            except ValueError:
                return False                
        if filters["num_rubrics"]:
            try:
                if row["num_rubrics"] < int(filters["num_rubrics"]):
                    return False
            except ValueError:
                return False
        if filters["median_short"]:
            try:
                if row["median_short"] < int(filters["median_short"]):
                    return False
            except ValueError:
                return False
        if filters["median_long"]:
            try:
                if row["median_long"] < int(filters["median_long"]):
                    return False
            except ValueError:
                return False
        if filters["median_mentor"]:
            try:
                if row["median_mentor"] < int(filters["median_mentor"]):
                    return False
            except ValueError:
                return False
        if filters["has_images"] == "yes" and not row["has_images"]:
            return False
        if filters["has_images"] == "no" and row["has_images"]:
            return False
        if filters["has_feedback"] == "yes" and row["thread_id"] not in FEEDBACK_BY_THREAD:
            return False
        if filters["has_feedback"] == "no" and row["thread_id"] in FEEDBACK_BY_THREAD:
            return False            
        return True

    all_metrics = list(THREAD_METRICS.values())
    filtered = [row for row in all_metrics if passes_filters(row)]
    total = len(filtered)
    max_page = max(1, (total + THREADS_PER_PAGE - 1) // THREADS_PER_PAGE)
    page = max(1, min(page, max_page))
    paged = filtered[offset:offset + THREADS_PER_PAGE]

    return render_template_string(
        INDEX_TEMPLATE,
        threads=paged,
        total=total,
        page=page,
        max_page=max_page,
        form=filters,
        feedback_by_thread=FEEDBACK_BY_THREAD
    )

@app.route('/thread/<int:thread_id>')
def view_thread(thread_id):
    if thread_id not in THREADS:
        abort(404)

    data = THREADS[thread_id]
    return render_template_string(DETAIL_TEMPLATE,
                                  thread_id=thread_id,
                                  puzzle_text=data["puzzle_text"],
                                  submissions=data["submissions"])

@app.route('/feedback/<int:thread_id>', methods=["GET", "POST"])
def feedback(thread_id):
    if thread_id not in THREADS:
        abort(404)

    if request.method == "POST":
        feedback_entry = {
            "thread_id": thread_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Capture all form fields
        for key in request.form:
            feedback_entry[key] = request.form[key]

        # Save to feedback db
        conn = sqlite3.connect(FEEDBACK_DB)
        cur = conn.cursor()
        cur.execute("INSERT INTO feedback (thread_id, entry) VALUES (?, ?)", [feedback_entry["thread_id"], json.dumps(feedback_entry, ensure_ascii=False)])
        conn.commit()
        conn.close()
        
        FEEDBACK_BY_THREAD.setdefault(thread_id, []).append(feedback_entry)

        return f"<p>Thank you for your feedback! <a href='{url_for('index')}'>Return to list</a></p>"

    data = THREADS[thread_id]
    return render_template_string(FEEDBACK_TEMPLATE,
                                  thread_id=thread_id,
                                  puzzle_text=data["puzzle_text"],
                                  submissions=data["submissions"])

@app.route('/feedback/view/<int:thread_id>')
def view_feedback(thread_id):
    if thread_id not in THREADS:
        abort(404)

    # Use the in-memory feedback cache
    feedback_list = FEEDBACK_BY_THREAD.get(thread_id, [])

    data = THREADS[thread_id]
    return render_template_string(
        FEEDBACK_VIEW_TEMPLATE,
        thread_id=thread_id,
        puzzle_text=data["puzzle_text"],
        submissions=data["submissions"],
        feedback_list=feedback_list
    )
                                  
# -- Entry Point --
if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        init_feedback_db()
        load_data()
        load_feedback()
    app.run(debug=True)