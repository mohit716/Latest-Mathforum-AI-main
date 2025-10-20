import sqlite3
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox
import html
import base64
import re
from io import BytesIO
from PIL import Image, ImageTk
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tkinter import Menu
from difflib import SequenceMatcher
import os
from collections import OrderedDict
import time
from queue import Queue

# Constants
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
DETAIL_WIDTH = 1200
DETAIL_HEIGHT = 800

try:
    from PIL.Image import Resampling
    RESAMPLE = Resampling.LANCZOS
except ImportError:
    RESAMPLE = Image.LANCZOS

db_file = os.path.abspath("mathforum.db")
submission_cache = OrderedDict()  # Global cache of fetched submissions
previous_submission_id_cache = OrderedDict() # key = (thread_id, submission_id) -> int or None
already_prefetched = set()
MAX_CACHE_SIZE = 1000
prefetch_abort_flag = threading.Event() # to stop prefetching for an on-demand fetch
fetch_in_progress = set()
fetch_lock = threading.Lock()

def sanitize(text):
    if not text:
        return ''
    placeholder_pattern = re.compile(r'\[[^\[\]]+? Image \d+\]')
    preserved = {}
    def preserve(match):
        key = f"__PLACEHOLDER_{len(preserved)}__"
        preserved[key] = match.group(0)
        return key
    text = placeholder_pattern.sub(preserve, text)
    text = html.unescape(text)
    text = re.sub(r'<[^>]+?>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = text.encode('utf-8').decode('unicode_escape')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[ ]*\n[ ]*', '\n', text)
    for key, val in preserved.items():
        text = text.replace(key, val)
    return text.strip()

def replace_base64_placeholders(field_label, text):
    if not text:
        return "", []
    pattern = re.compile(r'<[^>]+?base64,([^"\'>\s]+)[^>]*>', re.IGNORECASE)
    b64_list = []
    def repl(match):
        idx = len(b64_list) + 1
        b64_data = match.group(1)
        placeholder = f"[{field_label} Image {idx}]"
        b64_list.append((field_label, idx, b64_data))
        return placeholder
    new_text = pattern.sub(repl, text)
    return new_text, b64_list
    
def get_threads_and_submissions():
    global db_file

    print("Loading initial submission data...")
    conn = sqlite3.connect(f'file:{db_file}?mode=ro', uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.executescript("""
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA cache_size = 100000;
    """)
    
    def get_submissions():
        conn2 = sqlite3.connect(f'file:{db_file}?mode=ro', uri=True, check_same_thread=False)
        conn2.row_factory = sqlite3.Row
        cursor2 = conn2.cursor()

        cursor2.execute("""
            SELECT COUNT(*)
            FROM pow_submissions s
            LEFT JOIN pow_responses r ON r.submission_id = s.id
            LEFT JOIN pow_rubric rb ON r.rubric_id = rb.id
            WHERE s.shortanswer IS NOT NULL AND s.longanswer IS NOT NULL AND r.id IS NOT NULL
        """)
        count2 = cursor2.fetchone()[0]

        cursor2.execute("""
            SELECT
                s.id AS submission_id,
                s.thread_id,
                s.shortanswer,
                s.longanswer,
                r.id AS response_id,
                r.message AS r_message,
                rb.strategy, rb.interpretation, rb.completeness,
                rb.clarity, rb.reflection, rb.accuracy
            FROM pow_submissions s
            LEFT JOIN pow_responses r ON r.submission_id = s.id
            LEFT JOIN pow_rubric rb ON r.rubric_id = rb.id
            WHERE s.shortanswer IS NOT NULL AND s.longanswer IS NOT NULL AND r.id IS NOT NULL
        """)

        rows2 = []
        for row in tqdm(cursor2, total=count2, desc="Fetching submissions", unit="row", position=1):
            rows2.append(row)
    
        conn2.close()
        
        submission_queue.put(count2)  
        submission_queue.put(rows2)  

    # Start getting submissions in background while we get threads
    submission_queue = Queue()
    count_thread = threading.Thread(target=get_submissions, daemon=True)
    count_thread.start()        

    # --- Count threads ---
    cursor.execute("""
        SELECT COUNT(*)
        FROM pow_threads t
        LEFT JOIN pow_publications p ON t.publication = p.id
        LEFT JOIN pow_puzzles z ON p.puzzle = z.id
    """)
    thread_total = cursor.fetchone()[0]

    # --- Load threads ---
    thread_map = {}
    thread_query = cursor.execute("""
        SELECT t.id AS thread_id, z.text AS puzzle_text
        FROM pow_threads t
        LEFT JOIN pow_publications p ON t.publication = p.id
        LEFT JOIN pow_puzzles z ON p.puzzle = z.id
    """)
    for thread in tqdm(thread_query, total=thread_total, desc="Loading all threads", unit="thread", position=0):
        if 'thread_id' in thread.keys():
            tid = thread['thread_id']
        else:
            print(f"[WARNING] Missing 'thread_id' in row: keys = {thread.keys()}")
            continue  
            
        thread_map[tid] = {
            'puzzle_text': sanitize(thread['puzzle_text']),
            'submissions': []
        }

    # --- Load submissions ---
    # These calls to get will block until submissions thread has finished getting all records above
    submission_total = submission_queue.get()
    rows = submission_queue.get()

    def process_submission_row(row):
        b64_images = []
        longanswer_cleaned, imgs = replace_base64_placeholders("Long Answer", row['longanswer'])
        b64_images.extend(imgs)
        shortanswer_cleaned, imgs = replace_base64_placeholders("Short Answer", row['shortanswer'])
        b64_images.extend(imgs)
        rmsg_cleaned, imgs = replace_base64_placeholders("Mentor Message", row['r_message'])
        b64_images.extend(imgs)

        return (row['thread_id'], {
            'submission_id': row['submission_id'],
            'thread_id': row['thread_id'],
            'shortanswer': sanitize(shortanswer_cleaned),
            'longanswer': sanitize(longanswer_cleaned),
            'r_message': sanitize(rmsg_cleaned),
            'b64_images': b64_images,
            'rubric': {
                'Strategy': row['strategy'],
                'Interpretation': row['interpretation'],
                'Completeness': row['completeness'],
                'Clarity': row['clarity'],
                'Reflection': row['reflection'],
                'Accuracy': row['accuracy']
            }
        })

    with ThreadPoolExecutor() as executor:
        futures = list(tqdm(executor.map(process_submission_row, rows),
                            total=len(rows),
                            desc="Consolidating submission conversation threads",
                            unit="submission",
                            position=2))

    for thread_id, submission in futures:
        thread_map[thread_id]['submissions'].append(submission)

    return thread_map

def fetch_full_submission(sub_id, wait_if_pending=True):
    global submission_cache, db_file, fetch_in_progress

    waited = False

    while True:
        with fetch_lock:
            if sub_id in submission_cache:
                print(f"[Cache] Found submission {sub_id}")
                submission_cache.move_to_end(sub_id)
                return submission_cache[sub_id]

            if sub_id in fetch_in_progress:
                if wait_if_pending:
                    print(f"[Wait] Waiting for submission {sub_id} to be fetched...")
                    waited = True
                else:
                    print(f"[Fetch] Already in progress for {sub_id}, skipping duplicate")
                    return None
            else:
                fetch_in_progress.add(sub_id)
                break  # This thread owns the fetch

        if waited:
            time.sleep(1)  # Sleep briefly and check again
        else:
            return None  # We are not waiting, so return None immediately

    try:
        print(f"[Fetch] Getting full submission {sub_id}")
        conn = sqlite3.connect(f'file:{db_file}?mode=ro', uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("""
        SELECT
            s.id AS submission_id,
            s.thread_id,
            s.shortanswer,
            s.longanswer,
            r.message AS r_message,
            rb.strategy, rb.interpretation, rb.completeness,
            rb.clarity, rb.reflection, rb.accuracy,
            du.first_name || ' ' || du.last_name AS mentor_name,
            su.first_name || ' ' || su.last_name AS student_name,
            su.ageinyears AS student_age,
            dg.name AS school_name,
            z.text AS puzzle_text
        FROM pow_submissions s
        LEFT JOIN pow_responses r ON r.submission_id = s.id
        LEFT JOIN pow_rubric rb ON r.rubric_id = rb.id
        LEFT JOIN pow_threads t ON s.thread_id = t.id
        LEFT JOIN pow_publications p ON t.publication = p.id
        LEFT JOIN pow_puzzles z ON p.puzzle = z.id
        LEFT JOIN dir_users du ON t.mentor = du.id
        LEFT JOIN dir_users su ON su.id = s.creator
        LEFT JOIN dir_memberships dm ON dm.user_id = s.creator
        LEFT JOIN dir_groups dg ON dg.id = dm.group_id
        WHERE s.id = ?
        """, (sub_id,))
        
        row = cur.fetchone()
        conn.close()

        if not row:
            return None

        longanswer, long_imgs = replace_base64_placeholders("Long Answer", row['longanswer'])
        shortanswer, short_imgs = replace_base64_placeholders("Short Answer", row['shortanswer'])
        rmsg, rmsg_imgs = replace_base64_placeholders("Mentor Message", row['r_message'])

        entry = {
            'submission_id': row['submission_id'],
            'thread_id': row['thread_id'],
            'shortanswer': sanitize(shortanswer),
            'longanswer': sanitize(longanswer),
            'r_message': sanitize(rmsg),
            'student_name': sanitize(row['student_name'] or ''),
            'mentor_name': sanitize(row['mentor_name'] or ''),
            'school_name': sanitize(row['school_name'] or ''),
            'student_age': row['student_age'],
            'rubric': {
                'Strategy': row['strategy'],
                'Interpretation': row['interpretation'],
                'Completeness': row['completeness'],
                'Clarity': row['clarity'],
                'Reflection': row['reflection'],
                'Accuracy': row['accuracy']
            },
            'b64_images': short_imgs + long_imgs + rmsg_imgs,
            'puzzle_text': sanitize(row['puzzle_text'] or '')
        }

        with fetch_lock:
            if len(submission_cache) >= MAX_CACHE_SIZE:
                evicted_sub_id, _ = submission_cache.popitem(last=False)
                already_prefetched.discard(evicted_sub_id)
            submission_cache[sub_id] = entry

        return entry
    finally: # this executes after the return
        with fetch_lock:
            fetch_in_progress.discard(sub_id)    

def fetch_previous_submission(thread_id, submission_id):
    global previous_submission_id_cache
    global db_file
    
    key = (thread_id, submission_id)
    if key in previous_submission_id_cache:
        previous_submission_id_cache.move_to_end(key)  # mark as recently used
        prev_id = previous_submission_id_cache[key]
        return fetch_full_submission(prev_id) if prev_id is not None else None

    conn = sqlite3.connect(f'file:{db_file}?mode=ro', uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT s.id
        FROM pow_submissions s
        WHERE s.thread_id = ? AND s.id < ?
        ORDER BY s.id DESC LIMIT 1
    """, (thread_id, submission_id))

    row = cur.fetchone()
    conn.close()

    prev_id = row['id'] if row else None

    if len(previous_submission_id_cache) >= MAX_CACHE_SIZE:
        previous_submission_id_cache.popitem(last=False)  # evict LRU
    
    previous_submission_id_cache[key] = prev_id

    return fetch_full_submission(prev_id) if prev_id is not None else None

def render_detail_window(entry, previous_entry=None, threads=None):
    detail = tk.Toplevel()
    detail.title(f"Submission {entry['submission_id']}")

    canvas = tk.Canvas(detail)
    v_scrollbar = ttk.Scrollbar(detail, orient="vertical", command=canvas.yview)
    h_scrollbar = ttk.Scrollbar(detail, orient="horizontal", command=canvas.xview)
    canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

    canvas.grid(row=0, column=0, sticky="nsew")
    v_scrollbar.grid(row=0, column=1, sticky="ns")
    h_scrollbar.grid(row=1, column=0, sticky="ew")

    detail.grid_rowconfigure(0, weight=1)
    detail.grid_columnconfigure(0, weight=1)

    scrollable_frame = ttk.Frame(canvas)
    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    meta_frame = ttk.Frame(scrollable_frame)
    meta_frame.pack(fill='x', padx=5, pady=(5, 0))
    small_font = ('Arial', 9)

    # --- Meta Frame Line 1 ---
    meta_frame1 = ttk.Frame(scrollable_frame)
    meta_frame1.pack(fill='x', padx=5, pady=(5, 0))
    small_font = ('Arial', 9)

    ttk.Label(meta_frame1, text="Submission ID:", font=small_font).pack(side='left')
    ttk.Label(meta_frame1, text=str(entry['submission_id']), font=small_font).pack(side='left', padx=(0, 10))

    ttk.Label(meta_frame1, text="Student Name:", font=small_font).pack(side='left')
    ttk.Label(meta_frame1, text=entry.get('student_name', ''), font=small_font).pack(side='left', padx=(0, 10))
    
    ttk.Label(meta_frame1, text="Student Age:", font=small_font).pack(side='left')
    ttk.Label(meta_frame1, text=str(entry.get('student_age', 'N/A')), font=small_font).pack(side='left', padx=(0, 10))    

    # --- Meta Frame Line 2 ---
    meta_frame2 = ttk.Frame(scrollable_frame)
    meta_frame2.pack(fill='x', padx=5, pady=(0, 0))

    ttk.Label(meta_frame2, text="School Name:", font=small_font).pack(side='left')
    ttk.Label(meta_frame2, text=entry.get('school_name', ''), font=small_font).pack(side='left', padx=(0, 10))

    ttk.Label(meta_frame2, text="Mentor Name:", font=small_font).pack(side='left')
    ttk.Label(meta_frame2, text=entry['mentor_name'], font=small_font).pack(side='left', padx=(0, 10))

    rubric_frame = ttk.Frame(scrollable_frame)
    rubric_frame.pack(fill='x', padx=5, pady=(0, 10))
    
    for label in ["Strategy", "Interpretation", "Completeness", "Clarity", "Reflection", "Accuracy"]:
        ttk.Label(rubric_frame, text=f"{label}:", font=small_font).pack(side='left')
        val = entry['rubric'].get(label, 'N/A')
        ttk.Label(rubric_frame, text=str(val), font=small_font).pack(side='left', padx=(0, 10))

    fields = [
        ("Puzzle Text", 'puzzle_text', 4),
        ("Short Answer", 'shortanswer', 4),
        ("Long Answer", 'longanswer', 8),
        ("Mentor Message", 'r_message', 4)
    ]
    
    # Determine if this is a singleton submission in its thread
    thread_id = entry['thread_id']
    sub_id = entry['submission_id']
    is_singleton_thread = True
    if threads and thread_id in threads:
        submissions = threads[thread_id]['submissions']
        is_singleton_thread = len(submissions) == 1    

    for label, key, height in fields:
        def render_field(label, current_content, previous_content, height, highlight=False):
            ttk.Label(scrollable_frame, text=label, font=('Arial', 10, 'bold')).pack(anchor='w', pady=(10, 0), padx=5)
            text_widget = ScrolledText(scrollable_frame, wrap=tk.WORD, height=height, width=100, font=('DejaVu Sans', 10))

            if highlight:
                text_widget.tag_config('diff', foreground='red')
                sm = SequenceMatcher(None, previous_content.strip(), current_content.strip())
                for tag, i1, i2, j1, j2 in sm.get_opcodes():
                    segment = current_content.strip()[j1:j2]
                    if tag == 'equal':
                        text_widget.insert(tk.END, segment)
                    else:
                        text_widget.insert(tk.END, segment, 'diff')
            else:
                text_widget.insert(tk.END, current_content)

            def block_edit_keys(event):
                if event.state & 0x4 and event.keysym.lower() == 'c':  # Control + C
                    return None  # allow copy
                return "break"

            text_widget.bind("<Key>", block_edit_keys)

            menu = Menu(text_widget, tearoff=0)
            menu.add_command(label="Copy", command=lambda w=text_widget: w.event_generate("<<Copy>>"))

            def show_menu(event):
                text_widget.focus_set()
                index = text_widget.index(f"@{event.x},{event.y}")
                if not text_widget.tag_ranges(tk.SEL):
                    text_widget.tag_remove(tk.SEL, "1.0", tk.END)
                    text_widget.tag_add(tk.SEL, index + " wordstart", index + " wordend")
                    text_widget.mark_set(tk.INSERT, index)
                menu.tk_popup(event.x_root, event.y_root)

            text_widget.bind("<Button-3>", show_menu)
            text_widget.bind("<Control-Button-1>", show_menu)
            text_widget.bind("<Control-c>", lambda e: text_widget.event_generate("<<Copy>>"))

            text_widget.pack(fill=tk.BOTH, expand=True, padx=5)
                    
        current_content = entry.get(key, '')

        # Determine whether to highlight differences
        if is_singleton_thread:
            previous_content = ''
            highlight = False
        elif previous_entry and key in previous_entry and previous_entry[key].strip():
            previous_content = previous_entry[key]
            highlight = True
        else:
            previous_content = ''
            highlight = False

        render_field(label, current_content, previous_content, height, highlight)

    def render_image_buttons():
        def show_image(b64data):
            try:
                img = Image.open(BytesIO(base64.b64decode(b64data)))
                width, height = img.size

                image_window = tk.Toplevel()
                image_window.title("Decoded Image")

                canvas = tk.Canvas(image_window, bg='black')
                canvas.grid(row=0, column=0, sticky='nsew')
                image_window.grid_rowconfigure(0, weight=1)
                image_window.grid_columnconfigure(0, weight=1)

                if width <= IMAGE_WIDTH and height <= IMAGE_HEIGHT:
                    image_window.geometry(f"{width+20}x{height+20}")
                    img_tk = ImageTk.PhotoImage(img)
                    canvas.create_image(0, 0, anchor='nw', image=img_tk)
                    canvas.image = img_tk
                else:
                    img.thumbnail((IMAGE_WIDTH, IMAGE_HEIGHT), RESAMPLE)
                    image_window.geometry(f"{IMAGE_WIDTH+50}x{IMAGE_HEIGHT+50}")

                    hbar = ttk.Scrollbar(image_window, orient=tk.HORIZONTAL, command=canvas.xview)
                    vbar = ttk.Scrollbar(image_window, orient=tk.VERTICAL, command=canvas.yview)
                    hbar.grid(row=1, column=0, sticky='ew')
                    vbar.grid(row=0, column=1, sticky='ns')
                    canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

                    img_tk = ImageTk.PhotoImage(img)
                    canvas.create_image(0, 0, anchor='nw', image=img_tk)
                    canvas.image = img_tk
                    canvas.config(scrollregion=canvas.bbox(tk.ALL))

                    canvas.bind("<ButtonPress-1>", lambda e: canvas.scan_mark(e.x, e.y))
                    canvas.bind("<B1-Motion>", lambda e: canvas.scan_dragto(e.x, e.y, gain=1))

                    def zoom(event):
                        factor = 1.2 if event.delta > 0 else 0.8
                        canvas.scale(tk.ALL, event.x, event.y, factor, factor)
                        canvas.configure(scrollregion=canvas.bbox(tk.ALL))

                    canvas.bind("<MouseWheel>", zoom)
                    canvas.bind("<Button-4>", lambda e: zoom(type('Event', (), {'delta': 120})))
                    canvas.bind("<Button-5>", lambda e: zoom(type('Event', (), {'delta': -120})))
            except Exception as e:
                messagebox.showerror("Image Error", str(e))

        for field_label, idx, b64_data in entry['b64_images']:
            def create_button(fl=field_label, i=idx, b64=b64_data):
                btn = ttk.Button(scrollable_frame, text=f"Show {fl} Image {i}", command=lambda b=b64: show_image(b))
                btn.pack(pady=2, padx=5, anchor='w')

            scrollable_frame.after(0, create_button)

    if entry['b64_images']:
        render_image_buttons()

    detail.update_idletasks()
    content_width = scrollable_frame.winfo_reqwidth() + 40
    content_height = scrollable_frame.winfo_reqheight() + 70

    window_width = min(content_width, DETAIL_WIDTH)
    window_height = min(content_height, DETAIL_HEIGHT)
    detail.geometry(f"{window_width}x{window_height}")

def create_gui():
    root = tk.Tk()
    root.title("Math Forum Submissions")

    filter_frame = ttk.Frame(root)
    filter_frame.pack(fill=tk.X)

    filter_vars = {
        'ID': tk.StringVar(),
        'Puzzle': tk.StringVar(),
        'Short Answer': tk.StringVar(),
        'Images (Y/N)': tk.StringVar(),
        'Min Score': tk.StringVar(),
        'Filled (Y/N)': tk.StringVar()
    }

    for idx, (key, var) in enumerate(filter_vars.items()):
        ttk.Label(filter_frame, text=key).grid(row=idx // 4, column=(idx % 4) * 2, padx=2, pady=2)
        ttk.Entry(filter_frame, textvariable=var, width=15).grid(row=idx // 4, column=(idx % 4) * 2 + 1, padx=2, pady=2)

    note_label = ttk.Label(root, text="Note: 'Filled' means a short answer, long answer, and mentor response are all present.", foreground='blue')
    note_label.pack(anchor='w', padx=5, pady=(5, 0))

    container = ttk.Frame(root)
    container.pack(fill=tk.BOTH, expand=True)

    tree_scrollbar = ttk.Scrollbar(container, orient="vertical")
    tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    tree = ttk.Treeview(container, columns=(
        'ID', 'Puzzle', 'Short Answer', 'Length', 'Images (Y/N)', 'Total Score', 'Filled (Y/N)'
    ), show='tree headings', yscrollcommand=tree_scrollbar.set)

    for col in ('ID', 'Puzzle', 'Short Answer', 'Length', 'Images (Y/N)', 'Total Score', 'Filled (Y/N)'):
        tree.heading(col, text=col)

    tree_scrollbar.config(command=tree.yview)
    tree.pack(fill=tk.BOTH, expand=True)

    threads = get_threads_and_submissions()

    def prefetch_visible_submissions(tree):
        print(f"[Prefetch] Starting Prefetcher")
        
        root.update_idletasks()  # Ensure layout is updated

        visible_sub_ids = []
        already_prefetched = set()
        count = 0
        
        def estimate_visible_items(tree, max_rows=30):
            """Estimate a slice of visible item IDs in the treeview."""
            # Fraction of scrollbar from 0.0 to 1.0
            top_frac, bottom_frac = tree.yview()

            all_items = []
            for item in tree.get_children():
                children = tree.get_children(item)
                if children:
                    all_items.extend(children)
                else:
                    all_items.append(item)

            total = len(all_items)
            if total == 0:
                return []

            # Convert fractional scroll into index range
            start_idx = int(top_frac * total)
            end_idx = int(bottom_frac * total)
            end_idx = min(end_idx + 1, total)

            # Optional: limit to a maximum number of rows (prevent huge spikes)
            if end_idx - start_idx > max_rows:
                end_idx = start_idx + max_rows
                print(f"Prefetch limiting rows to {start_idx} to {end_idx} due to large viewable area of {top_frac} to {bottom_frac}")
            else:
                print(f"Prefetching {start_idx} to {end_idx} from viewable area of {top_frac} to {bottom_frac}")

            return all_items[start_idx:end_idx]
        
        for item in estimate_visible_items(tree):
            children = tree.get_children(item)
            target_items = children if children else [item]

            for child in target_items:
                bbox = tree.bbox(child)
                if bbox is None:
                    continue

                if bbox and len(bbox) >= 4:
                    y, height = bbox[1], bbox[3]
                    if 0 <= y <= tree.winfo_height():                    
                        y, height = bbox[1], bbox[3]
                        if (0 <= y <= tree.winfo_height()) or (0 <= y + height <= tree.winfo_height()):
                            values = tree.item(child, 'values')
                            if values and values[0].isdigit():
                                sub_id = int(values[0])
                                if sub_id not in submission_cache and sub_id not in already_prefetched:
                                    visible_sub_ids.append(sub_id)
                                    already_prefetched.add(sub_id)

        if visible_sub_ids:
            def background():
                for sid in visible_sub_ids:
                    if prefetch_abort_flag.is_set():
                        print(f"[Prefetch] Paused due to abort flag (waiting to resume at {sid})")
                        
                        while prefetch_abort_flag.is_set():
                            time.sleep(1)  # Yield the processor
                            
                        print(f"[Prefetch] Resuming from cleared abort flag (resuming at {sid})")
                        
                    print(f"[Prefetch] Fetching submission {sid}")
                    fetch_full_submission(sid)
                    print(f"[Prefetch] Fetched submission {sid}")

            threading.Thread(target=background, daemon=True).start()
                
    def update_tree():
        global submission_cache 
    
        tree.delete(*tree.get_children())
        for thread_id, thread in threads.items():
            puzzle_text = thread['puzzle_text']
            submissions = thread['submissions']
            filtered_subs = []
            for sub in submissions:
                total_score = sum(v for v in sub['rubric'].values() if isinstance(v, (int, float)))
                has_filled = 'Y' if all([sub['shortanswer'], sub['longanswer'], sub['r_message']]) else 'N'
                has_images = 'Y' if sub['b64_images'] else 'N'

                if (filter_vars['ID'].get() and filter_vars['ID'].get() not in str(sub['submission_id'])) or \
                   (filter_vars['Puzzle'].get() and filter_vars['Puzzle'].get().lower() not in puzzle_text.lower()) or \
                   (filter_vars['Short Answer'].get() and filter_vars['Short Answer'].get().lower() not in sub['shortanswer'].lower()) or \
                   (filter_vars['Images (Y/N)'].get() and filter_vars['Images (Y/N)'].get().upper() != has_images):
                    continue
                if filter_vars['Min Score'].get():
                    try:
                        if total_score < float(filter_vars['Min Score'].get()):
                            continue
                    except ValueError:
                        continue
                if filter_vars['Filled (Y/N)'].get().upper() == 'Y':
                    if has_filled != 'Y':
                        continue
                elif filter_vars['Filled (Y/N)'].get().upper() == 'N':
                    if has_filled != 'N':
                        continue
                filtered_subs.append(sub)

            if not filtered_subs:
                continue

            parent = ''
            if len(filtered_subs) > 1:
                parent = tree.insert('', tk.END, values=(f"Thread {thread_id}", puzzle_text[:50], '', '', '', '', ''), tags=(f"thread_{thread_id}",))

            for sub in filtered_subs:
                total_score = sum(v for v in sub['rubric'].values() if isinstance(v, (int, float)))
                has_filled = 'Y' if all([sub['shortanswer'], sub['longanswer'], sub['r_message']]) else 'N'
                has_images = 'Y' if sub['b64_images'] else 'N'

                tree.insert(parent, tk.END, values=(
                    sub['submission_id'],
                    puzzle_text[:50] + '...' if len(puzzle_text) > 50 else puzzle_text,
                    sub['shortanswer'][:50] + '...' if len(sub['shortanswer']) > 50 else sub['shortanswer'],
                    len(sub.get('longanswer', 0)),
                    has_images,
                    total_score,
                    has_filled
                ), tags=(f"submission_{sub['submission_id']}",))
                
        prefetch_pending = [None]  # use a list to capture mutable reference

        def schedule_prefetch():
            if prefetch_pending[0] is not None:
                return  # already scheduled

            def wrapped():
                prefetch_visible_submissions(tree)
                prefetch_pending[0] = None

            prefetch_pending[0] = root.after(200, wrapped)  # delay in ms   

        # Resize
        tree.bind('<Configure>', lambda e: schedule_prefetch())

        # Mouse Wheel (Linux/Windows)
        tree.bind('<MouseWheel>', lambda e: schedule_prefetch())  # Windows
        tree.bind('<Button-4>', lambda e: schedule_prefetch())    # Linux up
        tree.bind('<Button-5>', lambda e: schedule_prefetch())    # Linux down

        # Arrow keys and Page Up/Down keys and Home/End keys trigger prefetch
        tree.bind('<Up>', lambda e: schedule_prefetch())
        tree.bind('<Down>', lambda e: schedule_prefetch())
        tree.bind('<Prior>', lambda e: schedule_prefetch())  # Page Up
        tree.bind('<Next>', lambda e: schedule_prefetch())   # Page Down            
        tree.bind('<Home>', lambda e: schedule_prefetch())
        tree.bind('<End>', lambda e: schedule_prefetch())        
        
        # Scrollbar moves
        tree_scrollbar.config(command=lambda *args: (tree.yview(*args), schedule_prefetch()))
            
    for var in filter_vars.values():
        var.trace_add('write', lambda *args: update_tree())

    update_tree()
    root.after_idle(lambda: prefetch_visible_submissions(tree))

    def on_click(event):
        root.config(cursor='watch')
        root.update_idletasks()
        
        prefetch_abort_flag.set()  # Interrupt any prefetching

        item_id = tree.identify_row(event.y)
        if not item_id:
            root.config(cursor='')  # Ensure cursor resets even on early return
            return

        values = tree.item(item_id, 'values')
        if not values or not values[0].isdigit():
            root.config(cursor='')
            return  # Clicked a thread row or invalid item

        sub_id = int(values[0])
        entry = fetch_full_submission(sub_id)

        if entry:
            previous_entry = fetch_previous_submission(entry['thread_id'], entry['submission_id'])
            render_detail_window(entry, previous_entry, threads)
        else:
            print(f"Submission ID {sub_id} not found.")

        prefetch_abort_flag.clear()  # Reset the flag after completing priority task
        
        root.config(cursor='')

    tree.bind('<Double-1>', on_click)
    
    def on_row_highlight(event):
        root.config(cursor='watch')
        root.update_idletasks()
        
        prefetch_abort_flag.set()  # Interrupt any prefetching
        
        selected = tree.selection()
        ids_to_fetch = []

        for item in selected:
            values = tree.item(item, 'values')
            if values and values[0].isdigit():
                sub_id = int(values[0])
                ids_to_fetch.append(sub_id)

        if not ids_to_fetch:
            root.config(cursor='')  # Reset cursor immediately if no work
            return

        def worker():
            for sub_id in ids_to_fetch:
                fetch_full_submission(sub_id)
                
            # Schedule the cursor reset on the main thread after fetching completes
            root.after(0, lambda: root.config(cursor=''))   

            prefetch_abort_flag.clear()  # Reset the flag after completing priority task

        threading.Thread(target=worker, daemon=True).start()
                 
    tree.bind("<<TreeviewSelect>>", on_row_highlight)
    
    def on_thread_expand(event):
        root.config(cursor='watch')
        root.update_idletasks()
        
        item_id = tree.focus()
        ids_to_fetch = []

        for child in tree.get_children(item_id):
            values = tree.item(child, 'values')
            if values and values[0].isdigit():
                sub_id = int(values[0])
                ids_to_fetch.append(sub_id)

        if not ids_to_fetch:
            root.config(cursor='')  # Reset cursor immediately if no work            
            return

        def worker():
            for sub_id in ids_to_fetch:
                fetch_full_submission(sub_id)
            
            # Schedule the cursor reset on the main thread after fetching completes
            root.after(0, lambda: root.config(cursor=''))                                

        threading.Thread(target=worker, daemon=True).start()
    
    tree.bind("<<TreeviewOpen>>", on_thread_expand)
    
    root.mainloop()

if __name__ == "__main__":
    create_gui()
