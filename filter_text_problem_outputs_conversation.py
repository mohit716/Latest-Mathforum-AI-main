import zipfile
import json
import re
import os
from typing import Any, Dict, Callable, Union, List
from tqdm import tqdm
import pandas as pd
import pprint
import difflib
import html
import traceback

INPUT_ZIP = "text_problem_outputs.zip"
OUTPUT_ZIP = "filtered_text_problem_outputs.zip"
FILTER_CONFIG_FILE = "filter_text_problem_outputs_conversation.json"
CSV_MAIN_OUTPUT_FILE = "text_problem_output.csv"
CSV_CONVERSATION_OUTPUT_FILE = "text_problem_conversation_output.csv"

def safe_decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    elif isinstance(value, str):
        return value
    else:
        return str(value)

def sanitize(text: str) -> str:
    if not text:
        return ''
        
    text = safe_decode(text)    

    # === Remove embedded base64 strings and HTML placeholders ===
    text = re.sub(r'data:[^;]+;base64,[A-Za-z0-9+/=\s]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[[^\[\]]*?Image \d+[^\[\]]*?\]', '', text, flags=re.IGNORECASE)

    # === Remove C-style and Word-style comment blocks ===
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)  # Word XML/HTML comments

    # === Remove Word XML tags: <w:*, <o:*, <v:*>, <m:*> etc. ===
    text = re.sub(r'</?(w|o|v|m|xml|st1):[^>]+>', '', text, flags=re.IGNORECASE)

    # === Remove general XML/HTML tags (fallthrough) ===
    text = re.sub(r'<[^>]+>', '', text, flags=re.DOTALL)

    # === Remove MS Office field codes, bookmarks, headers ===
    text = re.sub(r'\{\\\*[^{}]*\}', '', text)  # Word field codes like {\*\fldinst ...}
    text = re.sub(r'\{\\\w+[^{}]*\}', '', text) # General RTF-style control groups

    # === Remove MS Word SmartTag references ===
    text = re.sub(r'<st1:[^>]+>.*?</st1:[^>]+>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove Word "Smart HTML" patterns
    text = re.sub(r"Normal 0 false false false.*?MicrosoftInternetExplorer4", "", text, flags=re.DOTALL)
    text = re.sub(r"mso-[\w\-]+:[^;]+;", "", text)  # Microsoft Office-specific CSS
    text = re.sub(r"table\.MsoNormalTable\s*\{[^}]+\}", "", text, flags=re.DOTALL)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)  # CSS block comments    
    text = re.sub(r'Normal\.dotm.*?false false false', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'mso-[^:]+:[^;"]+;?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'class=["\']?Mso\w*["\']?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<(span|div)[^>]*mso-[^>]*>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # === Remove style and script blocks ===
    text = re.sub(r'<style[^>]*?>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<script[^>]*?>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # === HTML entity unescaping ===
    text = html.unescape(text)

    # === Unicode escape decoding and smart punctuation normalization ===
    text = text.encode('utf-8').decode('unicode_escape', errors='backslashreplace')
    text = text.replace('\u00a0', ' ')   # non-breaking space
    text = text.replace('\u2013', '-')   # en dash
    text = text.replace('\u2014', '-')   # em dash
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # single quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # double quotes

    # === Remove control characters (non-printables) ===
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)

    # === Normalize whitespace ===
    text = text.replace('\r\n', '\n').replace('\r', '\n')  # normalize newlines
    text = re.sub(r'\n{3,}', '\n\n', text)                 # collapse blank lines
    text = re.sub(r'[ \t]+', ' ', text)                    # collapse repeated spaces
    text = re.sub(r'[ ]*\n[ ]*', '\n', text)               # strip lines

    return text.strip()
    
def load_filters_from_json(filepath: str) -> Dict[str, Dict[str, Any]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_filter_function(field_rules: Dict[str, Any]) -> Callable[[Any], bool]:
    def filter_func(value: Union[str, int, float]) -> bool:
        try:
            if 'improvement' in field_rules:
                return True  # Handled elsewhere

            num_value = None
            if isinstance(value, str):
                stripped = value.strip()
                num_value = (
                    int(stripped)
                    if stripped.isdigit()
                    else float(stripped)
                    if re.match(r'^-?\d+(\.\d+)?$', stripped)
                    else None
                )

            if 'min_length' in field_rules:
                if not isinstance(value, str) or len(value.strip()) < field_rules['min_length']:
                    return False

            if 'max_length' in field_rules:
                if not isinstance(value, str) or len(value.strip()) > field_rules['max_length']:
                    return False

            if 'min_value' in field_rules:
                if num_value is None or num_value < field_rules['min_value']:
                    return False

            if 'max_value' in field_rules:
                if num_value is None or num_value > field_rules['max_value']:
                    return False

            if 'contains' in field_rules and isinstance(value, str):
                must_contain = field_rules['contains']
                if isinstance(must_contain, str):
                    must_contain = [must_contain]
                if not any(substr.lower() in value.lower() for substr in must_contain):
                    return False

            if 'not_contains' in field_rules and isinstance(value, str):
                must_not_contain = field_rules['not_contains']
                if isinstance(must_not_contain, str):
                    must_not_contain = [must_not_contain]
                if any(substr.lower() in value.lower() for substr in must_not_contain):
                    return False

            if 'regex' in field_rules:
                if not re.search(field_rules['regex'], str(value)):
                    return False

            if field_rules.get('required') is True:
                if value is None or (isinstance(value, str) and not value.strip()):
                    return False

            return True
        except Exception as e:
            print(f"Exception in build_filter_function for value '{value}': {e}")
            traceback.print_exc()
            return False

    return filter_func

# Check if the max value in the conversation is above the threshold to pass
def should_keep(json_data: Dict[str, Any],
                filter_funcs: Dict[str, Callable[[Any], bool]],
                raw_filters: Dict[str, Dict[str, Any]]) -> bool:
    conversation = json_data.get("conversation", [])
    passed_all = True
    
    # Synthetic field: mentor_replies
    if "mentor_replies" in raw_filters:
        rule = raw_filters["mentor_replies"]
        mentor_reply_count = sum(
            1 for entry in conversation
            if isinstance(entry.get("response", ""), str) and entry.get("response", "").strip()
            and not entry.get("is_synthetic", False)
        )
        try:
            if "min_value" in rule and mentor_reply_count < rule["min_value"]:
                passed_all = False
            if "max_value" in rule and mentor_reply_count > rule["max_value"]:
                passed_all = False
        except Exception as e:
            print(f"Error in mentor_replies filter: {e}")
            traceback.print_exc()
            passed_all = False       

    for field, func in filter_funcs.items():
        field_value = json_data.get(field)
        rule = raw_filters.get(field, {})

        # Top-level field
        if field_value is not None:
            if not func(field_value):
                passed_all = False
            continue

        # Gather values from conversation or rubrics
        values = []
        for entry in conversation:
            val = entry.get(field)
            if val is None:
                val = entry.get("rubrics", {}).get(field)
            if val is not None:
                values.append(val)

        if not values:
            passed_all = False
            continue

        # Improvement logic
        if rule.get("improvement") is not None:
            try:
                numeric_vals = [float(v) for v in values if isinstance(v, (int, float)) or re.match(r'^-?\d+(\.\d+)?$', str(v))]
                if len(numeric_vals) < 2:
                    passed_all = False
                    continue
                improved = any(
                    later - earlier >= rule["improvement"]
                    for i, earlier in enumerate(numeric_vals)
                    for later in numeric_vals[i+1:]
                )
                if not improved:
                    passed_all = False
            except Exception as e:
                print(f"Error in improvement check for '{field}': {e}")
                traceback.print_exc()
                passed_all = False

        # Other rules
        try:
            if 'min_value' in rule:
                numeric_vals = [float(v) for v in values if isinstance(v, (int, float)) or re.match(r'^-?\d+(\.\d+)?$', str(v))]
                if not numeric_vals or max(numeric_vals) < rule['min_value']:
                    passed_all = False

            if 'max_value' in rule:
                numeric_vals = [float(v) for v in values if isinstance(v, (int, float)) or re.match(r'^-?\d+(\.\d+)?$', str(v))]
                if not numeric_vals or min(numeric_vals) > rule['max_value']:
                    passed_all = False

            if 'min_length' in rule:
                str_lengths = [len(str(v).strip()) for v in values if isinstance(v, str)]
                if not str_lengths or max(str_lengths) < rule['min_length']:
                    passed_all = False

            if 'max_length' in rule:
                str_lengths = [len(str(v).strip()) for v in values if isinstance(v, str)]
                if not str_lengths or min(str_lengths) > rule['max_length']:
                    passed_all = False

            if 'contains' in rule:
                substrings = rule['contains']
                if isinstance(substrings, str):
                    substrings = [substrings]
                if not any(any(substr.lower() in str(v).lower() for substr in substrings)
                           for v in values if isinstance(v, str)):
                    passed_all = False

            if 'not_contains' in rule:
                substrings = rule['not_contains']
                if isinstance(substrings, str):
                    substrings = [substrings]
                if all(any(substr.lower() in str(v).lower() for substr in substrings)
                       for v in values if isinstance(v, str)):
                    passed_all = False

            if 'regex' in rule:
                pattern = re.compile(rule['regex'])
                if not any(pattern.search(str(v)) for v in values):
                    passed_all = False

            if rule.get("required") is True:
                if all((v is None or (isinstance(v, str) and not v.strip())) for v in values):
                    passed_all = False

        except Exception as e:
            print(f"Exception in should_keep for field '{field}': {e}")
            traceback.print_exc()
            passed_all = False

    return passed_all
    
def compute_diff(a: str, b: str, context: int = 3) -> str:
    """Compute a more readable word-level diff with surrounding context."""
    if not a and not b:
        return ""

    a_words = a.split()
    b_words = b.split()

    sm = difflib.SequenceMatcher(None, a_words, b_words)
    result = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            if context > 0 and (i2 - i1) > context * 2:
                result.extend(a_words[i1:i1 + context])
                result.append("...")
                result.extend(a_words[i2 - context:i2])
            else:
                result.extend(a_words[i1:i2])
        elif tag == 'replace':
            result.append(f"[~~{' '.join(a_words[i1:i2])}~~ ➝ **{' '.join(b_words[j1:j2])}**]")
        elif tag == 'delete':
            result.append(f"[~~{' '.join(a_words[i1:i2])}~~]")
        elif tag == 'insert':
            result.append(f"[**{' '.join(b_words[j1:j2])}**]")

    return ' '.join(result)    

def compute_rubric_diff(prev: Any, curr: Any) -> str:
    """Compute rubric score difference as +N or -N."""
    try:
        if prev is None or curr is None:
            return ""
        prev_val = int(prev)
        curr_val = int(curr)
        delta = curr_val - prev_val
        sign = "+" if delta >= 0 else "-"
        return f"{sign}{abs(delta)}"
    except ValueError as e:
        print(f"ValueError Exception in compute_rubric_diff: {e}")
        traceback.print_exc()
        return ""
    except TypeError as e:
        print(f"TypeError Exception in compute_rubric_diff: {e}")
        traceback.print_exc()
        return ""

def passes_rubric_filter(json_data: Dict[str, Any],
                         rubric: str,
                         rule: Dict[str, Any]) -> bool:
    """
    Check if a given rubric field passes its associated filter rules.
    Supports min_value, max_value, improvement.
    """
    try:
        conversation = json_data.get("conversation", [])
        thread_id = json_data.get("thread_id", "UNKNOWN")

        values = []
        for entry in conversation:
            rubrics = entry.get("rubrics", {})
            raw_val = rubrics.get(rubric)
            if raw_val is None:
                continue
            try:
                val = float(raw_val)
                values.append(val)
            except (ValueError, TypeError):
                print(f"[{thread_id}] Skipping non-numeric rubric '{rubric}' value: {raw_val!r}")
                continue

        if not values:
            return False

        if "improvement" in rule:
            improvement_threshold = rule["improvement"]
            if len(values) < 2:
                return False
            improved = any(
                later - earlier >= improvement_threshold
                for i, earlier in enumerate(values)
                for later in values[i + 1:]
            )
            if not improved:
                return False

        if "min_value" in rule:
            max_val = max(values)
            if max_val < rule["min_value"]:
                return False

        if "max_value" in rule:
            min_val = min(values)
            if min_val > rule["max_value"]:
                return False

        return True

    except Exception as e:
        print(f"Exception in passes_rubric_filter for rubric '{rubric}': {e}")
        traceback.print_exc()
        return False

def process_zip(input_zip: str, output_zip: str, csv_main_output_file: str, csv_conversation_output_file: str,
                filter_funcs: Dict[str, Callable[[Any], bool]],
                raw_filters: Dict[str, Dict[str, Any]]):
    
    rubric_txt_matches: Dict[str, List[Tuple[str, str]]] = {rubric: [] for rubric, rules in raw_filters.items() if rules.get("is_rubric")}
    rubric_wide_records: Dict[str, List[Dict[str, Any]]] = {rubric: [] for rubric in rubric_txt_matches}
    rubric_counters: Dict[str, int] = {rubric: 0 for rubric in rubric_txt_matches}

    wide_conversation_records = []
    main_json_records = []

    rubric_fields = set(rubric_txt_matches.keys())

    non_rubric_filter_funcs = {
        field: func for field, func in filter_funcs.items()
        if field not in rubric_fields and not raw_filters[field].get("is_synthetic", False)
    }

    with zipfile.ZipFile(input_zip, 'r') as zin:
        items = zin.infolist()
        with zipfile.ZipFile(output_zip, 'w') as zout:
            json_items = [item for item in items if item.filename.endswith('.json')]
            progress_bar = tqdm(json_items, desc="Filtering submissions", unit="submission")
            for item in progress_bar:
                filename = item.filename
                try:
                    with zin.open(item) as f:
                        json_bytes = f.read()
                        json_data = json.loads(json_bytes.decode('utf-8'))

                    thread_id = json_data.get("thread_id")
                    puzzle_text = sanitize(json_data.get("puzzle_text", ""))
                    conversation = json_data.get("conversation", [])

                    wide_row = {
                        "thread_id": thread_id,
                        "puzzle_text": puzzle_text,
                        "student_name": json_data.get("student_name", ""),
                        "school_name": json_data.get("school_name", ""),
                        "mentor_name": json_data.get("mentor_name", ""),
                        "age": json_data.get("age", "")
                    }

                    for idx, entry in enumerate(conversation):
                        prefix = f"msg_{idx}_"
                        prior_prefix = f"msg_{idx-1}_" if idx > 0 else None

                        short_answer = sanitize(entry.get("short_answer", ""))
                        long_answer = sanitize(entry.get("long_answer", ""))
                        response = sanitize(entry.get("response", ""))

                        wide_row[f"{prefix}submission_id"] = entry.get("submission_id")
                        wide_row[f"{prefix}submission_date"] = entry.get("submission_date")
                        wide_row[f"{prefix}short_answer"] = short_answer
                        wide_row[f"{prefix}long_answer"] = long_answer
                        wide_row[f"{prefix}response"] = response
                        wide_row[f"{prefix}response_date"] = entry.get("response_date")

                        if prior_prefix:
                            wide_row[f"{prefix}short_answer_diff"] = compute_diff(
                                wide_row.get(f"{prior_prefix}short_answer", ""), short_answer
                            )
                            wide_row[f"{prefix}long_answer_diff"] = compute_diff(
                                wide_row.get(f"{prior_prefix}long_answer", ""), long_answer
                            )
                            wide_row[f"{prefix}response_diff"] = compute_diff(
                                wide_row.get(f"{prior_prefix}response", ""), response
                            )

                        rubrics = entry.get("rubrics", {})
                        for rubric_key, rubric_val in rubrics.items():
                            wide_row[f"{prefix}rubrics_{rubric_key}"] = rubric_val
                            if prior_prefix:
                                prev_val = wide_row.get(f"{prior_prefix}rubrics_{rubric_key}")
                                wide_row[f"{prefix}rubrics_{rubric_key}_diff"] = compute_rubric_diff(prev_val, rubric_val)

                    wide_conversation_records.append(wide_row)

                    # === MAIN FILTER ===
                    keep_for_main = should_keep(json_data, filter_funcs, raw_filters)

                    # === RUBRIC FILTERS (before mutating json_data) ===
                    for rubric in rubric_fields:
                        if rubric in raw_filters:
                            rubric_rules = raw_filters[rubric]
                            if passes_rubric_filter(json_data, rubric, rubric_rules):
                                passes_non_rubric = should_keep(json_data, non_rubric_filter_funcs, raw_filters)
                                if passes_non_rubric:
                                    txt_filename = filename.replace('.json', '.txt')
                                    try:
                                        txt_data = zin.read(txt_filename)
                                        txt_data = sanitize(txt_data)
                                        rubric_txt_matches[rubric].append((txt_filename, txt_data))
                                        rubric_wide_records[rubric].append(wide_row.copy())
                                        rubric_counters[rubric] += 1
                                    except KeyError:
                                        continue

                    # === MAIN ZIP WRITING (now safe to mutate json_data) ===
                    if keep_for_main:
                        formatted_conversation = pprint.pformat(conversation, width=100, compact=True)
                        json_data["conversation_text"] = formatted_conversation
                        json_data.pop("conversation", None)
                        main_json_records.append(wide_row)

                        zout.writestr(item, json_bytes)
                        txt_filename = filename.replace('.json', '.txt')
                        try:
                            txt_data = zin.read(txt_filename)
                            txt_data = sanitize(txt_data)
                            zout.writestr(txt_filename, txt_data)
                        except KeyError:
                            print(f"Warning: TXT file not found for {filename}")

                    # === Live rubric counter update ===
                    rubric_postfix = ", ".join(f"{k[:4]}: {rubric_counters[k]}" for k in sorted(rubric_counters))
                    progress_bar.set_postfix_str(f"Main: {len(main_json_records)} | {rubric_postfix}")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    traceback.print_exc()

    print(f"\nFiltered ZIP written to: {output_zip}")

    # Write rubric-specific ZIPs
    for rubric, txt_entries in rubric_txt_matches.items():
        rubric_zip_path = f"{rubric}_{output_zip}"
        with zipfile.ZipFile(rubric_zip_path, 'w') as rzip:
            for txt_filename, txt_data in tqdm(txt_entries, desc=f"Writing {rubric}", unit="file", leave=False):
                rzip.writestr(txt_filename, txt_data)
        print(f"Rubric ZIP written to: {rubric_zip_path}")

    # Save CSVs
    total_csvs = len(rubric_wide_records) + 2
    pbar = tqdm(total=total_csvs, desc="Writing CSVs", unit="file")

    if wide_conversation_records:
        pd.DataFrame(wide_conversation_records).to_csv(csv_conversation_output_file, index=False)
    pbar.update(1)

    if main_json_records:
        pd.DataFrame(main_json_records).to_csv(csv_main_output_file, index=False)
    pbar.update(1)

    for rubric, data in rubric_wide_records.items():
        if data:
            filename = f"{rubric}_{csv_conversation_output_file}"
            pd.DataFrame(data).to_csv(filename, index=False)
        pbar.update(1)

    pbar.close()
    print("\nCSV files written for main filter, conversation details, and rubric-specific outputs.")

# ------------------ Main Program ------------------

if __name__ == "__main__":
    raw_filters = load_filters_from_json(FILTER_CONFIG_FILE)
    filter_funcs = {
        field: build_filter_function(rules)
        for field, rules in raw_filters.items()
        if not rules.get("is_synthetic", False)
    }

    process_zip(INPUT_ZIP, OUTPUT_ZIP, CSV_MAIN_OUTPUT_FILE, CSV_CONVERSATION_OUTPUT_FILE, filter_funcs, raw_filters)
