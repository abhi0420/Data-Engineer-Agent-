from config.model_config import get_llm
from langchain.agents import create_agent
from langchain.tools import tool
from bigquery_source import BigQuerySource
from google.cloud import bigquery as bq
from dotenv import load_dotenv
import os
import json
import warnings
import jinja2
from markupsafe import Markup
from langchain_community.callbacks import get_openai_callback

warnings.filterwarnings("ignore")
load_dotenv()

model = get_llm()

# ─── Thresholds ───────────────────────────────────────────────────────────────
NULL_WARN_PCT   = 5.0    # warn if null rate > 5%
NULL_FAIL_PCT   = 15.0   # fail if null rate > 15%
UNIQUE_WARN_PCT = 95.0   # warn if uniqueness < 95%

# Forbidden keywords in LLM-generated SQL (DML/DDL safety check)
_FORBIDDEN_SQL = {"insert", "update", "delete", "drop", "truncate", "merge",
                  "create", "alter", "call", "execute", "exec"}


# Severity check for metrics, what is higher_is_worse?
def _severity(value: float, warn: float, fail: float, higher_is_worse: bool = True) -> str:
    if higher_is_worse:
        if value >= fail:  return "FAIL"
        if value >= warn:  return "WARN"
    else:
        if value <= fail:  return "FAIL"
        if value <= warn:  return "WARN"
    return "PASS"

# Scans the SQL query for forbidden keywords
def _safe_sql(sql: str) -> bool:
    """Returns True if SQL only contains SELECT/WITH statements."""
    first_word = sql.strip().split()[0].lower() if sql.strip() else ""
    return first_word in ("select", "with") and not any(
        kw in sql.lower() for kw in _FORBIDDEN_SQL
    )


# ─── Tool 1: discover_schema ──────────────────────────────────────────────────

@tool
def discover_schema(project_id: str, dataset_id: str, table_id: str) -> str:
    """Fetches column names, types, modes from INFORMATION_SCHEMA, table size
    metadata, and up to 3 sample rows. Call this first before any other DQ tool.

    Required parameters:
        - project_id:  GCP project ID
        - dataset_id:  BigQuery dataset name
        - table_id:    BigQuery table name
    """
    try:
        bq_obj = BigQuerySource(project_id)

        schema_sql = f"""
            SELECT column_name, data_type, is_nullable
            FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{table_id}'
            ORDER BY ordinal_position
        """
        schema_df = bq_obj.query_or_raise(schema_sql)

        if schema_df.empty:
            return f"ERROR: Table `{dataset_id}.{table_id}` not found or has no columns."

        # Fetch the approximate row count and size from __TABLES__ metadata
        size_sql = f"""
            SELECT row_count, ROUND(size_bytes / POW(1024, 3), 3) AS size_gb
            FROM `{project_id}.{dataset_id}.__TABLES__`
            WHERE table_id = '{table_id}'
        """
        try:
            size_df          = bq_obj.query_or_raise(size_sql)
            approx_row_count = int(size_df.iloc[0]["row_count"]) if not size_df.empty else None
            table_size_gb    = float(size_df.iloc[0]["size_gb"])  if not size_df.empty else None
        except Exception:
            approx_row_count = None
            table_size_gb    = None

        sample_sql = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}` LIMIT 3"
        sample_df  = bq_obj.query_or_raise(sample_sql)

        columns = schema_df.to_dict(orient="records")
        sample  = sample_df.to_dict(orient="records")

        return json.dumps({
            "table":            f"{dataset_id}.{table_id}",
            "columns":          columns,
            "approx_row_count": approx_row_count,
            "table_size_gb":    table_size_gb,
            "sample":           sample,
        }, default=str)

    except Exception as e:
        return f"ERROR: Could not fetch schema — {str(e)}"


# ─── Tool 2: run_standard_checks ─────────────────────────────────────────────

@tool
def run_standard_checks(project_id: str, dataset_id: str, table_id: str) -> str:
    """Runs standard DQ checks in a SINGLE BigQuery scan:
    - Row count, null rate, completeness, APPROX_COUNT_DISTINCT per column
    - MIN / MAX / AVG for numeric columns
    - MAX timestamp for columns ending in _at / _date / _time (freshness)

    For large tables (>50 GB) uses TABLESAMPLE to control cost and marks
    results as approximate.

    Required parameters:
        - project_id:  GCP project ID
        - dataset_id:  BigQuery dataset name
        - table_id:    BigQuery table name
    """
    try:
        bq_obj = BigQuerySource(project_id)

        # Fetch column schema internally
        schema_sql = f"""
            SELECT column_name, data_type
            FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{table_id}'
            ORDER BY ordinal_position
        """
        schema_df = bq_obj.query_or_raise(schema_sql)
        if schema_df.empty:
            return "ERROR: Table not found or has no columns."

        col_names     = list(schema_df["column_name"])
        col_types     = dict(zip(schema_df["column_name"], schema_df["data_type"].str.upper()))
        numeric_types = {"INT64", "INTEGER", "FLOAT64", "FLOAT", "NUMERIC", "BIGNUMERIC"}
        ts_suffixes   = ("_at", "_date", "_time", "_timestamp")

        # Build SELECT expressions — one scan
        exprs = ["COUNT(*) AS _total_rows"]
        for col in col_names:
            safe_col = f"`{col}`"
            exprs.append(f"COUNTIF({safe_col} IS NULL) AS `_null_{col}`")
            exprs.append(f"APPROX_COUNT_DISTINCT({safe_col}) AS `_distinct_{col}`")
            if col_types.get(col, "") in numeric_types:
                exprs.append(f"MIN({safe_col}) AS `_min_{col}`")
                exprs.append(f"MAX({safe_col}) AS `_max_{col}`")
                exprs.append(f"AVG(CAST({safe_col} AS FLOAT64)) AS `_avg_{col}`")
            if col.lower().endswith(ts_suffixes):
                exprs.append(f"MAX({safe_col}) AS `_max_ts_{col}`")

        table_ref = f"`{project_id}.{dataset_id}.{table_id}`"

        # Generate full SQL for standard checks
        sql_full  = f"SELECT {', '.join(exprs)} FROM {table_ref}"

        # Dry-run cost check
        job_config = bq.QueryJobConfig(dry_run=True, use_query_cache=False)
        dry_job    = bq_obj.client.query(sql_full, job_config=job_config)
        gb_est     = (dry_job.total_bytes_processed or 0) / (1024 ** 3)

        if gb_est > 50:
            sample_pct  = 1 if gb_est > 500 else 10
            sql         = f"SELECT {', '.join(exprs)} FROM {table_ref} TABLESAMPLE SYSTEM ({sample_pct} PERCENT)"
            approximate = True
            sample_note = f"Approximate results — {sample_pct}% TABLESAMPLE used (table is {gb_est:.0f} GB)"
        else:
            sql         = sql_full
            approximate = False
            sample_note = None

        result = bq_obj.query_or_raise(sql)
        row    = result.iloc[0].to_dict()
        total  = int(row.get("_total_rows", 0))

        results = {
            "total_rows":   total,
            "approximate":  approximate,
            "sample_note":  sample_note,
            "estimated_gb": round(gb_est, 3),
            "columns": {},
        }

        for col in col_names:
            null_count = int(row.get(f"_null_{col}", 0) or 0)
            distinct   = int(row.get(f"_distinct_{col}", 0) or 0)
            null_pct   = round((null_count / total * 100), 2) if total > 0 else 0.0
            complete   = round(100 - null_pct, 2)
            unique_pct = round((distinct / total * 100), 2) if total > 0 else 0.0

            col_result: dict = {
                "null_count":       null_count,
                "null_pct":         null_pct,
                "completeness_pct": complete,
                "distinct_count":   distinct,
                "uniqueness_pct":   unique_pct,
                "null_severity":    _severity(null_pct, NULL_WARN_PCT, NULL_FAIL_PCT),
                "unique_severity":  _severity(unique_pct, UNIQUE_WARN_PCT, UNIQUE_WARN_PCT - 5,
                                              higher_is_worse=False),
            }

            if f"_min_{col}" in row:
                col_result["min"] = row[f"_min_{col}"]
                col_result["max"] = row[f"_max_{col}"]
                col_result["avg"] = round(float(row[f"_avg_{col}"] or 0), 4)

            if f"_max_ts_{col}" in row:
                col_result["latest_value"] = str(row[f"_max_ts_{col}"])

            results["columns"][col] = col_result

        return json.dumps(results, default=str)

    except Exception as e:
        return f"ERROR: Standard checks failed — {str(e)}"


# ─── Tool 3: run_dynamic_checks ───────────────────────────────────────────────

@tool
def run_dynamic_checks(project_id: str, dataset_id: str, table_id: str,
                       standard_results: str = "{}") -> str:
    """Uses the LLM to generate context-aware SQL checks (uniqueness on IDs,
    format validation on emails, range checks on amounts, etc.) and executes them.
    Pass standard_results from run_standard_checks so the LLM can target anomalies.
    Every LLM-generated SQL is dry-run validated before execution.

    Required parameters:
        - project_id:      GCP project ID
        - dataset_id:      BigQuery dataset name
        - table_id:        BigQuery table name

    Optional parameters:
        - standard_results: JSON string from run_standard_checks (improves targeting)
    """
    try:
        bq_obj = BigQuerySource(project_id)

        # Fetch column schema internally (no sample needed — saves tokens)
        schema_sql = f"""
            SELECT column_name, data_type, is_nullable
            FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{table_id}'
            ORDER BY ordinal_position
        """
        schema_df = bq_obj.query_or_raise(schema_sql)
        if schema_df.empty:
            return "ERROR: Table not found."
        schema_info = schema_df.to_dict(orient="records")

        # Pass full column stats to the LLM so it can decide what to investigate
        std_summary = ""
        if standard_results and standard_results != "{}":
            try:
                std = json.loads(standard_results) if isinstance(standard_results, str) else standard_results
                cols = std.get("columns", {})
                if cols:
                    std_summary = "\n\nStandard check results — use these to decide what to investigate:\n"
                    for col, stats in cols.items():
                        parts = [f"nulls={stats.get('null_pct')}%"]
                        parts.append(f"uniqueness={stats.get('uniqueness_pct')}%")
                        if stats.get("min") is not None:
                            parts.append(f"min={stats['min']}, max={stats['max']}, avg={stats.get('avg')}")
                        if stats.get("latest_value") is not None:
                            parts.append(f"latest={stats['latest_value']}")
                        std_summary += f"- {col}: {', '.join(parts)}\n"
            except Exception:
                pass

        # Ask LLM to generate targeted checks
        generation_prompt = f"""
You are a data quality engineer. Generate targeted SQL data quality checks for this BigQuery table.

Table: `{project_id}.{dataset_id}.{table_id}`
Columns:
{json.dumps(schema_info, indent=2)}{std_summary}

Output a JSON array. Each element must have:
- "name": short check name (e.g. "unique_customer_id")
- "description": what is being checked
- "sql": a SELECT query returning a SINGLE numeric value (count of violations or ratio)
- "pass_condition": What is the pass condition for this check, e.g. "result == 0" or "result > 0.95"

Rules:
- Only SELECT queries. No DML (INSERT, UPDATE, DELETE, DROP, etc.)
- Limit to 5 checks maximum
- Use fully qualified table name `{project_id}.{dataset_id}.{table_id}`
- If anomalies are listed above, generate checks that investigate those columns first

Respond with ONLY the JSON array, no explanation.
"""
        llm_response = model.invoke(generation_prompt)
        raw_checks   = llm_response.content.strip()

        # Strip markdown code fences if present
        if raw_checks.startswith("```"):
            raw_checks = raw_checks.split("```")[1]
            if raw_checks.startswith("json"):
                raw_checks = raw_checks[4:]

        checks  = json.loads(raw_checks)
        results = []

        for check in checks:
            name        = check.get("name", "unnamed")
            description = check.get("description", "")
            sql         = check.get("sql", "").strip()
            condition   = check.get("pass_condition", "")

            # Safety: reject non-SELECT SQL
            if not _safe_sql(sql):
                results.append({
                    "name":        name,
                    "description": description,
                    "status":      "SKIPPED",
                    "reason":      "SQL rejected — contains forbidden keyword",
                })
                continue

            # Dry-run cost check
            try:
                job_config  = bq.QueryJobConfig(dry_run=True, use_query_cache=False)
                dry_job     = bq_obj.client.query(sql, job_config=job_config)
                bytes_proc  = dry_job.total_bytes_processed or 0
                gb_proc     = bytes_proc / (1024 ** 3)

                if gb_proc > 10:
                    results.append({
                        "name":        name,
                        "description": description,
                        "status":      "SKIPPED",
                        "reason":      f"Dry-run estimated {gb_proc:.1f} GB — too expensive",
                    })
                    continue
            except Exception as e:
                results.append({
                    "name":        name,
                    "description": description,
                    "status":      "ERROR",
                    "reason":      f"Dry-run failed: {str(e)}",
                })
                continue

            # Execute
            try:
                df    = bq_obj.query_or_raise(sql)
                value = float(df.iloc[0, 0]) if not df.empty else None

                # Evaluate pass condition safely
                try:
                    passed = bool(eval(condition, {"__builtins__": {}}, {"result": value}))
                except Exception:
                    passed = None

                results.append({
                    "name":           name,
                    "description":    description,
                    "sql":            sql,
                    "result":         value,
                    "pass_condition": condition,
                    "status":         "PASS" if passed is True else ("FAIL" if passed is False else "UNKNOWN"),
                })
            except Exception as e:
                results.append({
                    "name":        name,
                    "description": description,
                    "status":      "ERROR",
                    "reason":      str(e),
                })

        return json.dumps({"dynamic_checks": results}, default=str)

    except Exception as e:
        return f"ERROR: Dynamic checks failed — {str(e)}"


# ─── Tool 4: list_dataset_tables ─────────────────────────────────────────────

@tool
def list_dataset_tables(project_id: str, dataset_id: str) -> str:
    """Lists all tables in the dataset with approximate row counts and sizes.
    Use this to discover related tables for cross-table integrity checks — e.g.
    if the target table has a customer_id column, look for a 'customers' table here.

    Required parameters:
        - project_id:  GCP project ID
        - dataset_id:  BigQuery dataset name
    """
    try:
        bq_obj = BigQuerySource(project_id)
        sql = f"""
            SELECT table_id,
                   row_count,
                   ROUND(size_bytes / POW(1024, 3), 3) AS size_gb,
                   CASE type WHEN 1 THEN 'TABLE' WHEN 2 THEN 'VIEW' WHEN 3 THEN 'EXTERNAL' ELSE 'UNKNOWN' END AS table_type
            FROM `{project_id}.{dataset_id}.__TABLES__`
            ORDER BY table_id
        """
        df = bq_obj.query_or_raise(sql)
        if df.empty:
            return json.dumps({"dataset": dataset_id, "tables": []})
        return json.dumps({"dataset": dataset_id, "tables": df.to_dict(orient="records")}, default=str)

    except Exception as e:
        return f"ERROR: Could not list dataset tables — {str(e)}"


# ─── Tool 5: run_cross_table_checks ──────────────────────────────────────────

@tool
def run_cross_table_checks(project_id: str, dataset_id: str,
                           table1: str, table2: str, join_key: str) -> str:
    """Checks referential integrity between two tables using a LEFT JOIN.
    Finds rows in table1 whose join_key value has no match in table2 (orphaned records).
    Dry-runs the query before executing to guard against cost.

    Required parameters:
        - project_id:  GCP project ID
        - dataset_id:  BigQuery dataset name
        - table1:      Source / child table (e.g. orders)
        - table2:      Reference / parent table (e.g. customers)
        - join_key:    Column name present in both tables
    """
    try:
        sql = f"""
            SELECT COUNT(*) AS orphaned_rows
            FROM `{project_id}.{dataset_id}.{table1}` t1
            LEFT JOIN `{project_id}.{dataset_id}.{table2}` t2
              ON t1.`{join_key}` = t2.`{join_key}`
            WHERE t2.`{join_key}` IS NULL
        """

        bq_obj = BigQuerySource(project_id)

        # Dry-run guard
        job_config = bq.QueryJobConfig(dry_run=True, use_query_cache=False)
        dry_job    = bq_obj.client.query(sql, job_config=job_config)
        gb_proc    = (dry_job.total_bytes_processed or 0) / (1024 ** 3)

        if gb_proc > 20:
            return json.dumps({
                "check":  "referential_integrity",
                "status": "SKIPPED",
                "reason": f"Estimated {gb_proc:.1f} GB — exceeds 20 GB safety limit for cross-table check",
            })

        df            = bq_obj.query_or_raise(sql)
        orphaned_rows = int(df.iloc[0, 0]) if not df.empty else 0

        result = {
            "check":         "referential_integrity",
            "table1":        table1,
            "table2":        table2,
            "join_key":      join_key,
            "orphaned_rows": orphaned_rows,
            "status":        "PASS" if orphaned_rows == 0 else "FAIL",
            "estimated_gb":  round(gb_proc, 3),
        }

        # Append full result to temp file — generate_dq_report reads and deletes this
        _CROSS_TEMP = "./data/.cross_temp.json"
        os.makedirs("./data", exist_ok=True)
        existing = []
        if os.path.exists(_CROSS_TEMP):
            try:
                with open(_CROSS_TEMP) as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        existing.append(result)
        with open(_CROSS_TEMP, "w", encoding="utf-8") as f:
            json.dump(existing, f)

        # Return one-line summary so agent stays aware without bloating context
        if orphaned_rows == 0:
            return f"Cross-table check PASS: {table1} → {table2} on `{join_key}` — no orphaned rows ({round(gb_proc, 3)} GB scanned)"
        else:
            return f"Cross-table check FAIL: {table1} → {table2} on `{join_key}` — {orphaned_rows} orphaned rows found ({round(gb_proc, 3)} GB scanned)"

    except Exception as e:
        return f"ERROR: Cross-table check failed — {str(e)}"


# ─── Tool 6: generate_dq_report ──────────────────────────────────────────────
# HTML template lives at config/dq_report_template.html (Jinja2)

_STATUS_COLOR = {"PASS": "#22c55e", "WARN": "#f59e0b", "FAIL": "#ef4444",
                 "SKIPPED": "#94a3b8", "ERROR": "#ef4444", "N/A": "#94a3b8"}
_STATUS_BG    = {"PASS": "#dcfce7", "WARN": "#fef3c7", "FAIL": "#fee2e2",
                 "SKIPPED": "#f1f5f9", "ERROR": "#fee2e2", "N/A": "#f1f5f9"}

def _status_color(status: str) -> str:
    return _STATUS_COLOR.get(status, "#94a3b8")

def _badge(status: str) -> Markup:
    color = _STATUS_COLOR.get(status, "#94a3b8")
    bg    = _STATUS_BG.get(status, "#f1f5f9")
    return Markup(f'<span class="badge" style="background:{bg};color:{color};">{status}</span>')

def _progress_bar(pct: float, status: str) -> Markup:
    color = _STATUS_COLOR.get(status, "#94a3b8")
    safe  = max(0.0, min(float(pct or 0), 100.0))
    return Markup(
        f'<div class="progress-wrap">'
        f'<div class="progress-bar" style="background:{color};width:{safe}%;"></div>'
        f'</div>'
    )


@tool
def generate_dq_report(standard_results: str, dynamic_results: str = "{}",
                       table_name: str = "table") -> str:
    """Aggregates results from all DQ checks into an HTML report with charts and
    saves it to ./data/dq_report_<table>.html using the Jinja2 template at
    config/dq_report_template.html. Call this as the final step after all checks.
    Cross-table results are read automatically from the internal temp file written
    by run_cross_table_checks — do not pass them manually.

    Required parameters:
        - standard_results:  JSON string from run_standard_checks

    Optional parameters:
        - dynamic_results:   JSON string from run_dynamic_checks  (default: empty)
        - table_name:        Table name label for the report title (default: "table")
    """
    import datetime, os
    from pathlib import Path

    try:
        std  = json.loads(standard_results)  if isinstance(standard_results, str) else standard_results
        dyn  = json.loads(dynamic_results)   if isinstance(dynamic_results,  str) else dynamic_results

        # Read cross-table results from temp file written by run_cross_table_checks
        _CROSS_TEMP = "./data/.cross_temp.json"
        cross_list = []
        if os.path.exists(_CROSS_TEMP):
            try:
                with open(_CROSS_TEMP) as f:
                    cross_list = json.load(f)
                os.remove(_CROSS_TEMP)
            except Exception:
                cross_list = []

        columns      = std.get("columns", {})
        total_rows   = std.get("total_rows", 0)
        col_count    = len(columns)
        approximate  = std.get("approximate", False)
        sample_note  = std.get("sample_note")      # e.g. "Approximate results — 10% TABLESAMPLE used (table is 200 GB)"
        estimated_gb = std.get("estimated_gb")

        # ── Scores ────────────────────────────────────────────────────────────
        completeness_scores = [v["completeness_pct"] for v in columns.values()]
        completeness_avg    = round(sum(completeness_scores) / col_count, 2) if col_count else 0.0
        completeness_status = "PASS" if completeness_avg >= 95 else ("WARN" if completeness_avg >= 85 else "FAIL")

        uniqueness_scores = [v["uniqueness_pct"] for v in columns.values()]
        uniqueness_avg    = round(sum(uniqueness_scores) / col_count, 2) if col_count else 0.0
        uniqueness_status = "PASS" if uniqueness_avg >= 80 else ("WARN" if uniqueness_avg >= 60 else "FAIL")

        dyn_checks = dyn.get("dynamic_checks", [])
        dyn_pass   = sum(1 for c in dyn_checks if c.get("status") == "PASS")
        dyn_total  = sum(1 for c in dyn_checks if c.get("status") in ("PASS", "FAIL"))
        validity_pct    = round(dyn_pass / dyn_total * 100, 2) if dyn_total > 0 else None
        validity_status = ("PASS" if validity_pct == 100 else ("WARN" if (validity_pct or 0) >= 80 else "FAIL")) if validity_pct is not None else "N/A"

        # RI status — FAIL if any cross-table check failed, PASS if all passed, N/A if none ran
        if cross_list:
            ri_status = "FAIL" if any(x.get("status") == "FAIL" for x in cross_list) else "PASS"
        else:
            ri_status = "N/A"

        scores = [completeness_avg]
        if validity_pct is not None:
            scores.append(validity_pct)
        if ri_status in ("PASS", "FAIL"):
            scores.append(100.0 if ri_status == "PASS" else 0.0)
        overall_score  = round(sum(scores) / len(scores), 2)
        overall_status = "PASS" if overall_score >= 90 else ("WARN" if overall_score >= 70 else "FAIL")

        generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── Pillars list (used by template {% for %}) ─────────────────────────
        pillars = [
            {"name": "Completeness", "score": completeness_avg, "status": completeness_status, "na_label": ""},
            {"name": "Uniqueness",   "score": uniqueness_avg,   "status": uniqueness_status,   "na_label": ""},
            {"name": "Validity",     "score": validity_pct,     "status": validity_status,
             "na_label": "No dynamic checks run"},
            {"name": "Ref. Integrity",
             "score": (100.0 if ri_status == "PASS" else 0.0) if ri_status in ("PASS","FAIL") else None,
             "status": ri_status, "na_label": "Not checked"},
        ]

        # ── Chart data (injected as a single JSON object into the template) ───
        col_labels    = list(columns.keys())
        null_data     = [v["null_pct"] for v in columns.values()]
        complete_data = [v["completeness_pct"] for v in columns.values()]
        unique_data   = [v["uniqueness_pct"] for v in columns.values()]

        pillar_labels = [p["name"] for p in pillars if p["score"] is not None]
        pillar_values = [p["score"] for p in pillars if p["score"] is not None]
        pillar_colors = [_status_color("PASS" if v >= 90 else ("WARN" if v >= 70 else "FAIL"))
                         for v in pillar_values]
        null_colors   = [_status_color("FAIL" if v >= NULL_FAIL_PCT else ("WARN" if v >= NULL_WARN_PCT else "PASS"))
                         for v in null_data]
        unique_colors = [_status_color("PASS" if v >= UNIQUE_WARN_PCT else ("WARN" if v >= UNIQUE_WARN_PCT - 15 else "FAIL"))
                         for v in unique_data]

        chart_data = json.dumps({
            "pillar_labels": pillar_labels, "pillar_values": pillar_values,
            "pillar_colors": pillar_colors, "col_labels": col_labels,
            "null_data": null_data, "null_colors": null_colors,
            "complete_data": complete_data,
            "unique_data": unique_data, "unique_colors": unique_colors,
        })

        # ── Render template ───────────────────────────────────────────────────
        template_dir  = Path(__file__).parent.parent / "config"
        jinja_env     = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=jinja2.select_autoescape(["html"]),
        )
        # Register helper functions as globals so the template can call them
        jinja_env.globals.update(badge=_badge, progress_bar=_progress_bar,
                                 status_color=_status_color)

        template = jinja_env.get_template("dq_report_template.html")
        html = template.render(
            table_name=table_name,
            generated_at=generated_at,
            total_rows=total_rows,
            col_count=col_count,
            overall_score=overall_score,
            overall_status=overall_status,
            dyn_pass=dyn_pass,
            dyn_total=dyn_total,
            pillars=pillars,
            columns=columns,
            dyn_checks=dyn_checks,
            cross_checks=cross_list,
            show_cross=len(cross_list) > 0,
            chart_data=chart_data,
            approximate=approximate,
            sample_note=sample_note,
            estimated_gb=estimated_gb,
        )

        # ── Save to file ──────────────────────────────────────────────────────
        os.makedirs("./data", exist_ok=True)
        timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name   = table_name.replace(".", "_").replace(" ", "_")
        output_path = f"./data/dq_report_{safe_name}_{timestamp}.html"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return (f"DQ Report generated successfully.\n"
                f"File: {output_path}\n"
                f"Overall Health: {overall_score}% [{overall_status}]\n"
                f"  Completeness: {completeness_avg}% [{completeness_status}]\n"
                f"  Uniqueness:   {uniqueness_avg}% [{uniqueness_status}]\n"
                f"  Validity:     {f'{validity_pct}%' if validity_pct is not None else 'N/A'} [{validity_status}]\n"
                f"  Ref Integrity: [{ri_status}]\n"
                f"Total rows: {total_rows:,} · {col_count} columns")

    except Exception as e:
        return f"ERROR: Report generation failed — {str(e)}"


# ─── Agent ────────────────────────────────────────────────────────────────────

dq_agent = create_agent(
    model=model,
    system_prompt="""You are a Data Quality Agent. Assess the quality of BigQuery tables through intelligent investigation — not just mechanical execution.

Follow this workflow, applying judgment at each step:

1. Call discover_schema to understand the table structure, size, and sample data.

2. Call run_standard_checks to measure completeness, uniqueness, and freshness.

3. Analyse the standard_results before proceeding. Identify:
   - Columns with elevated null rates (null_severity WARN or FAIL)
   - Columns ending in _id, _key, _fk, _ref — potential foreign keys
   - Numeric columns with suspicious min/max ranges
   Use these observations to inform step 4.

4. Call run_dynamic_checks, passing the JSON output of run_standard_checks as standard_results.
   The LLM will generate checks targeted at the anomalies you identified.

5. If you spotted _id or _key columns in step 3, call list_dataset_tables to see what other
   tables exist in the dataset. If a plausible parent table is found (e.g. customer_id → customers
   table, order_id → orders table), call run_cross_table_checks with the relevant join_key.
   You may call run_cross_table_checks multiple times for different FK columns — each call
   automatically records its result. Only check relationships clearly indicated by naming.

6. Call generate_dq_report last, passing standard_results, dynamic_results, and table_name
   formatted as "dataset.table" (e.g. "sales.orders"). Cross-table results are automatically
   collected from any run_cross_table_checks calls — do not pass them manually.

Rules:
- If project_id, dataset_id, or table_id are missing, respond: "ERROR: Missing parameters — [list]."
- Individual check failures do NOT stop the workflow — capture them in the report.
- Only return a terminal ERROR if the table cannot be found or accessed.
""",
    tools=[discover_schema, run_standard_checks, run_dynamic_checks,
           list_dataset_tables, run_cross_table_checks, generate_dq_report]
)


if __name__ == "__main__":
    with get_openai_callback() as cb:
        result = dq_agent.invoke({
            "messages": [{
                "role": "user",
                "content": "Run data quality checks on table Orders in dataset init_data_4778 in project data-eng-proj-496117"
            }]
        })
        print(result["messages"][-1].content)
        print("Total Tokens:", cb.total_tokens)
        print("Total Cost: $", cb.total_cost)
