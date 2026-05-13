import sqlite3
import json
import re
import random
import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).parent


def get_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='payroll'")
    schema = cursor.fetchone()[0]

    conn.close()
    return schema


def get_random_rows(db_path, num_rows=3):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM payroll")
    total_rows = cursor.fetchone()[0]

    if total_rows == 0:
        conn.close()
        return json.dumps([], indent=2)

    # LIMIT/OFFSET avoids the full-table scan of ORDER BY RANDOM()
    random_indices = random.sample(range(1, total_rows + 1), min(num_rows, total_rows))
    results = []

    for idx in random_indices:
        cursor.execute("SELECT * FROM payroll LIMIT 1 OFFSET ?", (idx - 1,))
        row = cursor.fetchone()
        if row:
            results.append(dict(row))

    conn.close()
    return json.dumps(results, indent=2)


def query_groq(schema, sample_data=None):
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        client = Groq(api_key=api_key)

        prompt = f"""From the following table description in SQLite please write a query to list the
average hourly rate for people working in the LAPD for each year\n\n{schema}"""

        if sample_data:
            prompt += f"\n\nSample rows:\n{sample_data}"

        messages = [
            {
                "role": "system",
                "content": "You are an expert SQL query generator. Generate only valid SQL queries without explanations unless asked. Format SQL queries in code blocks."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=0.95,
            stream=False
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {e}"


def execute_query(query, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(query)
    results = cursor.fetchall()

    conn.close()
    return results


def extract_sql(generated_query):
    sql_pattern = r"```sql\s*(.*?)\s*```"
    matches = re.findall(sql_pattern, generated_query, re.DOTALL | re.IGNORECASE)

    if not matches:
        select_pattern = r"(SELECT.*?;?)"
        select_matches = re.findall(select_pattern, generated_query, re.DOTALL | re.IGNORECASE)

        if select_matches:
            sql_query = select_matches[0].strip()
            if sql_query.endswith(';'):
                sql_query = sql_query[:-1]
            return sql_query
        else:
            return None
    else:
        return matches[0].strip()


def main():
    db_path = str(HERE / "city_payroll.db")

    print("=" * 70)
    print("LA City Payroll SQL Query Generator using Groq")
    print("=" * 70)

    schema = get_schema(db_path)
    print("Schema:")
    print(schema)

    sample_data = get_random_rows(db_path)
    print("\nSample Data:")
    print(sample_data)

    print("\n" + "=" * 70)
    print("Generating SQL query with Groq...")
    print("=" * 70)
    generated_query = query_groq(schema, sample_data)
    print("\nGenerated Response:")
    print(generated_query)

    sql_query = extract_sql(generated_query)

    if not sql_query:
        print("\nError: No SQL query found")
        print("Raw response:")
        print(generated_query)
        return

    print(f"\nExtracted SQL: {sql_query}")

    try:
        print("\n" + "=" * 70)
        print("Executing query...")
        print("=" * 70)
        results = execute_query(sql_query, db_path)

        print("\nResults:")
        print("| Year | Average |")
        print("|------|---------|")
        for year, average in results:
            print(f"| {year} | {average:.2f} |")

    except sqlite3.Error as e:
        print(f"Error executing SQL: {e}")


if __name__ == "__main__":
    main()
