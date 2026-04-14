import pandas as pd
import psycopg2
import os

def export_to_excel(data, filename, output_dir):

    rows = []

    for entry in data.get("lab_results", []):
        rows.append({
            "test_name": entry.get("test_name"),
            "value": entry.get("value"),
            "unit": entry.get("unit"),
            "status": entry.get("status"),
            "interpretation": entry.get("interpretation"),
            "interpretation_fr": entry.get("interpretation_fr"),
        })

    df = pd.DataFrame(rows)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename + ".xlsx")

    df.to_excel(path, index=False)

    return path

def insert_into_postgres(data):

    conn = psycopg2.connect(
        dbname="medical_db",
        user="postgres",
        password="ninjaas84116",
        host="localhost",
        port="5432"
    )

    cursor = conn.cursor()

    # Insertion du rapport principal
    cursor.execute("""
    INSERT INTO reports (summary, summary_fr, priority, severity)
    VALUES (%s, %s, %s, %s)
    RETURNING id
    """, (
    data.get("summary"),
    data.get("summary_fr"),
    data.get("clinical_priority"),
    data.get("severity_score")
    ))

    result = cursor.fetchone()
    if result is None:
        raise ValueError("Impossible d'insérer le rapport: aucun ID retourné")
    report_id = result[0]

    # Resultats de laboratoire
    for entry in data.get("lab_results", []):
        cursor.execute("""
            INSERT INTO lab_results (
                report_id, test_name, value, unit, status
            )
            VALUES (%s, %s, %s, %s, %s)
        """, (
            report_id,
            entry.get("test_name"),
            entry.get("value"),
            entry.get("unit"),
            entry.get("status")
        ))

    conn.commit()
    cursor.close()
    conn.close()