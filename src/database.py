import sqlite3
import os

DB_PATH = "database.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alunos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_url TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def execute_query(query, params=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)
    conn.commit()
    results = cursor.fetchall()
    conn.close()
    return results

def insert_aluno(name, image_url):
    query = "INSERT INTO alunos (name, image_url) VALUES (?, ?)"
    execute_query(query, (name, image_url))

def select_alunos():
    query = "SELECT * FROM alunos"
    return execute_query(query)
