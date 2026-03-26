"""Initialize the Richiebot SQLite database."""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "richiebot.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS pdfs (
        id INTEGER PRIMARY KEY,
        nas_path TEXT,
        local_path TEXT,
        filename TEXT,
        folder TEXT,
        size_bytes INTEGER,
        num_pages INTEGER,
        downloaded_at TIMESTAMP,
        status TEXT DEFAULT 'pending',
        UNIQUE(nas_path)
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS pages (
        id INTEGER PRIMARY KEY,
        pdf_id INTEGER REFERENCES pdfs(id),
        pdf_name TEXT,
        page_num INTEGER,
        image_path TEXT,
        class TEXT,
        class_confidence REAL,
        rotation_hint INTEGER DEFAULT 0,
        status TEXT DEFAULT 'pending',
        model_used TEXT,
        text TEXT,
        chars INTEGER,
        time_sec REAL,
        quality_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        processed_at TIMESTAMP,
        UNIQUE(pdf_name, page_num)
    )""")

    c.execute("CREATE INDEX IF NOT EXISTS idx_pages_status ON pages(status)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_pages_class ON pages(class)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_pdfs_status ON pdfs(status)")

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")
    return DB_PATH

if __name__ == "__main__":
    init_db()
