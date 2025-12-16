import pyodbc

def get_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=login_app;"
        "UID=sa;"
        "PWD=12345;"
    )
    return conn

def log_login(username, role):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO login_logs (username, role) VALUES (?, ?)",
            (username, role)
        )
        conn.commit()
    except Exception as e:
        print("DEBUG ERROR log_login:", e)
    finally:
        conn.close()


def check_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()

    print(f"DEBUG INPUT -> username: '{username}', password: '{password}'")

    cursor.execute(
        "SELECT username, role, is_active FROM users WHERE username=? AND password=?",
        (username, password)
    )
    user = cursor.fetchone()

    print(f"DEBUG DB RESULT -> {user}")
    conn.close()

    return user  # (username, role, is_active)

def add_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password, role, is_active) VALUES (?, ?, ?, 1)",
            (username, password, "user")
        )
        conn.commit()
        return True
    except Exception as e:
        print("DEBUG ERROR add_user:", e)
        return False
    finally:
        conn.close()

def user_exists(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE username=?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user is not None

def save_dataset(dataset_name, description, label, file_content, file_name, file_type, uploaded_by, is_selected=1):
    """Menyimpan dataset ke database dengan kolom is_selected"""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO datasets 
            (dataset_name, description, label, file_content, file_name, file_type, uploaded_by, is_selected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (dataset_name, description, label, file_content, file_name, file_type, uploaded_by, is_selected))
        conn.commit()
        return True
    except Exception as e:
        print("DEBUG ERROR save_dataset:", e)
        return False
    finally:
        conn.close()

def get_datasets_for_user():
    """Mengambil dataset yang dipilih untuk ditampilkan ke user"""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT dataset_name, description, file_name, uploaded_by, uploaded_at, file_content
            FROM datasets 
            WHERE is_selected = 1
            ORDER BY uploaded_at DESC
        """)
        rows = cur.fetchall()
        cols = [column[0] for column in cur.description]
        return rows, cols
    except Exception as e:
        print("DEBUG ERROR get_datasets_for_user:", e)
        return [], []
    finally:
        conn.close()

def get_all_datasets():
    """Mengambil semua dataset untuk admin"""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, dataset_name, description, label, file_name, file_type, 
                   uploaded_by, uploaded_at, is_selected
            FROM datasets
            ORDER BY uploaded_at DESC
        """)
        rows = cur.fetchall()
        cols = [column[0] for column in cur.description]
        return rows, cols
    except Exception as e:
        print("DEBUG ERROR get_all_datasets:", e)
        return [], []
    finally:
        conn.close()

def update_dataset_selection(dataset_id, is_selected):
    """Update status tampil/sembunyi dataset untuk user"""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE datasets 
            SET is_selected = ? 
            WHERE id = ?
        """, (is_selected, dataset_id))
        conn.commit()
        return True
    except Exception as e:
        print("DEBUG ERROR update_dataset_selection:", e)
        return False
    finally:
        conn.close()

def delete_dataset(dataset_id):
    """Menghapus dataset dari database"""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        conn.commit()
        return True
    except Exception as e:
        print("DEBUG ERROR delete_dataset:", e)
        return False
    finally:
        conn.close()