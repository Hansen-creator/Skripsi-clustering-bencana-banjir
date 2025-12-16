import pyodbc
from datetime import datetime

# KONEKSI KE DATABASE
def get_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=login_app;"
        "UID=sa;"
        "PWD=12345;"
    )

# GET USER PROFILE
def get_user_profile(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT username, role, bio, avatar, avatar_filename FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "username": row[0],
        "role": row[1],
        "bio": row[2],
        "avatar": row[3],
        "avatar_filename": row[4]
    }

# CHECK USER LOGIN
def check_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT username, role, bio, avatar, avatar_filename FROM users WHERE username=? AND password=?", (username, password))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "username": row[0],
        "role": row[1],
        "bio": row[2],
        "avatar": row[3],
        "avatar_filename": row[4]
    }

# ADD USER (UNIK: USERNAME, PASSWORD)
def add_user(username, password, bio=None, avatar_bytes=None, avatar_filename=None):
    conn = get_connection()
    cur = conn.cursor()
    try:
        # cek username unik
        cur.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        if cur.fetchone():
            return False

        # cek password unik
        cur.execute("SELECT 1 FROM users WHERE password = ?", (password,))
        if cur.fetchone():
            return False

        # bio sekarang bebas (tidak dicek unik)
        cur.execute("""
            INSERT INTO users (username, password, role, bio, avatar, avatar_filename, created_at)
            VALUES (?, ?, 'user', ?, ?, ?, ?)
        """, (username, password, bio, avatar_bytes, avatar_filename, datetime.now()))
        conn.commit()
        return True
    except Exception as e:
        print("DEBUG ERROR add_user:", e)
        return False
    finally:
        conn.close()

# CEK USER EXIST
def user_exists(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists

# UPDATE USER (UNIK: USERNAME, PASSWORD)
def update_user(old_username, new_username=None, new_password=None, new_bio=None, avatar_bytes=None, avatar_filename=None):
    conn = get_connection()
    cur = conn.cursor()
    try:
        # cek username unik
        if new_username:
            cur.execute("SELECT 1 FROM users WHERE username = ? AND username != ?", (new_username, old_username))
            if cur.fetchone():
                return False, "❌ Username sudah digunakan user lain."

        # cek password unik
        if new_password:
            cur.execute("SELECT 1 FROM users WHERE password = ? AND username != ?", (new_password, old_username))
            if cur.fetchone():
                return False, "❌ Password sudah digunakan user lain."
            
        sets = []
        params = []
        if new_username is not None:
            sets.append("username = ?")
            params.append(new_username)
        if new_password is not None:
            sets.append("password = ?")
            params.append(new_password)
        if new_bio is not None:
            sets.append("bio = ?")
            params.append(new_bio)
        if avatar_bytes is not None:
            sets.append("avatar = ?")
            params.append(avatar_bytes)
            sets.append("avatar_filename = ?")
            params.append(avatar_filename)

        sets.append("last_updated = ?")
        params.append(datetime.now())

        if len(sets) == 0:
            return False, "❌ Tidak ada data yang diubah."

        sql = f"UPDATE users SET {', '.join(sets)} WHERE username = ?"
        params.append(old_username)

        cur.execute(sql, tuple(params))
        conn.commit()
        return True, "✅ Data berhasil diperbarui!"
    except Exception as e:
        print("DEBUG ERROR update_user:", e)
        return False, "❌ Gagal memperbarui data."
    finally:
        conn.close()
