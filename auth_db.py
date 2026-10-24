import os
import sqlite3
import hashlib
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")

def _hash_password(password: str) -> str:
    """Hash password using SHA-256 with consistent salt."""
    salt = "retailflow_ai_secure_salt_2026"
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

def get_connection():
    """Returns a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database tables and executes migrations if needed."""
    conn = get_connection()
    c = conn.cursor()

    # 1. Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'Store Manager',
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Auto-migrate columns if table already existed without role, status, updated_at
    c.execute("PRAGMA table_info(users)")
    cols = [col[1] for col in c.fetchall()]
    if "role" not in cols:
        try:
            c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'Store Manager'")
        except Exception:
            pass
    if "status" not in cols:
        try:
            c.execute("ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'approved'")
        except Exception:
            pass
    if "updated_at" not in cols:
        try:
            c.execute("ALTER TABLE users ADD COLUMN updated_at TIMESTAMP")
        except Exception:
            pass

    # 2. Admins table (unique credentials exclusively for project team & guide)
    c.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_id TEXT UNIQUE NOT NULL COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            display_name TEXT DEFAULT 'Administrator',
            role TEXT DEFAULT 'System Administrator',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 3. Audit log table
    c.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()

    # NOTE: Default accounts are NOT seeded with hardcoded credentials.
    # Create accounts via the registration flow or the Admin Console.
    conn.commit()
    conn.close()

def log_audit(actor: str, action: str, details: str = ""):
    """Logs an administrative or security event to the audit_logs table."""
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            INSERT INTO audit_logs (actor, action, details)
            VALUES (?, ?, ?)
        """, (actor, action, details))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Audit log error: {e}")

# ========================================================
# USER AUTHENTICATION & MANAGEMENT
# ========================================================

def register_user(email: str, password: str, role: str = "Store Manager"):
    """
    Registers a new user. Newly registered accounts start as 'pending'
    and require administrator approval before logging in.
    """
    init_db()
    clean_email = email.strip().lower() if email else ""
    if not clean_email or "@" not in clean_email:
        return False, "Please provide a valid email address.", None

    if not password or len(password) < 4:
        return False, "Password must be at least 4 characters long.", None

    pw_hash = _hash_password(password)

    conn = get_connection()
    c = conn.cursor()
    try:
        # Save as pending approval
        c.execute("""
            INSERT INTO users (email, password_hash, role, status)
            VALUES (?, ?, ?, 'pending')
        """, (clean_email, pw_hash, role))
        conn.commit()
        user_id = c.lastrowid
        conn.close()

        log_audit(clean_email, "USER_REGISTERED", f"Role: {role}, Initial Status: pending")

        return True, "Account registered successfully! Your account is now pending admin approval before you can sign in.", {
            "id": user_id,
            "email": clean_email,
            "role": role,
            "status": "pending"
        }
    except sqlite3.IntegrityError:
        conn.close()
        return False, f"An account with email '{clean_email}' already exists. Please sign in or contact the admin.", None
    except Exception as e:
        conn.close()
        return False, f"Database error: {e}", None

def verify_user(email: str, password: str):
    """
    Verifies credentials against the database.
    Checks whether the user is approved, pending, or rejected.
    """
    init_db()
    clean_email = email.strip().lower() if email else ""
    if not clean_email:
        return False, "Please enter your email address.", None

    pw_hash = _hash_password(password)

    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT id, email, password_hash, role, status, created_at
        FROM users
        WHERE email = ?
    """, (clean_email,))
    row = c.fetchone()
    conn.close()

    if not row:
        return False, f"No account found for '{clean_email}'. Please register first.", None

    if row["password_hash"] != pw_hash:
        log_audit(clean_email, "LOGIN_FAILED", "Incorrect password attempt")
        return False, "Incorrect password. Please verify your credentials.", None

    status = (row["status"] or "approved").lower()

    if status == "pending":
        return False, f"Your account ({clean_email}) is pending administrator approval. Please wait for an admin to approve your request.", {
            "id": row["id"],
            "email": row["email"],
            "role": row["role"],
            "status": "pending"
        }

    if status == "rejected":
        return False, f"Access rejected: Your account ({clean_email}) was rejected by the administrator.", None

    if status == "suspended":
        return False, f"Access suspended: Your account ({clean_email}) has been deactivated by the administrator.", None

    # Status is approved!
    log_audit(clean_email, "LOGIN_SUCCESS", f"Role: {row['role']}")
    return True, f"Welcome back, {row['email']}!", {
        "id": row["id"],
        "email": row["email"],
        "role": row["role"],
        "status": row["status"],
        "created_at": row["created_at"]
    }

# ========================================================
# ADMIN AUTHENTICATION & CONSOLE OPERATIONS
# ========================================================

def verify_admin(admin_id: str, password: str):
    """
    Verifies admin credentials using our unique ID and password.
    Returns (success: bool, message: str, admin_info: dict or None).
    """
    init_db()
    clean_id = admin_id.strip() if admin_id else ""
    if not clean_id or not password:
        return False, "Please provide your unique Admin ID and password.", None

    pw_hash = _hash_password(password)

    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT id, admin_id, password_hash, display_name, role, created_at
        FROM admins
        WHERE admin_id = ? COLLATE NOCASE
    """, (clean_id,))
    row = c.fetchone()
    conn.close()

    if not row or row["password_hash"] != pw_hash:
        log_audit(clean_id or "unknown", "ADMIN_LOGIN_FAILED", "Invalid admin credentials")
        return False, "Invalid Admin ID or password. Access is restricted to authorized project administrators.", None

    log_audit(row["admin_id"], "ADMIN_LOGIN_SUCCESS", f"Name: {row['display_name']}")
    return True, f"Admin authentication successful! Welcome, {row['display_name']}.", {
        "id": row["id"],
        "admin_id": row["admin_id"],
        "display_name": row["display_name"],
        "role": row["role"],
        "created_at": row["created_at"]
    }

def get_all_users_admin():
    """Returns all registered users with their details and approval statuses."""
    init_db()
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT id, email, role, status, created_at, updated_at
        FROM users
        ORDER BY 
            CASE status 
                WHEN 'pending' THEN 1 
                WHEN 'approved' THEN 2 
                ELSE 3 
            END,
            created_at DESC
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def get_admin_metrics():
    """Returns aggregate user statistics for admin scorecards."""
    init_db()
    conn = get_connection()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM users")
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM users WHERE status = 'pending'")
    pending = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM users WHERE status = 'approved'")
    approved = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM users WHERE status IN ('rejected', 'suspended')")
    rejected = c.fetchone()[0]

    conn.close()
    return {
        "total": total,
        "pending": pending,
        "approved": approved,
        "rejected": rejected
    }

def approve_user(user_id: int, admin_actor: str = "admin"):
    """Approves a user account, allowing them to sign in."""
    init_db()
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT email FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False, f"User with ID {user_id} was not found."

    email = row["email"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("UPDATE users SET status = 'approved', updated_at = ? WHERE id = ?", (now, user_id))
    conn.commit()
    conn.close()

    log_audit(admin_actor, "USER_APPROVED", f"Approved user {email} (ID: {user_id})")
    return True, f"User '{email}' has been approved successfully."

def reject_user(user_id: int, admin_actor: str = "admin"):
    """Rejects a user account, preventing login."""
    init_db()
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT email FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False, f"User with ID {user_id} was not found."

    email = row["email"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("UPDATE users SET status = 'rejected', updated_at = ? WHERE id = ?", (now, user_id))
    conn.commit()
    conn.close()

    log_audit(admin_actor, "USER_REJECTED", f"Rejected user {email} (ID: {user_id})")
    return True, f"User '{email}' has been marked as rejected."

def update_user_details(user_id: int, email: str, role: str, status: str, new_password: str = None, admin_actor: str = "admin"):
    """Updates user information (email, role, status, and optional new password)."""
    init_db()
    clean_email = email.strip().lower() if email else ""
    if not clean_email or "@" not in clean_email:
        return False, "Please provide a valid email address."

    conn = get_connection()
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        if new_password and len(new_password.strip()) >= 4:
            pw_hash = _hash_password(new_password.strip())
            c.execute("""
                UPDATE users
                SET email = ?, role = ?, status = ?, password_hash = ?, updated_at = ?
                WHERE id = ?
            """, (clean_email, role, status, pw_hash, now, user_id))
            pwd_note = " (Password reset)"
        else:
            c.execute("""
                UPDATE users
                SET email = ?, role = ?, status = ?, updated_at = ?
                WHERE id = ?
            """, (clean_email, role, status, now, user_id))
            pwd_note = ""

        conn.commit()
        conn.close()

        log_audit(admin_actor, "USER_UPDATED", f"Updated user ID {user_id}: {clean_email}, Role: {role}, Status: {status}{pwd_note}")
        return True, f"User details for '{clean_email}' updated successfully{pwd_note}."
    except sqlite3.IntegrityError:
        conn.close()
        return False, f"Email '{clean_email}' is already in use by another user."
    except Exception as e:
        conn.close()
        return False, f"Update error: {e}"

def delete_user_account(user_id: int, admin_actor: str = "admin"):
    """Permanently deletes a user from the database."""
    init_db()
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT email FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False, f"User with ID {user_id} was not found."

    email = row["email"]
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

    log_audit(admin_actor, "USER_DELETED", f"Deleted user {email} (ID: {user_id})")
    return True, f"User '{email}' has been deleted from the database."

def add_user_by_admin(email: str, password: str, role: str, status: str = "approved", admin_actor: str = "admin"):
    """Allows administrator to manually provision a user account."""
    init_db()
    clean_email = email.strip().lower() if email else ""
    if not clean_email or "@" not in clean_email:
        return False, "Please provide a valid email address.", None

    if not password or len(password) < 4:
        return False, "Password must be at least 4 characters long.", None

    pw_hash = _hash_password(password)

    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO users (email, password_hash, role, status)
            VALUES (?, ?, ?, ?)
        """, (clean_email, pw_hash, role, status))
        conn.commit()
        user_id = c.lastrowid
        conn.close()

        log_audit(admin_actor, "ADMIN_ADDED_USER", f"Added user {clean_email}, Role: {role}, Status: {status}")
        return True, f"User '{clean_email}' successfully added with status '{status}'.", {
            "id": user_id,
            "email": clean_email,
            "role": role,
            "status": status
        }
    except sqlite3.IntegrityError:
        conn.close()
        return False, f"An account with email '{clean_email}' already exists.", None
    except Exception as e:
        conn.close()
        return False, f"Database error: {e}", None

def get_audit_logs(limit: int = 50):
    """Returns the most recent admin audit trail entries."""
    init_db()
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, actor, action, details, created_at FROM audit_logs ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

# Initialize on import
init_db()
