"""Account registry, opaque login sessions, and isolated farm databases."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
import uuid
from contextvars import ContextVar
from pathlib import Path

import aiosqlite

from farm_database import FarmDatabase, now_text

BASE = Path(__file__).resolve().parent
DATA = BASE / "private_data"
current_database = ContextVar("farm_database", default=None)
current_actor = ContextVar("actor", default=None)
current_session = ContextVar("counting_state", default=None)
current_farm = ContextVar("farm_id", default=None)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def password_hash(password):
    if len(password) < 12 or len(password) > 256:
        raise ValueError("Use a password with 12 to 256 characters.")
    salt = secrets.token_bytes(16)
    result = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
    return "scrypt$" + salt.hex() + "$" + result.hex()


def password_matches(password, stored):
    try:
        _, salt, expected = stored.split("$")
        result = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1, dklen=32)
        return hmac.compare_digest(result.hex(), expected)
    except (ValueError, TypeError):
        return False


class DatabaseProxy:
    def __init__(self, fallback):
        object.__setattr__(self, "fallback", fallback)

    def __getattr__(self, name):
        database = current_database.get() or self.fallback
        return getattr(database, name)


class StateProxy:
    SHARED = {"model_pool", "active_models", "conf", "iou", "monitoring_mode", "device", "device_name"}

    def __init__(self, shared):
        object.__setattr__(self, "shared", shared)

    def __getattr__(self, name):
        target = self.shared if name in self.SHARED else current_session.get() or self.shared
        return getattr(target, name)

    def __setattr__(self, name, value):
        target = self.shared if name in self.SHARED else current_session.get() or self.shared
        setattr(target, name, value)


class Accounts:
    def __init__(self, path=DATA / "accounts.db", farm_directory=DATA / "farms", legacy_path=BASE / "tilapia_web_analytics.db"):
        self.path = Path(path)
        self.farm_directory = Path(farm_directory)
        self.legacy_path = Path(legacy_path)
        self.conn = None
        self.lock = asyncio.Lock()
        self.databases = {}
        self.database_lock = asyncio.Lock()
        self.login_attempts = {}

    async def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.farm_directory.mkdir(parents=True, exist_ok=True)
        self.conn = await aiosqlite.connect(str(self.path))
        self.conn.row_factory = aiosqlite.Row
        await self.conn.executescript("""
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS farms(id TEXT PRIMARY KEY,name TEXT NOT NULL,created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY,username TEXT UNIQUE NOT NULL,
                display_name TEXT NOT NULL,password_hash TEXT NOT NULL,role TEXT NOT NULL CHECK(role IN ('admin','user')),
                farm_id TEXT REFERENCES farms(id),active INTEGER NOT NULL DEFAULT 1,
                must_change_password INTEGER NOT NULL DEFAULT 1,created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS login_sessions(token_hash TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id),
                csrf TEXT NOT NULL,expires_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS audit_log(id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_id TEXT, farm_id TEXT,action TEXT NOT NULL,detail TEXT NOT NULL,timestamp TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS installation_settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        """)
        await self.conn.execute("INSERT OR IGNORE INTO farms VALUES('legacy','Existing farm',?)", (now_text(),))
        await self.conn.execute("DELETE FROM login_sessions WHERE expires_at<=?", (time.time(),))
        await self.conn.commit()
        username, password = os.getenv("TILAPIA_ADMIN_USERNAME"), os.getenv("TILAPIA_ADMIN_PASSWORD")
        if username and password and not await self.rows("SELECT id FROM users WHERE role='admin'"):
            await self.create_admin(username, password)

    async def rows(self, sql, args=()):
        cursor = await self.conn.execute(sql, args)
        return [dict(row) for row in await cursor.fetchall()]

    async def production_enabled(self):
        rows = await self.rows("SELECT value FROM installation_settings WHERE key='production'")
        return bool(rows and rows[0]['value']=='1')

    async def set_production(self, enabled, actor):
        async with self.lock:
            try:
                await self.conn.execute('BEGIN IMMEDIATE')
                await self.conn.execute("INSERT OR REPLACE INTO installation_settings VALUES('production',?)", ('1' if enabled else '0',))
                await self.conn.execute('INSERT INTO audit_log(actor_id,action,detail,timestamp) VALUES(?,?,?,?)',
                    (actor['id'],'production_mode','Enabled' if enabled else 'Disabled',now_text()))
                await self.conn.commit()
            except Exception:
                await self.conn.rollback()
                raise

    async def close(self):
        for database in self.databases.values():
            await database.close()
        self.databases.clear()
        if self.conn:
            await self.conn.close()
            self.conn = None

    async def create_admin(self, username, password):
        username = self.valid_username(username)
        hashed = await asyncio.to_thread(password_hash, password)
        async with self.lock:
            await self.conn.execute("INSERT INTO users VALUES(?,?,?,?,?,NULL,1,0,?)",
                (uuid.uuid4().hex, username, "Administrator", hashed, "admin", now_text()))
            await self.conn.commit()

    @staticmethod
    def valid_username(value):
        name = str(value).strip().casefold()
        if not 3 <= len(name) <= 80 or not all(c.isalnum() or c in "@._-" for c in name):
            raise ValueError("Username must have 3–80 letters, digits, dots, @, dashes or underscores.")
        return name

    async def create_farmer(self, data, actor):
        username = self.valid_username(data.get("username", ""))
        name = str(data.get("display_name", "")).strip()
        farm_name = str(data.get("farm_name", "")).strip()
        if not name or not farm_name or len(name) > 120 or len(farm_name) > 120:
            raise ValueError("Enter a farmer name and farm name (up to 120 characters each).")
        hashed = await asyncio.to_thread(password_hash, str(data.get("password", "")))
        farm_id, user_id = uuid.uuid4().hex, uuid.uuid4().hex
        async with self.lock:
            try:
                await self.conn.execute("BEGIN IMMEDIATE")
                await self.conn.execute("INSERT INTO farms VALUES(?,?,?)", (farm_id, farm_name, now_text()))
                await self.conn.execute("INSERT INTO users VALUES(?,?,?,?,?,?,1,1,?)",
                    (user_id, username, name, hashed, "user", farm_id, now_text()))
                await self.conn.execute("INSERT INTO audit_log(actor_id,farm_id,action,detail,timestamp) VALUES(?,?,?,?,?)",
                    (actor["id"], farm_id, "create_account", username, now_text()))
                await self.conn.commit()
            except Exception as exc:
                await self.conn.rollback()
                if "UNIQUE" in str(exc):
                    raise ValueError("This username already exists.") from exc
                raise
        return {"id": user_id, "farm_id": farm_id, "username": username}

    async def login(self, username, password, client):
        key = (client, str(username).casefold())
        failures = [t for t in self.login_attempts.get(key, []) if t > time.time() - 900]
        self.login_attempts = {k: v for k, v in self.login_attempts.items() if v and v[-1] > time.time() - 900}
        if len(failures) >= 10:
            raise ValueError("Too many sign-in attempts. Try again in 15 minutes.")
        users = await self.rows("SELECT * FROM users WHERE username=? AND active=1", (str(username).strip().casefold(),))
        # Compute a password hash even for absent users to avoid a quick existence check.
        stored = users[0]["password_hash"] if users else "scrypt$" + "00" * 16 + "$" + "00" * 32
        valid = await asyncio.to_thread(password_matches, str(password)[:257], stored)
        if not users or not valid:
            self.login_attempts[key] = failures + [time.time()]
            raise ValueError("Username or password is incorrect.")
        self.login_attempts.pop(key, None)
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        async with self.lock:
            await self.conn.execute("INSERT INTO login_sessions VALUES(?,?,?,?)",
                (digest(token), users[0]["id"], csrf, time.time() + 8 * 3600))
            await self.conn.commit()
        return token, csrf, self.public_user(users[0])

    async def authenticate(self, token):
        if not token or not self.conn:
            return None
        rows = await self.rows("""SELECT u.*,s.csrf,s.expires_at FROM login_sessions s
            JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND s.expires_at>? AND u.active=1""", (digest(token), time.time()))
        return rows[0] if rows else None

    @staticmethod
    def public_user(user):
        return {k: user[k] for k in ("id", "username", "display_name", "role", "farm_id", "active", "must_change_password")}

    async def logout(self, token):
        async with self.lock:
            await self.conn.execute("DELETE FROM login_sessions WHERE token_hash=?", (digest(token),))
            await self.conn.commit()

    async def change_password(self, user, data):
        if not await asyncio.to_thread(password_matches, str(data.get("current_password", "")), user["password_hash"]):
            raise ValueError("Current password is incorrect.")
        hashed = await asyncio.to_thread(password_hash, str(data.get("new_password", "")))
        async with self.lock:
            await self.conn.execute("UPDATE users SET password_hash=?,must_change_password=0 WHERE id=?", (hashed, user["id"]))
            await self.conn.execute("DELETE FROM login_sessions WHERE user_id=?", (user["id"],))
            await self.conn.commit()

    async def update_account(self, user_id, data, actor):
        users = await self.rows("SELECT * FROM users WHERE id=? AND role='user'", (user_id,))
        if not users:
            raise ValueError("Farmer account not found.")
        active = bool(data.get("active", users[0]["active"]))
        hashed = await asyncio.to_thread(password_hash, str(data["password"])) if data.get("password") else None
        async with self.lock:
            if hashed:
                await self.conn.execute("UPDATE users SET password_hash=?,must_change_password=1 WHERE id=?", (hashed, user_id))
            await self.conn.execute("UPDATE users SET active=? WHERE id=?", (int(active), user_id))
            await self.conn.execute("DELETE FROM login_sessions WHERE user_id=?", (user_id,))
            await self.conn.commit()
        await self.audit(actor["id"], users[0]["farm_id"], "update_account", "password_reset" if hashed else f"active={active}")
        return {"status": "updated"}

    async def audit(self, actor_id, farm_id, action, detail):
        async with self.lock:
            await self.conn.execute("INSERT INTO audit_log(actor_id,farm_id,action,detail,timestamp) VALUES(?,?,?,?,?)",
                (actor_id, farm_id, action, detail[:1000], now_text()))
            await self.conn.commit()

    async def get_database(self, farm_id):
        async with self.database_lock:
            if farm_id not in self.databases:
                if not await self.rows("SELECT id FROM farms WHERE id=?", (farm_id,)):
                    raise ValueError("Farm not found.")
                path = self.legacy_path if farm_id == "legacy" else self.farm_directory / (farm_id + ".db")
                backup = path.with_suffix(".pre-farm.bak")
                if path.exists() and not backup.exists():
                    def back_up():
                        source = sqlite3.connect(str(path))
                        destination = sqlite3.connect(str(backup))
                        try:
                            source.backup(destination)
                        finally:
                            source.close()
                            destination.close()
                    await asyncio.to_thread(back_up)
                database = FarmDatabase(path)
                await database.connect()
                self.databases[farm_id] = database
            return self.databases[farm_id]

    def media_directory(self, farm_id):
        if not farm_id or (farm_id != "legacy" and not (len(farm_id) == 32 and all(c in "0123456789abcdef" for c in farm_id))):
            raise ValueError("Invalid farm.")
        directory = self.path.parent / "media" / farm_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory


accounts = Accounts()
