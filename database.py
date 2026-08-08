"""
database.py  —  SQLite storage for TrafficGuard
"""

import sqlite3
from datetime import datetime, timedelta
from config import DATABASE_PATH          # DB_PATH alias defined in config.py


def get_conn():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id      TEXT    NOT NULL,
            location    TEXT    DEFAULT '',
            obj_id      INTEGER,
            vtype       TEXT    NOT NULL,
            vehicle     TEXT    DEFAULT 'unknown',
            plate       TEXT    DEFAULT '--',
            confidence  REAL    DEFAULT 0.0,
            evidence    TEXT    DEFAULT '',
            reviewed    INTEGER DEFAULT 0,
            timestamp   TEXT    DEFAULT (datetime('now','localtime'))
        )
    """)
    conn.commit()
    conn.close()
    print("[DB] Ready:", DATABASE_PATH)


def insert_violation(cam_id, location, obj_id, vtype, vehicle="unknown",
                     confidence=0.0, evidence_path=""):
    conn = get_conn()
    conn.execute("""
        INSERT INTO violations (cam_id, location, obj_id, vtype, vehicle,
                                confidence, evidence, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (cam_id, location, obj_id, vtype, vehicle, confidence,
          evidence_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


def get_violations(limit=50, cam_id=None, vtype=None):
    conn = get_conn()
    q = "SELECT * FROM violations WHERE 1=1"
    params = []
    if cam_id:
        q += " AND cam_id=?"; params.append(cam_id)
    if vtype:
        q += " AND vtype=?"; params.append(vtype)
    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    """Return today's violation counts and pending reviews."""
    conn = get_conn()
    today = datetime.now().strftime("%Y-%m-%d")
    total = conn.execute(
        "SELECT COUNT(*) FROM violations WHERE DATE(timestamp)=?", (today,)
    ).fetchone()[0]
    by_type = conn.execute(
        "SELECT vtype, COUNT(*) as cnt FROM violations "
        "WHERE DATE(timestamp)=? GROUP BY vtype", (today,)
    ).fetchall()
    pending = conn.execute(
        "SELECT COUNT(*) FROM violations WHERE reviewed=0"
    ).fetchone()[0]
    conn.close()
    return {
        "total_today": total,
        "pending_reviews": pending,
        "by_type": [{"type": r["vtype"], "count": r["cnt"]} for r in by_type],
    }


# Backward-compat alias (old name used in some modules)
get_stats_today = get_stats


def mark_reviewed(vid):
    conn = get_conn()
    conn.execute("UPDATE violations SET reviewed=1 WHERE id=?", (vid,))
    conn.commit()
    conn.close()


def get_daily_analytics(date=None):
    """Get detailed analytics for a specific day."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    conn = get_conn()
    
    # Hourly breakdown
    hourly = conn.execute("""
        SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
        FROM violations 
        WHERE DATE(timestamp)=?
        GROUP BY hour
        ORDER BY hour
    """, (date,)).fetchall()
    
    # By type for the day
    by_type = conn.execute("""
        SELECT vtype, COUNT(*) as count
        FROM violations 
        WHERE DATE(timestamp)=?
        GROUP BY vtype
    """, (date,)).fetchall()
    
    # By camera for the day
    by_camera = conn.execute("""
        SELECT cam_id, location, COUNT(*) as count
        FROM violations 
        WHERE DATE(timestamp)=?
        GROUP BY cam_id, location
    """, (date,)).fetchall()
    
    # Peak hour
    peak_hour = conn.execute("""
        SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
        FROM violations 
        WHERE DATE(timestamp)=?
        GROUP BY hour
        ORDER BY count DESC
        LIMIT 1
    """, (date,)).fetchone()
    
    conn.close()
    
    return {
        "date": date,
        "hourly_breakdown": [{"hour": h["hour"], "count": h["count"]} for h in hourly],
        "by_type": [{"type": t["vtype"], "count": t["count"]} for t in by_type],
        "by_camera": [{"cam_id": c["cam_id"], "location": c["location"], "count": c["count"]} for c in by_camera],
        "peak_hour": {"hour": peak_hour["hour"], "count": peak_hour["count"]} if peak_hour else None,
        "total_violations": sum(h["count"] for h in hourly)
    }


def get_weekly_analytics(weeks_back=0):
    """Get detailed analytics for a specific week."""
    # Calculate the start and end of the week
    today = datetime.now()
    start_of_week = today - timedelta(days=today.weekday() + (7 * weeks_back))
    end_of_week = start_of_week + timedelta(days=6)
    
    start_date = start_of_week.strftime("%Y-%m-%d")
    end_date = end_of_week.strftime("%Y-%m-%d")
    
    conn = get_conn()
    
    # Daily breakdown for the week
    daily = conn.execute("""
        SELECT DATE(timestamp) as date, COUNT(*) as count
        FROM violations 
        WHERE DATE(timestamp) BETWEEN ? AND ?
        GROUP BY DATE(timestamp)
        ORDER BY date
    """, (start_date, end_date)).fetchall()
    
    # By type for the week
    by_type = conn.execute("""
        SELECT vtype, COUNT(*) as count
        FROM violations 
        WHERE DATE(timestamp) BETWEEN ? AND ?
        GROUP BY vtype
    """, (start_date, end_date)).fetchall()
    
    # By camera for the week
    by_camera = conn.execute("""
        SELECT cam_id, location, COUNT(*) as count
        FROM violations 
        WHERE DATE(timestamp) BETWEEN ? AND ?
        GROUP BY cam_id, location
    """, (start_date, end_date)).fetchall()
    
    # Peak day
    peak_day = conn.execute("""
        SELECT DATE(timestamp) as date, COUNT(*) as count
        FROM violations 
        WHERE DATE(timestamp) BETWEEN ? AND ?
        GROUP BY DATE(timestamp)
        ORDER BY count DESC
        LIMIT 1
    """, (start_date, end_date)).fetchone()
    
    conn.close()
    
    return {
        "week_start": start_date,
        "week_end": end_date,
        "daily_breakdown": [{"date": d["date"], "count": d["count"]} for d in daily],
        "by_type": [{"type": t["vtype"], "count": t["count"]} for t in by_type],
        "by_camera": [{"cam_id": c["cam_id"], "location": c["location"], "count": c["count"]} for c in by_camera],
        "peak_day": {"date": peak_day["date"], "count": peak_day["count"]} if peak_day else None,
        "total_violations": sum(d["count"] for d in daily)
    }


def get_monthly_analytics(months_back=0):
    """Get detailed analytics for a specific month."""
    today = datetime.now()
    year = today.year
    month = today.month - months_back
    
    # Handle year rollover
    while month <= 0:
        month += 12
        year -= 1
    
    start_date = f"{year}-{month:02d}-01"
    
    # Get the last day of the month
    if month == 12:
        end_date = f"{year + 1}-01-01"
    else:
        end_date = f"{year}-{month + 1:02d}-01"
    
    conn = get_conn()
    
    # Daily breakdown for the month
    daily = conn.execute("""
        SELECT DATE(timestamp) as date, COUNT(*) as count
        FROM violations 
        WHERE DATE(timestamp) >= ? AND DATE(timestamp) < ?
        GROUP BY DATE(timestamp)
        ORDER BY date
    """, (start_date, end_date)).fetchall()
    
    # Weekly breakdown
    weekly = conn.execute("""
        SELECT strftime('%W', timestamp) as week, COUNT(*) as count
        FROM violations 
        WHERE DATE(timestamp) >= ? AND DATE(timestamp) < ?
        GROUP BY week
        ORDER BY week
    """, (start_date, end_date)).fetchall()
    
    # By type for the month
    by_type = conn.execute("""
        SELECT vtype, COUNT(*) as count
        FROM violations 
        WHERE DATE(timestamp) >= ? AND DATE(timestamp) < ?
        GROUP BY vtype
    """, (start_date, end_date)).fetchall()
    
    # By camera for the month
    by_camera = conn.execute("""
        SELECT cam_id, location, COUNT(*) as count
        FROM violations 
        WHERE DATE(timestamp) >= ? AND DATE(timestamp) < ?
        GROUP BY cam_id, location
    """, (start_date, end_date)).fetchall()
    
    # Peak day
    peak_day = conn.execute("""
        SELECT DATE(timestamp) as date, COUNT(*) as count
        FROM violations 
        WHERE DATE(timestamp) >= ? AND DATE(timestamp) < ?
        GROUP BY DATE(timestamp)
        ORDER BY count DESC
        LIMIT 1
    """, (start_date, end_date)).fetchone()
    
    conn.close()
    
    return {
        "month": f"{year}-{month:02d}",
        "month_start": start_date,
        "month_end": end_date,
        "daily_breakdown": [{"date": d["date"], "count": d["count"]} for d in daily],
        "weekly_breakdown": [{"week": w["week"], "count": w["count"]} for w in weekly],
        "by_type": [{"type": t["vtype"], "count": t["count"]} for t in by_type],
        "by_camera": [{"cam_id": c["cam_id"], "location": c["location"], "count": c["count"]} for c in by_camera],
        "peak_day": {"date": peak_day["date"], "count": peak_day["count"]} if peak_day else None,
        "total_violations": sum(d["count"] for d in daily)
    }


def get_time_based_analytics():
    """Get time-based patterns (peak hours, days, etc.) across all data."""
    conn = get_conn()
    
    # Hourly patterns (all time)
    hourly_patterns = conn.execute("""
        SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
        FROM violations 
        GROUP BY hour
        ORDER BY hour
    """).fetchall()
    
    # Day of week patterns
    day_patterns = conn.execute("""
        SELECT strftime('%w', timestamp) as day_of_week, COUNT(*) as count
        FROM violations 
        GROUP BY day_of_week
        ORDER BY day_of_week
    """).fetchall()
    
    # Monthly patterns
    monthly_patterns = conn.execute("""
        SELECT strftime('%Y-%m', timestamp) as month, COUNT(*) as count
        FROM violations 
        GROUP BY month
        ORDER BY month
    """).fetchall()
    
    conn.close()
    
    day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    
    return {
        "hourly_patterns": [{"hour": h["hour"], "count": h["count"]} for h in hourly_patterns],
        "day_patterns": [{"day": day_names[int(d["day_of_week"])], "day_of_week": int(d["day_of_week"]), "count": d["count"]} for d in day_patterns],
        "monthly_patterns": [{"month": m["month"], "count": m["count"]} for m in monthly_patterns],
    }


def get_comprehensive_analytics():
    """Get all analytics data for the chatbot."""
    return {
        "daily": get_daily_analytics(),
        "weekly": get_weekly_analytics(),
        "monthly": get_monthly_analytics(),
        "time_patterns": get_time_based_analytics(),
        "current_stats": get_stats()
    }


import os   # needed by init_db — placed here to avoid circular at top
