"""
delete_records.py — CLI utility to delete records from the Tilapia Web App database
====================================================================================
Usage (run from the project root, ideally with the project venv activated):

    python delete_records.py                 # Wipe ALL records (interactive confirmation)
    python delete_records.py --yes            # Wipe ALL records without asking
    python delete_records.py --events-only    # Only detection events, boxes & evaluations
    python delete_records.py --tanks          # Also delete tank definitions (all farm data)
    python delete_records.py --tank TANK-02   # Delete a single tank and all its records
    python delete_records.py --db path/to.db  # Target a different database file

Scopes
------
default        All detection_events, bounding_boxes, evaluation_runs,
               evaluation_benchmarks, dispersals, tank_production_logs,
               sessions; tank current_count reset to 0. Tanks themselves are kept.
--events-only  Only bounding_boxes, detection_events, evaluation_runs,
               evaluation_benchmarks (tanks, dispersals and logs untouched).
--tanks        Everything above PLUS the tanks table itself
               (tank definitions are removed; they are re-seeded on next app start).
--tank ID      Delete one tank plus its detection events, bounding boxes,
               dispersals and production logs.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running from anywhere: make the project root importable
sys.path.insert(0, str(Path(__file__).parent))

from database import Database


TABLE_LABELS = [
    ("detection_events", "Detection events"),
    ("bounding_boxes", "Bounding boxes"),
    ("evaluation_runs", "Evaluation runs"),
    ("evaluation_benchmarks", "Evaluation benchmarks"),
    ("dispersals", "Dispersals"),
    ("tank_production_logs", "Tank production logs"),
    ("sessions", "Sessions"),
    ("tanks", "Tanks"),
]


async def fetch_counts(db: Database, tables: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in tables:
        cursor = await db._conn.execute(f"SELECT COUNT(*) AS c FROM {table}")
        row = await cursor.fetchone()
        counts[table] = int(row["c"]) if row else 0
    return counts


def print_counts(counts: dict[str, int]) -> None:
    print()
    print("Current record counts")
    print("-" * 42)
    for table, label in TABLE_LABELS:
        if table in counts:
            print(f"  {label:<28} {counts[table]:>10,}")
    print("-" * 42)
    total = sum(counts.values())
    print(f"  {'TOTAL':<28} {total:>10,}")
    print()


def confirm(prompt: str) -> bool:
    answer = input(f"{prompt} Type 'DELETE' to confirm: ").strip()
    return answer == "DELETE"


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete records from the Tilapia Web App SQLite database."
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to the SQLite database file (defaults to tilapia_web_analytics.db).",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    parser.add_argument(
        "--events-only",
        action="store_true",
        help="Delete only detection events, bounding boxes and evaluation records. "
             "Farm data (tanks, dispersals, production logs) is kept.",
    )
    parser.add_argument(
        "--tanks",
        action="store_true",
        help="Delete ALL records including tank definitions "
             "(tanks are re-seeded with defaults on next app start).",
    )
    parser.add_argument(
        "--tank",
        default=None,
        metavar="TANK_ID",
        help="Delete a single tank (e.g. TANK-02) plus all of its associated records.",
    )
    parser.add_argument(
        "--list-tanks",
        action="store_true",
        help="List all tanks with their current counts and exit.",
    )
    args = parser.parse_args()

    db = Database(Path(args.db)) if args.db else Database()
    await db.connect()

    try:
        # ------------------------------------------------------------------
        # Single tank deletion
        # ------------------------------------------------------------------
        if args.tank:
            tank = await db.get_tank(args.tank)
            if not tank:
                print(f"[ERROR] Tank '{args.tank}' not found.")
                available = await db.get_tanks()
                if available:
                    print("Available tanks:")
                    for t in available:
                        print(f"  - {t['tank_id']}: {t['name']} (count={t['current_count']})")
                return 1

            print(f"Tank to delete: {tank['tank_id']} — {tank['name']}")
            print(f"  current_count     = {tank['current_count']}")
            print(f"  max_capacity      = {tank['max_capacity']}")
            print("  This also removes the tank's detection events, bounding boxes, "
                  "dispersals and production logs.")
            if not args.yes and not confirm(f"Delete tank '{args.tank}' and ALL of its records?"):
                print("Aborted. Nothing was deleted.")
                return 0

            deleted = await db.delete_tank(args.tank)
            if deleted:
                print(f"[OK] Tank '{args.tank}' and all associated records were deleted.")
                await db._conn.execute("VACUUM")
                return 0
            print(f"[ERROR] Failed to delete tank '{args.tank}'.")
            return 1

        # ------------------------------------------------------------------
        # Listing only
        # ------------------------------------------------------------------
        if args.list_tanks:
            tanks = await db.get_tanks()
            print(f"{len(tanks)} tank(s):")
            for t in tanks:
                print(f"  - {t['tank_id']}: {t['name']} "
                      f"(count={t['current_count']}, capacity={t['max_capacity']}, status={t['status']})")
            return 0

        # ------------------------------------------------------------------
        # Full / partial wipe
        # ------------------------------------------------------------------
        if args.events_only and args.tanks:
            print("[ERROR] --events-only and --tanks are mutually exclusive.")
            return 1

        if args.events_only:
            scope_tables = ["bounding_boxes", "detection_events",
                            "evaluation_runs", "evaluation_benchmarks"]
        else:
            scope_tables = [t for t, _ in TABLE_LABELS]

        counts = await fetch_counts(db, scope_tables)
        print_counts(counts)

        if all(v == 0 for v in counts.values()):
            print("Nothing to delete — all selected tables are already empty.")
            return 0

        if args.events_only:
            print("Scope: detection events, bounding boxes and evaluation records ONLY.")
        elif args.tanks:
            print("Scope: ALL records including tank definitions.")
        else:
            print("Scope: all records. Tank definitions are kept; counts reset to 0.")

        if not args.yes and not confirm("Permanently delete the records listed above?"):
            print("Aborted. Nothing was deleted.")
            return 0

        if args.events_only:
            await db.delete_all_events()
            print("[OK] Detection events, bounding boxes and evaluation records deleted.")
        else:
            await db.reset_all_data()
            print("[OK] All records deleted; tank counts reset to 0.")
            if args.tanks:
                async with db._lock:
                    await db._conn.execute("DELETE FROM tanks")
                    await db._conn.commit()
                print("[OK] Tank definitions deleted (defaults are re-seeded on next app start).")

        # Show the result
        counts_after = await fetch_counts(db, scope_tables)
        print()
        print("After deletion")
        print("-" * 42)
        for table, label in TABLE_LABELS:
            if table in counts_after:
                print(f"  {label:<28} {counts_after[table]:>10,}")
        print("-" * 42)
        return 0
    finally:
        await db.close()


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nAborted. Nothing was deleted.")
        sys.exit(0)
