"""Local administrator setup: python manage.py create-admin --username admin."""
import argparse
import asyncio
import getpass

from access import accounts


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["create-admin"])
    parser.add_argument("--username", required=True)
    args = parser.parse_args()
    password = getpass.getpass("Administrator password (at least 12 characters): ")
    if password != getpass.getpass("Confirm password: "):
        raise SystemExit("Passwords do not match.")
    await accounts.connect()
    try:
        await accounts.create_admin(args.username, password)
        print("Administrator created. Sign in at /login.")
    finally:
        await accounts.close()


if __name__ == "__main__":
    asyncio.run(main())
