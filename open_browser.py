"""Wait for AquaTrack server to start, then open the dashboard in default browser."""
import time
import urllib.request
import webbrowser


def main():
    target_url = "http://localhost:8000"
    check_url = "http://127.0.0.1:8000/login"
    deadline = time.time() + 60.0

    while time.time() < deadline:
        try:
            req = urllib.request.Request(check_url, headers={"User-Agent": "AquaTrack-BrowserLauncher"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status in (200, 301, 302, 307, 308):
                    webbrowser.open(target_url)
                    return
        except Exception:
            pass
        time.sleep(0.5)

    # Fallback if timeout reached
    webbrowser.open(target_url)


if __name__ == "__main__":
    main()
