"""Authentication, permissions and new management endpoints."""
import hashlib
import hmac
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from fastapi import HTTPException, Request,Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

from access import (accounts, current_actor, current_database, current_farm,
                    current_session, digest)
from detection_profile import DETECTORS, fingerprint, load_profile, public_value

ROOT = Path(__file__).parent
COOKIE = "tilapia_session"


def same_origin(request):
    origin = request.headers.get("origin")
    if not origin:
        return True
    parsed = urlparse(origin)
    return parsed.scheme in ("http", "https") and parsed.netloc == request.headers.get("host")


def validate_source(source, farm_id):
    source = str(source).strip()
    if not source:
        return ""
    if source.isdigit():
        if int(source) > 20:
            raise ValueError("Camera number must be between 0 and 20.")
        return str(int(source))
    if source.startswith(("rtsp://", "rtsps://")):
        parsed=urlparse(source)
        if not parsed.hostname:
            raise ValueError("Enter a valid camera stream address.")
        try:
            if parsed.port is not None and parsed.port==0:
                raise ValueError()
        except ValueError:
            raise ValueError('Enter a valid camera stream port.')
        return source
    directory = accounts.media_directory(farm_id)
    candidate = Path(source)
    if not candidate.is_absolute():
        candidate = directory / candidate.name
    resolved = candidate.resolve()
    if resolved.parent == directory.resolve() and resolved.is_file():
        return str(resolved)
    if farm_id == "legacy" and resolved == (ROOT / "sample.mp4").resolve() and resolved.is_file():
        return str(resolved)
    raise ValueError("Choose your farm's uploaded video, a camera number, or an RTSP camera address.")


async def websocket_context(ws, factory):
    user = await accounts.authenticate(ws.cookies.get(COOKIE))
    if not user or user["must_change_password"] or not same_origin(ws) or not hmac.compare_digest(ws.query_params.get("csrf", ""), user["csrf"]):
        await ws.close(code=4401)
        return None
    farm_id = user["farm_id"] if user["role"] == "user" else ws.query_params.get("farm_id")
    if user["role"] == "user" and ws.query_params.get("farm_id") not in (None, user["farm_id"]):
        await ws.close(code=4403)
        return None
    if not farm_id:
        await ws.close(code=4403)
        return None
    try:
        database = await accounts.get_database(farm_id)
    except ValueError:
        await ws.close(code=4404)
        return None
    return [(current_actor, current_actor.set(user)), (current_farm, current_farm.set(farm_id)),
        (current_database, current_database.set(database)), (current_session, current_session.set(factory()))]


def validate_tank_selection(body, farm_id, production=False):
    body=dict(body)
    if 'source' in body:
        choice=body['source']
        if 'camera_source' in body or not isinstance(choice,dict) or choice.get('type') not in ('none','usb','rtsp','video'):
            raise ValueError('Choose one valid tank source.')
        kind=choice['type']; value=str(choice.get('value','')).strip()
        if kind=='none':
            if value:
                raise ValueError('No source must have an empty value.')
        else:
            value=validate_source(value,farm_id)
            if kind=='usb' and not value.isdigit() or kind=='rtsp' and not value.startswith(('rtsp://','rtsps://')):
                raise ValueError('The source does not match the selected camera type.')
            if kind=='video' and (not value or value.isdigit() or value.startswith(('rtsp://','rtsps://'))):
                raise ValueError("Choose your farm's uploaded video.")
            if production and kind=='video':
                raise ValueError('Production Mode accepts only live cameras.')
        body['source']={'type':kind,'value':value}
    elif 'camera_source' in body:
        body['camera_source']=validate_source(body['camera_source'],farm_id)
    return body


def register(app, state, state_factory, session_states, db):
    @app.middleware("http")
    async def protected_farms(request: Request, call_next):
        path = request.url.path
        public = path in ("/", "/landing", "/login", "/api/auth/login", "/api/auth/status") or path.startswith("/static/")
        if public:
            if request.method not in ("GET", "HEAD") and not same_origin(request):
                return JSONResponse({"detail": "Use the sign-in page on this server."}, status_code=403)
            return await call_next(request)
        user = await accounts.authenticate(request.cookies.get(COOKIE))
        if not user:
            if path.startswith(("/api/", "/media/", "/uploads/")):
                return JSONResponse({"detail": "Sign in to continue."}, status_code=401)
            return RedirectResponse("/login", status_code=303)
        if user["must_change_password"] and path not in ("/api/auth/me", "/api/auth/password", "/api/auth/logout", "/user", "/admin"):
            return JSONResponse({"detail": "Change your initial password before using your farm."}, status_code=403)
        if path.startswith("/admin") and user["role"] != "admin":
            return RedirectResponse("/user", status_code=303)
        if path in ("/app", "/dashboard", "/counter"):
            return RedirectResponse("/admin" if user["role"] == "admin" else "/user", status_code=303)
        admin_only = path.startswith(("/api/admin/", "/api/evaluate", "/api/evaluation-benchmarks"))
        if admin_only and user["role"] != "admin":
            return JSONResponse({"detail": "Administrator access is required."}, status_code=403)
        production = getattr(request.app.state,'production',None)
        if production and production.enabled and (path.startswith('/api/upload/') or path in
                ('/api/tanks/upload-video','/api/reprocess-current','/api/evaluate-sample')):
            return JSONResponse({'detail':'Production Mode accepts only live cameras.'},status_code=409)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            if not same_origin(request) or not hmac.compare_digest(request.headers.get("x-csrf-token", ""), user["csrf"]):
                return JSONResponse({"detail": "Refresh the page and try again."}, status_code=403)
        if path.startswith("/api/models") or path in ("/api/database/reset", "/api/stats/model-stats", "/api/analytics/model-stats",
                "/api/evaluate", "/api/evaluate/from-records", "/api/evaluate/upload-dataset"):
            return JSONResponse({"detail": "This technical control is no longer available."}, status_code=410)
        requested = request.headers.get("x-farm-id") or request.query_params.get("farm_id")
        if user["role"] == "user" and requested not in (None, user["farm_id"]):
            return JSONResponse({"detail": "Farm not found."}, status_code=404)
        farm_id = user["farm_id"] if user["role"] == "user" else requested
        no_farm = path.startswith(("/api/auth/", "/api/admin/")) or path in ("/admin", "/user", "/api/engine",'/api/production')
        if not farm_id and not no_farm and not admin_only:
            return JSONResponse({"detail": "Select a farm before opening its records."}, status_code=400)
        # Benchmarks without a farm are stored in the Admin-owned legacy farm.
        farm_id = farm_id or "legacy"
        try:
            database = await accounts.get_database(farm_id)
        except ValueError:
            return JSONResponse({"detail": "Farm not found."}, status_code=404)
        session_key = (digest(request.cookies[COOKIE]), farm_id)
        for key, (_, expiry) in list(session_states.items()):
            if expiry <= time.time():
                session_states.pop(key, None)
        if session_key not in session_states:
            session_states[session_key] = (state_factory(), user["expires_at"])
        contexts = [(current_actor, current_actor.set(user)), (current_database, current_database.set(database)),
                    (current_farm, current_farm.set(farm_id)), (current_session, current_session.set(session_states[session_key][0]))]
        try:
            if request.method in ("POST", "PUT", "PATCH") and "application/json" in request.headers.get("content-type", ""):
                try:
                    body = await request.json()
                except (ValueError, UnicodeDecodeError):
                    return JSONResponse({"detail": "Send valid JSON."}, status_code=400)
                if not isinstance(body, dict):
                    return JSONResponse({"detail": "Send a JSON object."}, status_code=400)
                if not path.startswith(("/api/auth/", "/api/admin/")) and set(body) & {"conf", "iou", "iou_thresh", "active_models", "active", "models", "mode"}:
                    return JSONResponse({"detail": "Counting settings are fixed by calibration."}, status_code=422)
                if "camera_source" in body:
                    try:
                        source=validate_source(body["camera_source"], farm_id)
                        from production import is_live
                        if production and production.enabled and source and not is_live(source):
                            raise ValueError('Production Mode accepts only live cameras.')
                    except ValueError as exc:
                        return JSONResponse({"detail": str(exc)}, status_code=400)
            response = await call_next(request)
            if response.status_code < 400 and request.method not in ("GET", "HEAD", "OPTIONS") and not path.startswith("/api/auth/"):
                # Record identifiers and operation values, never passwords or camera credentials.
                detail = {"path": path, "method": request.method}
                if "body" in locals() and not path.startswith("/api/admin/"):
                    detail["values"] = {k: v for k, v in body.items() if k not in ("password", "camera_source",'live_source','source', "current_password", "new_password")}
                await accounts.audit(user["id"], farm_id, "farm_operation", json.dumps(detail))
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                content = b"".join([chunk async for chunk in response.body_iterator])
                headers = dict(response.headers)
                headers.pop("content-length", None)
                response = JSONResponse(public_value(json.loads(content)), status_code=response.status_code, headers=headers)
            elif "text/csv" in content_type:
                iterator = response.body_iterator
                async def anonymous_csv():
                    async for chunk in iterator:
                        yield public_value(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk).replace("model_name", "engine")
                headers = dict(response.headers)
                headers.pop("content-length", None)
                response = StreamingResponse(anonymous_csv(), status_code=response.status_code, media_type="text/csv", headers=headers)
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
            return response
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        finally:
            for variable, token in reversed(contexts):
                variable.reset(token)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page():
        return HTMLResponse((ROOT / "templates/login.html").read_text(encoding="utf-8"))

    @app.get("/admin", response_class=HTMLResponse)
    @app.get("/user", response_class=HTMLResponse)
    async def role_page():
        return HTMLResponse((ROOT / "templates/index.html").read_text(encoding="utf-8"))

    @app.get("/api/auth/status")
    async def setup_status():
        return {"configured": bool(await accounts.rows("SELECT id FROM users WHERE role='admin'"))}

    @app.post("/api/auth/login")
    async def login(request: Request, body: dict):
        try:
            token, csrf, user = await accounts.login(body.get("username", ""), body.get("password", ""), request.client.host)
        except ValueError as exc:
            raise HTTPException(401, str(exc))
        response = JSONResponse({"user": user, "csrf": csrf})
        response.set_cookie(COOKIE, token, httponly=True, secure=request.url.scheme == "https", samesite="strict", max_age=8 * 3600)
        return response

    @app.get("/api/auth/me")
    async def me():
        user = current_actor.get()
        return {"user": accounts.public_user(user), "csrf": user["csrf"]}

    @app.post("/api/auth/logout")
    async def logout(request: Request):
        await accounts.logout(request.cookies[COOKIE])
        for key in list(session_states):
            if key[0] == digest(request.cookies[COOKIE]):
                session_states.pop(key, None)
        response = JSONResponse({"status": "signed_out"})
        response.delete_cookie(COOKIE)
        return response

    @app.post("/api/auth/password")
    async def change_password(body: dict):
        await accounts.change_password(current_actor.get(), body)
        response = JSONResponse({"status": "updated", "sign_in_required": True})
        response.delete_cookie(COOKIE)
        return response

    @app.get("/api/admin/accounts")
    async def list_accounts():
        return await accounts.rows("""SELECT u.id,u.username,u.display_name,u.role,u.farm_id,u.active,
            u.must_change_password,f.name AS farm_name FROM users u LEFT JOIN farms f ON f.id=u.farm_id ORDER BY u.created_at""")

    @app.post("/api/admin/accounts")
    async def create_account(body: dict):
        return await accounts.create_farmer(body, current_actor.get())

    @app.put("/api/admin/accounts/{user_id}")
    async def edit_account(user_id: str, body: dict):
        return await accounts.update_account(user_id, body, current_actor.get())

    @app.get("/api/admin/farms")
    async def list_farms():
        return await accounts.rows("SELECT * FROM farms ORDER BY created_at")

    @app.get("/api/admin/dashboard")
    async def admin_dashboard():
        farms = await list_farms()
        for farm in farms:
            database = await accounts.get_database(farm["id"])
            farm["summary"] = (await database.get_production_report())["summary"]
        numerator = sum(f["summary"]["mortality_count"] for f in farms if f["summary"]["mortality_rate_pct"] is not None)
        denominator = sum(f["summary"]["mortality_denominator"] for f in farms)
        return {"farms": farms, "summary": {"total_farms": len(farms),
            "total_population": sum(f["summary"]["total_population"] for f in farms),
            "total_feed_kg": round(sum(f["summary"]["total_feed_kg"] for f in farms), 3),
            "mortality_count": sum(f["summary"]["mortality_count"] for f in farms),
            "mortality_rate_pct": round(numerator / denominator * 100, 2) if denominator else None,
            "low_feed_items": sum(f["summary"]["low_feed_items"] for f in farms),
            "missing_checks": sum(f["summary"]["mortality_expected_tanks"] - f["summary"]["mortality_checked_tanks"] for f in farms)}}

    @app.get("/api/admin/audit")
    async def audit_log():
        return await accounts.rows("""SELECT a.*,u.username,f.name AS farm_name FROM audit_log a
            LEFT JOIN users u ON u.id=a.actor_id LEFT JOIN farms f ON f.id=a.farm_id ORDER BY a.id DESC LIMIT 200""")

    @app.get("/api/engine")
    async def engine_status():
        loaded = len([name for name in DETECTORS if name in state.model_pool])
        return {"status": "ready" if loaded == len(DETECTORS) else "degraded" if loaded else "unavailable",
            "calibrated": bool(load_profile().get("calibrated")), "profile": fingerprint(), "counting": "Automatic combined counting",
            "stride": state.inference_stride, "stream_max_dimension": state.stream_max_dimension}

    @app.get('/api/production')
    async def production_status():
        # Operating mode is installation-wide; no other farm details are exposed.
        return {'enabled':app.state.production.enabled,'startup_configured':(accounts.path.parent/'startup-installed.json').is_file()}

    @app.get('/api/admin/production')
    async def admin_production():
        return await production_status()

    @app.put('/api/admin/production')
    async def change_production(body:dict):
        if type(body.get('enabled')) is not bool:
            raise ValueError('Production Mode must be enabled or disabled.')
        await app.state.production.set_enabled(body['enabled'],current_actor.get())
        return await production_status()

    @app.get('/api/monitoring')
    async def monitoring_status():
        return await app.state.production.status(current_farm.get())

    @app.put('/api/tanks/{tank_id}/monitoring')
    async def configure_monitoring(tank_id:str,body:dict):
        from production import is_live
        if 'enabled' in body and type(body['enabled']) is not bool:
            raise ValueError('Choose whether monitoring is running or paused.')
        source=body.get('live_source')
        if source is not None:
            source=validate_source(source,current_farm.get())
            if not is_live(source):
                raise ValueError('Use a live camera number or RTSP address.')
        database=current_database.get()
        await database.configure_monitoring(tank_id,enabled=body.get('enabled'),live_source=source)
        await app.state.production.sync()
        return await app.state.production.status(current_farm.get())

    @app.post('/api/tanks/{tank_id}/validate-census')
    async def validate_census(tank_id:str,body:dict):
        await app.state.production.validate(current_farm.get(),tank_id,body.get('known_count'),body.get('whole_view'))
        return {'status':'validating'}

    @app.get('/api/census-events')
    async def census_events(tank_id:str=None,unresolved:bool=False,limit:int=Query(200,ge=1,le=2000),offset:int=Query(0,ge=0)):
        if tank_id and not await db.get_tank(tank_id):
            raise ValueError('Tank not found.')
        return await db.census_history(tank_id,unresolved,limit,offset)

    @app.get("/api/food")
    async def food():
        return await db.get_food()

    @app.post("/api/food/items")
    async def create_food(body: dict):
        return await db.create_feed_item(body)

    @app.post("/api/food/transactions")
    async def food_transaction(body: dict):
        return await db.feed_transaction(body)

    @app.post("/api/tanks/{tank_id}/stock")
    async def stock(tank_id: str, body: dict):
        return await db.stock_fish(tank_id, body)

    @app.post("/api/tanks/{tank_id}/mortality")
    async def mortality(tank_id: str, body: dict):
        return await db.record_mortality(tank_id, body)

    @app.get("/api/media")
    async def media_list():
        directory = accounts.media_directory(current_farm.get())
        return [{"name": file.name, "source": str(file), "url": f"/media/{file.name}?farm_id={current_farm.get()}"}
                for file in directory.iterdir() if file.is_file() and file.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm")]

    @app.get("/media/{filename}")
    async def media_file(filename: str):
        directory = accounts.media_directory(current_farm.get()).resolve()
        file = (directory / filename).resolve()
        if file.parent != directory or not file.is_file():
            raise HTTPException(404, "Video not found.")
        return FileResponse(file)
