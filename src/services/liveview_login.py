"""LiveView remote login — interactive Google login inside the admin dashboard.

A dedicated, transient headless Chromium (nodriver/CDP) is launched per login
session with a persistent per-token profile directory. The admin watches the
browser through screenshots streamed over WebSocket and drives it with mouse /
keyboard events. After the user logs in, cookies are harvested from the login
browser and committed through the same validated path as the cookie refresh
(`TokenManager.commit_refreshed_cookies`).

Design notes (see ~/workspace/flow2api/liveview-port/SPEC.md):
- Chromium/nodriver is kept — no Firefox/Camoufox migration.
- The login browser is a separate process from the captcha pool browser, so
  resident tabs / captcha flows are never disturbed.
- No Google password, TOTP secret or code is ever accepted, stored or logged
  by the server: login is interactive, the server only harvests cookies.
"""

import asyncio
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from ..core.logger import debug_logger
from .browser_captcha_personal import (
    PERSONAL_RUNTIME_DATA_DIR,
    BrowserCaptchaService,
    _build_personal_browser_args,
    _compose_proxy_url,
    _create_proxy_auth_extension,
    _normalize_personal_browser_args_for_launch,
    _redact_proxy_url,
    _resolve_browser_executable_path,
    _resolve_personal_browser_sandbox_enabled,
)

LIVEVIEW_VIEWPORT_WIDTH = 1280
LIVEVIEW_VIEWPORT_HEIGHT = 800
LIVEVIEW_SESSION_TTL_SECONDS = 15 * 60
LIVEVIEW_LAUNCH_TIMEOUT_SECONDS = 60.0
LIVEVIEW_NAVIGATE_TIMEOUT_SECONDS = 30.0
LIVEVIEW_PAGE_TARGET_WAIT_SECONDS = 10.0
LIVEVIEW_COMMAND_TIMEOUT_SECONDS = 5.0
LIVEVIEW_FRAME_INTERVAL_SECONDS = 0.5
LIVEVIEW_STATE_POLL_SECONDS = 2.0
LIVEVIEW_START_URL = "https://flow.google.com/projects"

# Same identity probe as BrowserCaptchaService.refresh_google_cookies.
LIVEVIEW_LOGIN_IDENTITY_SCRIPT = (
    "({host: location.hostname, ready: document.readyState, "
    "email: (window.WIZ_global_data || {}).oPEP7c || ''})"
)

_LIVEVIEW_NAVIGATE_ALLOWED_HOST_SUFFIXES = (".google.com", ".flow.google.com")
_LIVEVIEW_NAVIGATE_ALLOWED_HOSTS = {"google.com", "flow.google.com"}

_KEY_TEXT_BY_KEY = {
    "Enter": "\r",
    "Tab": "\t",
}

_MOUSE_BUTTONS = {"left", "right", "middle", "back", "forward"}


class LiveViewError(Exception):
    """Base LiveView error carrying an HTTP status code."""

    status_code = 500


class LiveViewBusyError(LiveViewError):
    status_code = 409


class LiveViewUnavailableError(LiveViewError):
    status_code = 503


class LiveViewNotFoundError(LiveViewError):
    status_code = 404


class LiveViewNoSessionError(LiveViewError):
    status_code = 404


class LiveViewNotLoggedInError(LiveViewError):
    status_code = 409


class LiveViewLaunchError(LiveViewError):
    status_code = 502


def liveview_profile_dir(token_id: int) -> Path:
    """Persistent per-token browser profile directory (named volume /app/data)."""
    return (
        PERSONAL_RUNTIME_DATA_DIR / "browser_profiles" / f"token_{int(token_id)}"
    ).resolve()


async def _wait_for_page_target(browser, timeout_seconds: float):
    """Poll update_targets() until a page target exists (or timeout).

    nodriver's Browser.get() does a bare next() over page targets, so calling
    it right after uc.start() raises StopIteration (surfaced as RuntimeError
    'coroutine raised StopIteration') when the first tab has not registered
    yet. Callers must wait for a target before navigating.
    """
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        try:
            await browser.update_targets()
        except Exception:
            pass
        for target in getattr(browser, "targets", []) or []:
            if getattr(target, "type_", None) == "page":
                return target
        if time.monotonic() >= deadline:
            return None
        await asyncio.sleep(0.2)


@dataclass
class LiveViewSession:
    token_id: int
    session_id: str
    email: str = ""
    browser: Any = None
    tab: Any = None
    started_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    logged_in: bool = False
    observed_email: str = ""
    current_url: str = ""
    proxy_ext_dir: Optional[str] = None
    closed: bool = False
    ttl_task: Optional[asyncio.Task] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def status_dict(self) -> Dict[str, Any]:
        return {
            "active": not self.closed,
            "session_id": self.session_id,
            "token_id": self.token_id,
            "email": self.email,
            "logged_in": self.logged_in,
            "observed_email": self.observed_email,
            "url": self.current_url,
            "started_at": self.started_at,
            "expires_at": self.expires_at,
        }


class LiveViewLoginService:
    """Owns LiveView login sessions (at most one active server-wide)."""

    def __init__(self, db=None, token_manager=None):
        self.db = db
        self.token_manager = token_manager
        self._guard = asyncio.Lock()
        self._active: Optional[LiveViewSession] = None

    # ---------- session lifecycle ----------

    def get_active_session(self, token_id: Optional[int] = None) -> Optional[LiveViewSession]:
        session = self._active
        if session is None or session.closed:
            return None
        if token_id is not None and int(session.token_id) != int(token_id):
            return None
        return session

    async def start_session(self, token_id: int) -> Dict[str, Any]:
        """Launch the dedicated login browser for a token.

        Raises LiveViewBusyError (another session active), LiveViewNotFoundError
        (unknown token), LiveViewUnavailableError (captcha_method != personal),
        LiveViewLaunchError (Chromium failed to start).
        """
        token_id = int(token_id)
        async with self._guard:
            if self.get_active_session() is not None:
                active = self._active
                raise LiveViewBusyError(
                    "Já existe uma sessão de login remoto ativa "
                    f"(token_id={active.token_id}, session_id={active.session_id})"
                )
            if self.db is None:
                raise LiveViewUnavailableError("Banco de dados não disponível")
            token = await self.db.get_token(token_id)
            if token is None:
                raise LiveViewNotFoundError(f"Token {token_id} não encontrado")
            # NOTE: no captcha_method gate here on purpose. The login browser
            # is a dedicated nodriver process, independent of the global
            # captcha_method (production runs 'browser'); only the proxy
            # resolution below reuses the personal service helpers.
            session = await self._launch_locked(token)
            self._active = session
            session.ttl_task = asyncio.create_task(self._session_ttl_watchdog(session))
            return {
                "session_id": session.session_id,
                "token_id": session.token_id,
                "email": session.email,
                "status": "ready",
                "viewport": {
                    "width": LIVEVIEW_VIEWPORT_WIDTH,
                    "height": LIVEVIEW_VIEWPORT_HEIGHT,
                },
                "expires_at": session.expires_at,
            }

    async def _launch_locked(self, token) -> LiveViewSession:
        from nodriver import cdp
        import nodriver as uc

        token_id = int(token.id)
        profile_dir = liveview_profile_dir(token_id)
        profile_dir.mkdir(parents=True, exist_ok=True)
        session_id = uuid.uuid4().hex[:12]
        session = LiveViewSession(
            token_id=token_id,
            session_id=session_id,
            email=str(getattr(token, "email", "") or ""),
            started_at=time.time(),
            expires_at=time.time() + LIVEVIEW_SESSION_TTL_SECONDS,
        )

        # Proxy: same priority as the captcha browser, with a reachability
        # check + env fallback (dead frps in [proxy] must not break login).
        # Resolved through a throwaway personal-service instance on purpose:
        # the login browser must not depend on the captcha pool or on the
        # global captcha_method (production runs 'browser'). The instance
        # never launches a browser; its ephemeral profile dir (if one was
        # created) is removed right after resolution.
        personal_service = BrowserCaptchaService(self.db)
        try:
            protocol, host, port, username, password = (
                await personal_service.resolve_login_proxy()
            )
        finally:
            ephemeral_dir = str(
                getattr(personal_service, "_runtime_ephemeral_user_data_dir", "")
                or ""
            ).strip()
            if ephemeral_dir and os.path.isdir(ephemeral_dir):
                await asyncio.to_thread(shutil.rmtree, ephemeral_dir, True)
        proxy_server_arg = None
        if protocol and host and port:
            if username and password:
                session.proxy_ext_dir = _create_proxy_auth_extension(
                    protocol, host, port, username, password
                )
                debug_logger.log_info(
                    "[LiveView] Proxy de login com autenticação via extensão "
                    f"(token_id={token_id})"
                )
            else:
                proxy_server_arg = f"--proxy-server={protocol}://{host}:{port}"
            debug_logger.log_info(
                "[LiveView] Proxy do navegador de login: "
                f"{_redact_proxy_url(_compose_proxy_url(protocol, host, port, username, password))}"
            )

        browser_executable_path, _source = _resolve_browser_executable_path()
        sandbox_enabled = _resolve_personal_browser_sandbox_enabled()
        browser_args = _build_personal_browser_args(
            headless=True,
            proxy_server_arg=proxy_server_arg,
            proxy_extension_dir=session.proxy_ext_dir,
        )
        browser_args = _normalize_personal_browser_args_for_launch(
            browser_args,
            sandbox_enabled=sandbox_enabled,
        )

        try:
            browser = await asyncio.wait_for(
                uc.start(
                    headless=True,
                    user_data_dir=str(profile_dir),
                    browser_executable_path=browser_executable_path,
                    browser_args=browser_args,
                    sandbox=sandbox_enabled,
                ),
                timeout=LIVEVIEW_LAUNCH_TIMEOUT_SECONDS,
            )
        except Exception as e:
            raise LiveViewLaunchError(
                f"Falha ao iniciar o navegador de login: {type(e).__name__}: {e}"
            ) from e

        session.browser = browser
        try:
            tab = await self._open_login_tab(browser)
            session.tab = tab
            await asyncio.wait_for(
                tab.send(
                    cdp.emulation.set_device_metrics_override(
                        width=LIVEVIEW_VIEWPORT_WIDTH,
                        height=LIVEVIEW_VIEWPORT_HEIGHT,
                        device_scale_factor=1.0,
                        mobile=False,
                    )
                ),
                timeout=LIVEVIEW_COMMAND_TIMEOUT_SECONDS,
            )
        except Exception as e:
            await self._close_browser_quietly(session)
            raise LiveViewLaunchError(
                f"Falha ao abrir a página de login: {type(e).__name__}: {e}"
            ) from e

        # Best-effort initial state (a reused profile may already be logged in).
        try:
            await self._probe_login_locked(session)
        except Exception:
            pass
        debug_logger.log_info(
            f"[LiveView] Sessão iniciada (token_id={token_id}, "
            f"session_id={session_id}, profile={profile_dir})"
        )
        return session

    async def _open_login_tab(self, browser) -> Any:
        """Open the login page on a guaranteed page target.

        Never call browser.get() on a fresh browser: nodriver 0.50.8 does a
        bare next() over page targets, so when the first tab has not
        registered yet it raises StopIteration, which Python turns into
        RuntimeError('coroutine raised StopIteration'). Wait for a page
        target first; if none appears, create one explicitly (same pattern
        as the captcha service's _create_default_context_target_tab).
        """
        from nodriver import cdp

        tab = await _wait_for_page_target(browser, LIVEVIEW_PAGE_TARGET_WAIT_SECONDS)
        if tab is not None:
            try:
                await tab.attach()
            except Exception:
                pass
            await asyncio.wait_for(
                tab.get(LIVEVIEW_START_URL),
                timeout=LIVEVIEW_NAVIGATE_TIMEOUT_SECONDS,
            )
            return tab
        target_id = await asyncio.wait_for(
            browser.send(
                cdp.target.create_target(
                    LIVEVIEW_START_URL,
                    new_window=True,
                    enable_begin_frame_control=True,
                )
            ),
            timeout=LIVEVIEW_NAVIGATE_TIMEOUT_SECONDS,
        )
        deadline = time.monotonic() + LIVEVIEW_NAVIGATE_TIMEOUT_SECONDS
        while True:
            try:
                await browser.update_targets()
            except Exception:
                pass
            tab = next(
                (
                    item
                    for item in getattr(browser, "targets", []) or []
                    if getattr(item, "type_", None) == "page"
                    and getattr(item, "target_id", None) == target_id
                ),
                None,
            )
            if tab is not None:
                await tab.attach()
                return tab
            if time.monotonic() >= deadline:
                raise LiveViewLaunchError(
                    "Falha ao abrir a página de login: target não apareceu "
                    "após create_target"
                )
            await asyncio.sleep(0.15)

    async def _session_ttl_watchdog(self, session: LiveViewSession) -> None:
        try:
            delay = max(0.0, session.expires_at - time.time())
            await asyncio.sleep(delay)
            if not session.closed:
                debug_logger.log_info(
                    f"[LiveView] Sessão expirada por TTL (token_id={session.token_id}, "
                    f"session_id={session.session_id})"
                )
                await self.close_session(session, reason="ttl_expired")
        except asyncio.CancelledError:
            return
        except Exception as e:
            debug_logger.log_warning(f"[LiveView] watchdog da sessão falhou: {e}")

    async def close_session(
        self, session: LiveViewSession, *, reason: str = "closed"
    ) -> None:
        if session is None or session.closed:
            return
        session.closed = True
        if session.ttl_task is not None and not session.ttl_task.done():
            session.ttl_task.cancel()
        await self._close_browser_quietly(session)
        async with self._guard:
            if self._active is session:
                self._active = None
        debug_logger.log_info(
            f"[LiveView] Sessão encerrada (token_id={session.token_id}, "
            f"session_id={session.session_id}, reason={reason})"
        )

    async def cancel_session(self, token_id: int) -> bool:
        session = self.get_active_session(token_id)
        if session is None:
            return False
        await self.close_session(session, reason="cancelled")
        return True

    async def delete_profile(self, token_id: int) -> bool:
        token_id = int(token_id)
        if self.get_active_session(token_id) is not None:
            raise LiveViewBusyError(
                f"Não é possível apagar o perfil do token {token_id} "
                "enquanto há uma sessão de login ativa"
            )
        profile_dir = liveview_profile_dir(token_id)
        existed = profile_dir.exists()
        if existed:
            await asyncio.to_thread(shutil.rmtree, str(profile_dir), True)
            debug_logger.log_info(
                f"[LiveView] Perfil apagado (token_id={token_id}, path={profile_dir})"
            )
        return existed

    async def _close_browser_quietly(self, session: LiveViewSession) -> None:
        browser = session.browser
        session.browser = None
        session.tab = None
        if browser is not None:
            try:
                # Browser.stop() is sync: it schedules the async close.
                browser.stop()
            except Exception as e:
                debug_logger.log_warning(f"[LiveView] browser.stop falhou: {e}")
        if session.proxy_ext_dir:
            proxy_ext_dir = session.proxy_ext_dir
            session.proxy_ext_dir = None
            try:
                await asyncio.to_thread(shutil.rmtree, proxy_ext_dir, True)
            except Exception:
                pass

    # ---------- status / probe ----------

    async def get_status(self, token_id: int) -> Dict[str, Any]:
        session = self.get_active_session(token_id)
        if session is None:
            return {"active": False, "token_id": int(token_id)}
        try:
            await self._probe_login_locked(session)
        except Exception:
            pass
        return session.status_dict()

    async def _probe_login_locked(self, session: LiveViewSession) -> Dict[str, Any]:
        """Same identity probe as refresh_google_cookies (no navigation)."""
        tab = session.tab
        if tab is None:
            return {"logged_in": False, "email": "", "url": session.current_url}
        try:
            result = await asyncio.wait_for(
                tab.evaluate(
                    LIVEVIEW_LOGIN_IDENTITY_SCRIPT, return_by_value=True
                ),
                timeout=LIVEVIEW_COMMAND_TIMEOUT_SECONDS,
            )
        except Exception:
            return {"logged_in": False, "email": "", "url": session.current_url}
        if not isinstance(result, dict):
            result = {}
        host = str(result.get("host") or "")
        ready = str(result.get("ready") or "")
        email = str(result.get("email") or "").strip()
        current_url = str(getattr(tab, "url", "") or "").strip()
        if current_url:
            session.current_url = current_url
        logged_in = (
            host == "flow.google.com" and ready == "complete" and bool(email)
        )
        session.logged_in = logged_in
        session.observed_email = email if logged_in else ""
        return {
            "logged_in": logged_in,
            "email": session.observed_email,
            "url": session.current_url,
        }

    async def probe_login(self, session: LiveViewSession) -> Dict[str, Any]:
        async with session.lock:
            return await self._probe_login_locked(session)

    # ---------- frames / input ----------

    async def capture_frame(self, session: LiveViewSession) -> Optional[str]:
        from nodriver import cdp

        if session.closed or session.tab is None:
            return None
        try:
            async with session.lock:
                result = await asyncio.wait_for(
                    session.tab.send(
                        cdp.page.capture_screenshot(
                            format_="jpeg", quality=55, from_surface=True
                        )
                    ),
                    timeout=LIVEVIEW_COMMAND_TIMEOUT_SECONDS,
                )
        except Exception:
            return None
        if isinstance(result, dict):
            return str(result.get("data") or "") or None
        if isinstance(result, str) and result:
            return result
        return None

    async def send_mouse(
        self,
        session: LiveViewSession,
        *,
        event: str,
        x: float,
        y: float,
        button: str = "left",
        buttons: int = 1,
        click_count: int = 1,
    ) -> None:
        from nodriver import cdp

        if session.closed or session.tab is None:
            raise LiveViewNoSessionError("Sessão de login encerrada")
        type_map = {
            "move": "mouseMoved",
            "press": "mousePressed",
            "release": "mouseReleased",
        }
        type_ = type_map.get(str(event or "").strip().lower())
        if type_ is None:
            raise ValueError(f"Evento de mouse inválido: {event!r}")
        clamped_x = max(0.0, min(float(LIVEVIEW_VIEWPORT_WIDTH), float(x)))
        clamped_y = max(0.0, min(float(LIVEVIEW_VIEWPORT_HEIGHT), float(y)))
        button_value = str(button or "left").strip().lower()
        if button_value not in _MOUSE_BUTTONS:
            button_value = "left"
        async with session.lock:
            await asyncio.wait_for(
                session.tab.send(
                    cdp.input_.dispatch_mouse_event(
                        type_=type_,
                        x=clamped_x,
                        y=clamped_y,
                        # NOTE: nodriver's generated code calls
                        # button.to_json() — a plain str raises
                        # AttributeError and broke EVERY mouse event
                        # ("Falha de input: AttributeError"). It needs
                        # the MouseButton enum member, not the string.
                        button=cdp.input_.MouseButton(
                            button_value if type_ != "mouseMoved" else "none"
                        ),
                        buttons=int(buttons or 0),
                        click_count=int(click_count or 1),
                    )
                ),
                timeout=LIVEVIEW_COMMAND_TIMEOUT_SECONDS,
            )

    async def send_key(
        self,
        session: LiveViewSession,
        *,
        event: str,
        key: str = "",
        code: str = "",
        key_code: int = 0,
    ) -> None:
        from nodriver import cdp

        if session.closed or session.tab is None:
            raise LiveViewNoSessionError("Sessão de login encerrada")
        event_value = str(event or "").strip().lower()
        if event_value == "down":
            type_ = "rawKeyDown"
        elif event_value == "up":
            type_ = "keyUp"
        else:
            raise ValueError(f"Evento de teclado inválido: {event!r}")
        key_value = str(key or "")
        text = _KEY_TEXT_BY_KEY.get(key_value) if type_ == "rawKeyDown" else None
        async with session.lock:
            await asyncio.wait_for(
                session.tab.send(
                    cdp.input_.dispatch_key_event(
                        type_=type_,
                        key=key_value or None,
                        code=str(code or "") or None,
                        windows_virtual_key_code=int(key_code or 0) or None,
                        text=text,
                    )
                ),
                timeout=LIVEVIEW_COMMAND_TIMEOUT_SECONDS,
            )

    async def send_text(self, session: LiveViewSession, *, text: str) -> None:
        from nodriver import cdp

        if session.closed or session.tab is None:
            raise LiveViewNoSessionError("Sessão de login encerrada")
        text_value = str(text or "")
        if not text_value:
            return
        async with session.lock:
            await asyncio.wait_for(
                session.tab.send(cdp.input_.insert_text(text=text_value)),
                timeout=LIVEVIEW_COMMAND_TIMEOUT_SECONDS,
            )

    async def navigate(self, session: LiveViewSession, *, url: str) -> None:
        target_url = str(url or "").strip()
        if not target_url:
            raise ValueError("URL vazia")
        host = (urlparse(target_url).hostname or "").lower()
        allowed = host in _LIVEVIEW_NAVIGATE_ALLOWED_HOSTS or any(
            host.endswith(suffix)
            for suffix in _LIVEVIEW_NAVIGATE_ALLOWED_HOST_SUFFIXES
        )
        if urlparse(target_url).scheme != "https" or not allowed:
            raise ValueError(f"Navegação não permitida: {target_url}")
        if session.closed or session.tab is None:
            raise LiveViewNoSessionError("Sessão de login encerrada")
        async with session.lock:
            await asyncio.wait_for(
                session.tab.get(target_url),
                timeout=LIVEVIEW_NAVIGATE_TIMEOUT_SECONDS,
            )

    # ---------- harvest / commit ----------

    async def finish_session(self, token_id: int) -> Dict[str, Any]:
        """Harvest cookies from the login browser and commit them to the token.

        The session stays open if the user is not logged in yet or the commit
        fails, so the user can retry without logging in again.
        """
        session = self.get_active_session(token_id)
        if session is None:
            raise LiveViewNoSessionError(
                f"Nenhuma sessão de login ativa para o token {token_id}"
            )
        if self.token_manager is None:
            raise LiveViewUnavailableError("TokenManager não disponível")
        probe = await self.probe_login(session)
        if not probe.get("logged_in"):
            raise LiveViewNotLoggedInError(
                "Login ainda não concluído: faça login na conta Google na tela "
                "remota até chegar ao Flow e tente novamente"
            )
        result = await self._harvest_cookies_locked(session, probe)
        await self.token_manager.commit_refreshed_cookies(int(token_id), result)
        token = await self.db.get_token(int(token_id)) if self.db is not None else None
        at_expires = None
        if token is not None and getattr(token, "at_expires", None) is not None:
            at_expires = token.at_expires.isoformat()
        await self.close_session(session, reason="finished")
        return {
            "success": True,
            "email": probe.get("email") or "",
            "at_expires": at_expires,
        }

    async def _harvest_cookies_locked(
        self, session: LiveViewSession, probe: Dict[str, Any]
    ) -> Dict[str, Any]:
        from nodriver import cdp

        if session.browser is None:
            raise LiveViewLaunchError("Navegador de login indisponível")
        async with session.lock:
            cookies = await asyncio.wait_for(
                session.browser.send(cdp.storage.get_cookies()),
                timeout=LIVEVIEW_COMMAND_TIMEOUT_SECONDS,
            )
        serialized = []
        for cookie in cookies or []:
            item = BrowserCaptchaService._serialize_browser_cookie_for_storage(cookie)
            if not item:
                continue
            host = str(
                item.get("domain")
                or urlparse(item.get("url", "")).hostname
                or ""
            ).lstrip(".")
            if host == "google.com" or host.endswith(".google.com"):
                serialized.append(item)
        if not serialized:
            raise LiveViewLaunchError(
                "Nenhum cookie Google encontrado no navegador de login"
            )
        return {
            "google_cookies": json.dumps(
                serialized, ensure_ascii=False, separators=(",", ":")
            ),
            "email": probe.get("email") or "",
        }
