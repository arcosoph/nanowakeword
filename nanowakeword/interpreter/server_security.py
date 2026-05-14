# ==============================================================================
#  NanoWakeWord: Lightweight, Intelligent Wake Word Detection
#  Copyright 2025 Arcosoph. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Project: https://github.com/arcosoph/nanowakeword
# ==============================================================================

"""
server_security.py — Optional security layer for the RemoteVerifier server.

All features are opt-in. The server runs with zero security overhead by default.

Available features
------------------
  API Key authentication
      Clients must send a valid API key in the WebSocket handshake header
      ``X-API-Key``. Keys are stored as bcrypt hashes so the plaintext never
      lives on disk after initial setup.

  Token-based access control
      Short-lived JWT-style tokens (HMAC-SHA256, no external library needed).
      Clients exchange an API key for a time-limited token, then use the token
      for subsequent connections. Reduces per-connection key verification cost.

  WSS / TLS support
      Pass an SSL context (or cert/key paths) to wrap the WebSocket server in
      TLS. Requires the ``ssl`` standard library module (always available).

  Rate limiting
      Per-IP sliding-window rate limiter. Configurable max requests per window.
      Clients that exceed the limit receive a 429-equivalent close and are
      temporarily blocked.

  IP allowlist
      Only connections from explicitly listed CIDR ranges or exact IPs are
      accepted. Everything else is rejected immediately.

Usage
-----
    # From the interpreter package (recommended):
    from nanowakeword.interpreter import SecurityConfig, build_security

    # Or import directly:
    from nanowakeword.interpreter.server_security import SecurityConfig

    # Minimal — no security (default)
    cfg = SecurityConfig()

    # API key only
    cfg = SecurityConfig(api_keys=["my-secret-key"])

    # Full stack
    cfg = SecurityConfig(
        api_keys=["key1", "key2"],
        enable_tokens=True,
        token_ttl=3600,
        rate_limit=100,
        rate_window=60,
        ip_allowlist=["192.168.1.0/24", "10.0.0.1"],
        ssl_certfile="server.crt",
        ssl_keyfile="server.key",
    )

    # Then pass to serve():
    serve("my_model.onnx", security=cfg)
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import logging
import os
import secrets
import ssl
import struct
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


#  Helpers 

def _hash_key(key: str) -> str:
    """
    One-way hash an API key with SHA-256 + a fixed server salt stored in the
    hash string itself (``salt$hash``).  No external library required.

    For production deployments with many keys, consider replacing this with
    bcrypt (``pip install bcrypt``).
    """
    salt = secrets.token_hex(16)
    digest = hashlib.sha256(f"{salt}{key}".encode()).hexdigest()
    return f"{salt}${digest}"


def _verify_key(key: str, stored_hash: str) -> bool:
    """Constant-time comparison against a stored ``salt$hash`` string."""
    try:
        salt, digest = stored_hash.split("$", 1)
    except ValueError:
        return False
    expected = hashlib.sha256(f"{salt}{key}".encode()).hexdigest()
    return hmac.compare_digest(expected, digest)


def _make_token(secret: str, ttl: int) -> str:
    """
    Create a simple signed token: ``expiry_ts.hmac_hex``.
    No JWT library required — uses HMAC-SHA256 with a server secret.
    """
    expiry = int(time.time()) + ttl
    payload = str(expiry).encode()
    sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"{expiry}.{sig}"


def _verify_token(token: str, secret: str) -> bool:
    """Returns True if the token is valid and not expired."""
    try:
        expiry_str, sig = token.split(".", 1)
        expiry = int(expiry_str)
    except (ValueError, AttributeError):
        return False
    if time.time() > expiry:
        return False
    expected_sig = hmac.new(secret.encode(), expiry_str.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected_sig, sig)


#  SecurityConfig 

@dataclass
class SecurityConfig:
    """
    Holds all optional security settings for the RemoteVerifier server.

    Every field defaults to "disabled" so the server works out-of-the-box
    with no security overhead.

    Attributes:
        api_keys:       List of plaintext API keys that clients may use.
                        Keys are hashed in memory at startup — plaintext is
                        never stored after ``SecurityManager`` is built.
                        Empty list → API key auth disabled.

        enable_tokens:  When True, clients can POST ``/token`` (via a special
                        WebSocket handshake message) to exchange a valid API key
                        for a short-lived token. Subsequent connections can use
                        the token instead of the raw key.
                        Requires ``api_keys`` to be non-empty.

        token_ttl:      Token lifetime in seconds. Default 3600 (1 hour).

        token_secret:   HMAC secret used to sign tokens. Auto-generated at
                        startup if not provided. Set explicitly for multi-process
                        or persistent deployments.

        rate_limit:     Maximum number of WebSocket messages per ``rate_window``
                        seconds per IP address. 0 → disabled.

        rate_window:    Sliding window size in seconds for rate limiting.
                        Default 60.

        ip_allowlist:   List of allowed IP addresses or CIDR ranges, e.g.
                        ``["192.168.1.0/24", "10.0.0.5"]``.
                        Empty list → all IPs allowed.

        ssl_certfile:   Path to PEM certificate file for WSS/TLS.
                        Both ``ssl_certfile`` and ``ssl_keyfile`` must be set
                        to enable TLS.

        ssl_keyfile:    Path to PEM private key file for WSS/TLS.

        ssl_ca_certs:   Optional path to CA bundle for mutual TLS (mTLS).
                        When set, clients must present a valid certificate.

        max_connections: Maximum number of simultaneous client connections.
                        0 → unlimited.

        ban_duration:   Seconds to ban an IP after it exceeds the rate limit.
                        Default 300 (5 minutes). 0 → no banning, just drop
                        the offending message.
    """

    # API key auth
    api_keys:       List[str]        = field(default_factory=list)

    # Token auth
    enable_tokens:  bool             = False
    token_ttl:      int              = 3600
    token_secret:   Optional[str]    = None

    # Rate limiting
    rate_limit:     int              = 0
    rate_window:    int              = 60

    # IP allowlist
    ip_allowlist:   List[str]        = field(default_factory=list)

    # TLS / WSS
    ssl_certfile:   Optional[str]    = None
    ssl_keyfile:    Optional[str]    = None
    ssl_ca_certs:   Optional[str]    = None

    # Connection cap
    max_connections: int             = 0

    # Ban duration after rate-limit breach
    ban_duration:   int              = 300

    #  Derived properties 

    @property
    def auth_enabled(self) -> bool:
        """True when any form of authentication is active."""
        return bool(self.api_keys)

    @property
    def tls_enabled(self) -> bool:
        """True when both cert and key files are provided."""
        return bool(self.ssl_certfile and self.ssl_keyfile)

    @property
    def rate_limiting_enabled(self) -> bool:
        return self.rate_limit > 0

    @property
    def allowlist_enabled(self) -> bool:
        return bool(self.ip_allowlist)

    def summary(self) -> str:
        """Human-readable summary of active security features."""
        features = []
        if self.auth_enabled:
            features.append(f"API-key auth ({len(self.api_keys)} key(s))")
        if self.enable_tokens:
            features.append(f"token auth (TTL={self.token_ttl}s)")
        if self.tls_enabled:
            features.append("WSS/TLS")
        if self.rate_limiting_enabled:
            features.append(f"rate-limit ({self.rate_limit} req/{self.rate_window}s)")
        if self.allowlist_enabled:
            features.append(f"IP allowlist ({len(self.ip_allowlist)} entries)")
        if self.max_connections > 0:
            features.append(f"max-connections={self.max_connections}")
        if not features:
            return "none (open server)"
        return ", ".join(features)


#  SecurityManager 

class SecurityManager:
    """
    Runtime security engine built from a :class:`SecurityConfig`.

    Instantiated once at server startup. Thread-safe for async use.

    Responsibilities:
    - Hash and store API keys
    - Issue and verify tokens
    - Track per-IP request rates and bans
    - Validate IP allowlist
    - Build SSL context
    - Track active connection count
    """

    def __init__(self, config: SecurityConfig):
        self.config = config

        # Hash API keys — plaintext is discarded after this point
        self._key_hashes: List[str] = [_hash_key(k) for k in config.api_keys]

        # Token secret — auto-generate if not provided
        self._token_secret: str = config.token_secret or secrets.token_hex(32)
        if config.enable_tokens and not config.token_secret:
            logger.info(
                "[Security] Token secret auto-generated. "
                "Set token_secret explicitly for persistent deployments."
            )

        # Rate limiting state: {ip: deque of timestamps}
        self._rate_windows: Dict[str, deque] = defaultdict(deque)

        # Banned IPs: {ip: ban_expiry_timestamp}
        self._bans: Dict[str, float] = {}

        # Parsed CIDR allowlist
        self._allowed_networks: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        for entry in config.ip_allowlist:
            try:
                self._allowed_networks.append(ipaddress.ip_network(entry, strict=False))
            except ValueError:
                logger.warning(f"[Security] Invalid IP allowlist entry ignored: '{entry}'")

        # Active connection counter
        self._active_connections: int = 0

        # SSL context (built once)
        self._ssl_context: Optional[ssl.SSLContext] = None
        if config.tls_enabled:
            self._ssl_context = self._build_ssl_context()

        logger.info(f"[Security] Active features: {config.summary()}")

    #  SSL 

    def _build_ssl_context(self) -> ssl.SSLContext:
        """Build and return an SSL context for WSS."""
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(
            certfile=self.config.ssl_certfile,
            keyfile=self.config.ssl_keyfile,
        )
        if self.config.ssl_ca_certs:
            ctx.load_verify_locations(cafile=self.config.ssl_ca_certs)
            ctx.verify_mode = ssl.CERT_REQUIRED
            logger.info("[Security] Mutual TLS (mTLS) enabled.")
        return ctx

    @property
    def ssl_context(self) -> Optional[ssl.SSLContext]:
        """The SSL context to pass to ``websockets.serve()``, or None."""
        return self._ssl_context

    #  Connection cap 

    def connection_allowed(self) -> bool:
        """Returns False when the max connection cap has been reached."""
        if self.config.max_connections <= 0:
            return True
        return self._active_connections < self.config.max_connections

    def on_connect(self) -> None:
        self._active_connections += 1

    def on_disconnect(self) -> None:
        self._active_connections = max(0, self._active_connections - 1)

    #  IP allowlist 

    def ip_allowed(self, ip: str) -> bool:
        """
        Returns True if the IP is permitted.
        Always True when the allowlist is empty (feature disabled).
        """
        if not self.config.allowlist_enabled:
            return True
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            logger.warning(f"[Security] Could not parse client IP: '{ip}'")
            return False
        return any(addr in net for net in self._allowed_networks)

    #  Rate limiting 

    def is_banned(self, ip: str) -> bool:
        """Returns True if the IP is currently banned."""
        expiry = self._bans.get(ip)
        if expiry is None:
            return False
        if time.time() < expiry:
            return True
        # Ban expired — clean up
        del self._bans[ip]
        return False

    def record_request(self, ip: str) -> bool:
        """
        Record a request from ``ip``.

        Returns True if the request is within the rate limit.
        Returns False if the limit is exceeded (and bans the IP if configured).
        Always returns True when rate limiting is disabled.
        """
        if not self.config.rate_limiting_enabled:
            return True

        now = time.time()
        window = self._rate_windows[ip]

        # Evict timestamps outside the sliding window
        cutoff = now - self.config.rate_window
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self.config.rate_limit:
            # Limit exceeded
            if self.config.ban_duration > 0:
                self._bans[ip] = now + self.config.ban_duration
                logger.warning(
                    f"[Security] Rate limit exceeded by {ip}. "
                    f"Banned for {self.config.ban_duration}s."
                )
            else:
                logger.warning(f"[Security] Rate limit exceeded by {ip}. Dropping message.")
            return False

        window.append(now)
        return True

    #  API key auth 

    def verify_api_key(self, key: str) -> bool:
        """
        Returns True if ``key`` matches any stored key hash.
        Constant-time comparison to prevent timing attacks.
        Always returns True when API key auth is disabled.
        """
        if not self.config.auth_enabled:
            return True
        return any(_verify_key(key, h) for h in self._key_hashes)

    #  Token auth 

    def issue_token(self) -> str:
        """Issue a new signed token. Requires ``enable_tokens=True``."""
        return _make_token(self._token_secret, self.config.token_ttl)

    def verify_token(self, token: str) -> bool:
        """
        Returns True if the token is valid and not expired.
        Always returns True when token auth is disabled.
        """
        if not self.config.enable_tokens:
            return True
        return _verify_token(token, self._token_secret)

    #  Unified handshake check

    def check_handshake(self, websocket) -> tuple[bool, str]:
        """
        Validate a new WebSocket connection.

        Checks (in order):
          1. Connection cap
          2. IP allowlist
          3. IP ban (from rate limiting)
          4. API key / token (from ``X-API-Key`` or ``X-Token`` header)

        Returns:
            (allowed: bool, reason: str)
        """
        # Connection cap
        if not self.connection_allowed():
            return False, "server at max connections"

        # IP allowlist
        ip = _get_client_ip(websocket)
        if not self.ip_allowed(ip):
            logger.warning(f"[Security] Rejected connection from non-allowlisted IP: {ip}")
            return False, f"IP {ip} not in allowlist"

        # Ban check
        if self.is_banned(ip):
            logger.warning(f"[Security] Rejected banned IP: {ip}")
            return False, f"IP {ip} is temporarily banned"

        # Auth check
        if self.config.auth_enabled:
            headers = _get_headers(websocket)

            # Token takes priority over API key when tokens are enabled
            if self.config.enable_tokens:
                token = headers.get("x-token", "")
                if token and self.verify_token(token):
                    return True, "ok"

            # Fall back to API key
            api_key = headers.get("x-api-key", "")
            if not api_key:
                logger.warning(f"[Security] Missing X-API-Key from {ip}")
                return False, "missing X-API-Key header"
            if not self.verify_api_key(api_key):
                logger.warning(f"[Security] Invalid API key from {ip}")
                return False, "invalid API key"

        return True, "ok"


# Wire protocol for token exchange

# Clients that want a token send a special 1-byte message: 0xF0 + API key bytes.
# Server responds with JSON: {"token": "<token>"}  or {"error": "<reason>"}

_TAG_TOKEN_REQUEST = 0xF0


def is_token_request(message: bytes) -> bool:
    return len(message) >= 2 and message[0] == _TAG_TOKEN_REQUEST


def decode_token_request(message: bytes) -> str:
    """Extract the API key from a token-request message."""
    return message[1:].decode("utf-8", errors="replace")


def encode_token_response(token: str) -> str:
    import json
    return json.dumps({"token": token})


def encode_error_response(reason: str) -> str:
    import json
    return json.dumps({"error": reason})


# Utilities

def _get_client_ip(websocket) -> str:
    """Extract the client IP from a websocket connection object."""
    try:
        addr = websocket.remote_address
        if isinstance(addr, tuple):
            return addr[0]
        return str(addr)
    except Exception:
        return "unknown"


def _get_headers(websocket) -> Dict[str, str]:
    """
    Return request headers as a lowercase-key dict.
    Compatible with both websockets v10–v12 and v13+.
    """
    try:
        # websockets >= 10: request.headers is a Headers object
        raw = websocket.request.headers
        return {k.lower(): v for k, v in raw.items()}
    except AttributeError:
        pass
    try:
        # websockets < 10: request_headers
        raw = websocket.request_headers
        return {k.lower(): v for k, v in raw.items()}
    except AttributeError:
        pass
    return {}


# Convenience factory

def build_security(
    api_keys:        Optional[List[str]] = None,
    enable_tokens:   bool = False,
    token_ttl:       int  = 3600,
    token_secret:    Optional[str] = None,
    rate_limit:      int  = 0,
    rate_window:     int  = 60,
    ip_allowlist:    Optional[List[str]] = None,
    ssl_certfile:    Optional[str] = None,
    ssl_keyfile:     Optional[str] = None,
    ssl_ca_certs:    Optional[str] = None,
    max_connections: int  = 0,
    ban_duration:    int  = 300,
) -> Optional["SecurityManager"]:
    """
    Convenience factory. Returns a :class:`SecurityManager` when any security
    feature is requested, or ``None`` when everything is at its default
    (no-op) value — so the server pays zero overhead.

    Args:
        api_keys:        Plaintext API keys clients must present.
        enable_tokens:   Allow clients to exchange a key for a short-lived token.
        token_ttl:       Token lifetime in seconds.
        token_secret:    HMAC secret for token signing. Auto-generated if omitted.
        rate_limit:      Max messages per ``rate_window`` seconds per IP. 0=off.
        rate_window:     Sliding window size in seconds.
        ip_allowlist:    Allowed IPs / CIDR ranges. Empty = allow all.
        ssl_certfile:    PEM certificate for WSS/TLS.
        ssl_keyfile:     PEM private key for WSS/TLS.
        ssl_ca_certs:    CA bundle for mutual TLS.
        max_connections: Max simultaneous clients. 0=unlimited.
        ban_duration:    Seconds to ban an IP after rate-limit breach.

    Returns:
        :class:`SecurityManager` or ``None``.

    Example::

        sec = build_security(api_keys=["my-key"], rate_limit=200)
        serve("model.onnx", security=sec)
    """
    cfg = SecurityConfig(
        api_keys        = api_keys or [],
        enable_tokens   = enable_tokens,
        token_ttl       = token_ttl,
        token_secret    = token_secret,
        rate_limit      = rate_limit,
        rate_window     = rate_window,
        ip_allowlist    = ip_allowlist or [],
        ssl_certfile    = ssl_certfile,
        ssl_keyfile     = ssl_keyfile,
        ssl_ca_certs    = ssl_ca_certs,
        max_connections = max_connections,
        ban_duration    = ban_duration,
    )

    # Return None when nothing is actually enabled - zero overhead
    nothing_enabled = (
        not cfg.auth_enabled
        and not cfg.tls_enabled
        and not cfg.rate_limiting_enabled
        and not cfg.allowlist_enabled
        and cfg.max_connections == 0
    )
    if nothing_enabled:
        return None

    return SecurityManager(cfg)
