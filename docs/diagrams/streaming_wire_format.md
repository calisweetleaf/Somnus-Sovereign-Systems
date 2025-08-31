# Streaming Wire Format Deep Dive

This document specifies the wire protocol for research streaming: envelope schemas, authentication, compression negotiation, and example frames.

## 1. Authentication

- Clients may include Authorization: Bearer <JWT> in the initial WebSocket upgrade request or as the first message after upgrade.
- JWT claims (HS256 by default):
  - sub: user identifier
  - session_id: active research session id
  - iat, exp: issued-at and expiry timestamps
  - jti: unique token id for replay detection
  - scopes: ["research:stream", ...]
- On success, server replies with subscription_ack. On failure, error(auth_failed) then close 4401.

Example claims:
```json
{
  "sub": "user_123",
  "session_id": "sess_abc",
  "iat": 1725100000,
  "exp": 1725101800,
  "jti": "4f6c2a2b-6cbb-4c1f-9fde-0a295f",
  "scopes": ["research:stream"]
}
```

## 2. Compression Negotiation

- Client announces support via an http header or initial message:
  - Header: Accept-Compression: zlib
  - Or message: {"type":"negotiate","compression":["zlib"]}
- Server acknowledges with compression_ack:
```json
{"type":"compression_ack","zlib":true}
```
- When enabled, payload is compressed with zlib, then base64-encoded as a string and placed in the payload field with compressed: true.

## 3. Envelope Schema

Top-level event envelope:
```json
{
  "type": "progress|contradiction|system|research_started|research_completed|flow_control|connection_status|ack|error",
  "session_id": "string",
  "correlation_id": "string",
  "timestamp": "ISO-8601",
  "compressed": false,
  "payload": {}
}
```

JSON Schema (simplified):
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["type","session_id","timestamp","payload"],
  "properties": {
    "type": {"type":"string"},
    "session_id": {"type":"string"},
    "correlation_id": {"type":"string"},
    "timestamp": {"type":"string","format":"date-time"},
    "compressed": {"type":"boolean"},
    "payload": {}
  }
}
```

## 4. Research Event Payloads

Research payload (expanded):
```json
{
  "query": "string",
  "sources": ["url or descriptor"],
  "status": "planning|collecting|analyzing|synthesizing|completed|failed",
  "result_ref": {"report_id":"...","uri":"..."},
  "metadata": {"depth":"comprehensive","quality":0.83}
}
```

Client-initiated event (feedback):
```json
{
  "kind": "pause|resume|focus|skip|clarify",
  "data": {"notes":"Focus on arXiv sources"}
}
```

## 5. Example Frames

### 5.1 Subscription Ack (uncompressed)
```json
{
  "type": "subscription_ack",
  "session_id": "sess_abc",
  "correlation_id": "corr_001",
  "timestamp": "2025-08-31T19:59:00Z",
  "compressed": false,
  "payload": {
    "rate_limit": {"rpm": 120},
    "heartbeat": {"mode": "event", "interval_ms": 15000},
    "filters": ["progress","contradiction","system"]
  }
}
```

### 5.2 Progress Event (uncompressed)
```json
{
  "type": "progress",
  "session_id": "sess_abc",
  "correlation_id": "corr_002",
  "timestamp": "2025-08-31T20:00:05Z",
  "compressed": false,
  "payload": {
    "query": "quantum error correction 2024",
    "sources": ["https://arxiv.org/abs/2309.12345"],
    "status": "collecting",
    "metadata": {"visited": 12, "queued": 4}
  }
}
```

### 5.3 Progress Event (compressed-zlib+base64)
- payload is the base64 string of compressed JSON; compressed: true.
```json
{
  "type": "progress",
  "session_id": "sess_abc",
  "correlation_id": "corr_003",
  "timestamp": "2025-08-31T20:00:08Z",
  "compressed": true,
  "payload": "eJyrVkrLz1eyUlBISsxLVchJLElVslJK1UvOz1NIyy9RyM/JLEkEAB2pCWI="
}
```

### 5.4 Flow Control Signal
```json
{
  "type": "flow_control",
  "session_id": "sess_abc",
  "timestamp": "2025-08-31T20:00:10Z",
  "compressed": false,
  "payload": {"action":"slow_down","reason":"server_backpressure"}
}
```

### 5.5 Connection Status Heartbeat
```json
{
  "type": "connection_status",
  "session_id": "sess_abc",
  "timestamp": "2025-08-31T20:00:15Z",
  "compressed": false,
  "payload": {"healthy": true, "latency_ms": 32}
}
```

## 6. Error Handling and Codes

- 4401 close: authentication required/failed
- 1000 normal closure
- error envelope example:
```json
{
  "type": "error",
  "session_id": "sess_abc",
  "timestamp": "2025-08-31T20:00:12Z",
  "compressed": false,
  "payload": {"code":"auth_failed","message":"Invalid JWT"}
}
```

## 7. Rate Limiting and Acknowledgements

- Server may send flow_control events; clients should reduce send rate or request fewer event types.
- Client events receive ack envelopes with the client_event_id.

## 8. Security Notes

- Secrets never logged; JWT only read via environment-backed secret handle.
- Compression is opt-in and negotiated; do not enable by default without client consent.
- All services bind to localhost by default to preserve local-first operation.

