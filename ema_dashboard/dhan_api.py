"""Dhan Trading + Data API helpers for live data and trade views."""

from __future__ import annotations

from typing import Any

import requests

BASE_URL = "https://api.dhan.co/v2"


def _headers(access_token: str, client_id: str | None = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "access-token": access_token}
    if client_id:
        headers["client-id"] = str(client_id)
    return headers


def _parse_response(resp: requests.Response) -> tuple[Any | None, str | None]:
    if resp.status_code != 200:
        return None, f"API Error {resp.status_code}: {resp.text}"
    try:
        data = resp.json()
    except Exception as exc:  # pragma: no cover - defensive for non-JSON errors
        return None, f"Invalid JSON response: {exc}"

    if isinstance(data, dict):
        if data.get("status") == "failure":
            message = data.get("errorMessage") or data.get("message") or data
            return None, f"API Error: {message}"
        if any(k in data for k in ["errorCode", "internalErrorMessage", "internalErrorCode"]):
            message = data.get("internalErrorMessage") or data.get("errorCode") or data
            return None, f"API Error: {message}"
    return data, None


def fetch_option_expiries(
    access_token: str,
    client_id: str,
    underlying_scrip: int,
    underlying_seg: str,
) -> tuple[dict | None, str | None]:
    payload = {
        "UnderlyingScrip": int(underlying_scrip),
        "UnderlyingSeg": underlying_seg,
    }
    resp = requests.post(
        f"{BASE_URL}/optionchain/expirylist",
        headers=_headers(access_token, client_id),
        json=payload,
        timeout=20,
    )
    return _parse_response(resp)


def fetch_option_chain(
    access_token: str,
    client_id: str,
    underlying_scrip: int,
    underlying_seg: str,
    expiry: str,
) -> tuple[dict | None, str | None]:
    payload = {
        "UnderlyingScrip": int(underlying_scrip),
        "UnderlyingSeg": underlying_seg,
        "Expiry": expiry,
    }
    resp = requests.post(
        f"{BASE_URL}/optionchain",
        headers=_headers(access_token, client_id),
        json=payload,
        timeout=20,
    )
    return _parse_response(resp)


def fetch_order_book(access_token: str) -> tuple[list | dict | None, str | None]:
    resp = requests.get(
        f"{BASE_URL}/orders",
        headers=_headers(access_token),
        timeout=20,
    )
    return _parse_response(resp)


def fetch_trade_book(access_token: str) -> tuple[list | dict | None, str | None]:
    resp = requests.get(
        f"{BASE_URL}/trades",
        headers=_headers(access_token),
        timeout=20,
    )
    return _parse_response(resp)


def fetch_trade_history(
    access_token: str,
    from_date: str,
    to_date: str,
    page: int = 0,
) -> tuple[list | dict | None, str | None]:
    resp = requests.get(
        f"{BASE_URL}/trades/{from_date}/{to_date}/{page}",
        headers=_headers(access_token),
        timeout=20,
    )
    return _parse_response(resp)


def fetch_trade_history_all(
    access_token: str,
    from_date: str,
    to_date: str,
    max_pages: int = 50,
) -> tuple[list | None, str | None]:
    all_rows: list = []
    for page in range(max_pages):
        data, error = fetch_trade_history(access_token, from_date, to_date, page=page)
        if error:
            return None, error
        if isinstance(data, dict) and "data" in data:
            data = data.get("data", [])
        if not data:
            break
        if isinstance(data, list):
            all_rows.extend(data)
        else:
            break
    return all_rows, None
