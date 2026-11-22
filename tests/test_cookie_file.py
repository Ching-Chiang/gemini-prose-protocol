"""`cookies_file` accepts both shapes a browser exports cookies in."""

from app.services.client import cookie_pairs

PLAYWRIGHT_DUMP = [
    {"domain": ".google.com", "name": "__Secure-1PSID", "path": "/", "value": "a"},
    {"domain": ".google.com", "name": "OSID", "path": "/", "value": "b"},
]


def test_a_browser_dump_is_read_by_name_and_value():
    assert cookie_pairs(PLAYWRIGHT_DUMP) == {"__Secure-1PSID": "a", "OSID": "b"}


def test_a_plain_jar_of_name_to_value_is_read_too():
    assert cookie_pairs({"__Secure-1PSID": "a"}) == {"__Secure-1PSID": "a"}


def test_an_entry_google_leaves_empty_is_kept():
    assert cookie_pairs([{"name": "NID", "value": ""}]) == {"NID": ""}


def test_entries_that_cannot_be_read_are_left_out():
    jar = [{"name": "missing-value"}, {"value": "no-name"}, "junk", {"name": "ok", "value": 7}]

    assert cookie_pairs(jar) == {"ok": "7"}


def test_anything_else_yields_nothing():
    assert cookie_pairs("nope") == {}
    assert cookie_pairs(123) == {}
