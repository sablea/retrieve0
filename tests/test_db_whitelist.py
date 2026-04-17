"""Unit tests for SQL read-only whitelist."""
import pytest

from re0.db.base import ReadOnlyViolation, validate_read_only


def test_accept_simple_select():
    assert validate_read_only("SELECT 1").lower().startswith("select")


def test_accept_select_with_comment_and_trailing_semicolon():
    sql = "/* hi */ SELECT id FROM t -- comment\n;"
    assert "select id from t" in validate_read_only(sql).lower()


def test_accept_show_desc_explain_with():
    for s in ["SHOW TABLES", "DESC t", "DESCRIBE t", "EXPLAIN SELECT 1", "WITH a AS (SELECT 1) SELECT * FROM a"]:
        validate_read_only(s)


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO t VALUES(1)",
        "UPDATE t SET a=1",
        "DELETE FROM t",
        "DROP TABLE t",
        "TRUNCATE t",
        "ALTER TABLE t ADD c INT",
        "CREATE TABLE t(a int)",
        "GRANT ALL ON *.* TO u",
        "SELECT 1; DROP TABLE t",
        "",
        "   ",
        "SELECT 1 /* ok */; SELECT 2",
    ],
)
def test_reject_writes_and_multi_stmt(sql):
    with pytest.raises(ReadOnlyViolation):
        validate_read_only(sql)


def test_accept_select_with_dml_keyword_in_literal():
    # 'drop' in a string literal must not trigger rejection.
    cleaned = validate_read_only("SELECT * FROM t WHERE note = 'drop'")
    assert cleaned.lower().startswith("select")
