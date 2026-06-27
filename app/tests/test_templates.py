from app.kg.templates import render, reply_prefix


def test_render_single():
    cqls = render("定义", {"Disease": "高血压"})
    assert len(cqls) == 1
    assert "高血压" in cqls[0] and "RETURN p.desc" in cqls[0]


def test_render_list_intent():
    cqls = render("治疗方法", {"Disease": "感冒"})
    assert len(cqls) == 3 and all("感冒" in c for c in cqls)


def test_unknown_intent():
    assert render("不存在的意图", {"Disease": "x"}) == []


def test_reply_prefix():
    assert "高血压" in reply_prefix("定义", {"Disease": "高血压"})
