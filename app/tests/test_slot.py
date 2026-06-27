import json
from app.nlu.slot import SlotFiller


def test_extract_from_dict(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps(["高血压", "高血压病", "感冒"], ensure_ascii=False), encoding="utf-8")
    sf = SlotFiller(str(p))
    got = sf.extract("高血压病怎么治")
    assert "高血压病" in got        # 命中最长
    assert sf.extract("我感冒了") == ["感冒"]


def test_no_hit(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps(["感冒"], ensure_ascii=False), encoding="utf-8")
    assert SlotFiller(str(p)).extract("你好呀") == []
