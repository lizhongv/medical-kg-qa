import json
from app.nlu.normalize import Normalizer


def test_exact(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps(["高血压", "感冒"], ensure_ascii=False), encoding="utf-8")
    n = Normalizer(str(p))
    assert n.normalize("高血压") == "高血压"
    assert n.normalize("不存在的病") is None
