import os
import pickle
from nlu.sklearn_Classification.train_modern import train_and_save

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "intent_recog_data.txt")
OUT = os.path.join(HERE, "..", "model_file")


def test_train_and_save(tmp_path):
    out = str(tmp_path)
    res = train_and_save(DATA, out)
    assert res["acc"] >= 0.5
    for fn in ["vec.pkl", "LR.pkl", "gbdt.pkl", "id2label.pkl"]:
        assert os.path.exists(os.path.join(out, fn))
    id2label = pickle.load(open(os.path.join(out, "id2label.pkl"), "rb"))
    assert "greet" in id2label.values()
