# -*- coding: utf-8 -*-
import re

_DISCLAIMER = "\n\n(以上内容仅供参考,不能替代专业医生的诊断与治疗,如有不适请及时就医。)"
_DIAG_PAT = re.compile(r"(我.*(得了|患了|是什么病))|(确诊)|(开.*药)|(吃.*剂量)|(我的症状)")
_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_IDCARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")


def apply(answer, user_text):
    out = answer
    if _DIAG_PAT.search(user_text or ""):
        out = "请注意:我无法替代医生做出诊断,建议您及时就医面诊。\n" + out
    out = _PHONE.sub("***********", out)
    out = _IDCARD.sub("******************", out)
    return out + _DISCLAIMER
