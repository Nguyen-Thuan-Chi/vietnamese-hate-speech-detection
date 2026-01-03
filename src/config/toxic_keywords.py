# src/config/toxic_keywords.py
"""
Shared toxic keyword vocabulary (Vietnamese).

USAGE:
- UI heuristic highlighting only
- NOT model features
- NOT span annotations
"""

TOXIC_KEYWORDS: list[str] = [

    # ------------------------------------------------------------------
    # CORE PROFANITY (short, high-frequency)
    # ------------------------------------------------------------------
    "đm", "đcm", "đmm", "đkm", "dkm", "dm", "dmm", "dcm",
    "vcl", "vkl", "vcc", "vch", "cl", "cc", "ml",
    "đéo", "éo", "éo le",
    "vãi", "vãi l", "vãi đ",

    # ------------------------------------------------------------------
    # SEXUAL / OBSCENE (core nouns & verbs)
    # ------------------------------------------------------------------
    "cặc", "buồi", "lồn", "đít",
    "ăn cặc", "bú cặc", "liếm cặc", "mút cặc",
    "đụ", "địt", "đụ mẹ", "địt mẹ",
    "đụ má", "địt má",

    # ------------------------------------------------------------------
    # FAMILY-BASED INSULTS (VERY common in VN hate speech)
    # ------------------------------------------------------------------
    "đĩ", "con đĩ",
    "đĩ mẹ", "đĩ má",
    "con đĩ mẹ", "con đĩ má",
    "đĩ mẹ mày", "đĩ má mày",
    "con đĩ mẹ mày", "con đĩ má mày",
    "đụ mẹ mày", "địt mẹ mày",
    "đụ má mày", "địt má mày",

    # ------------------------------------------------------------------
    # DIRECT INSULTS (single & compound)
    # ------------------------------------------------------------------
    "ngu", "ngốc", "đần", "dốt",
    "đồ ngu", "thằng ngu", "con ngu",
    "ngu vcl", "ngu vl", "ngu vãi",
    "óc chó", "não chó",
    "đầu buồi", "đầu lồn",

    # ------------------------------------------------------------------
    # ANIMAL-BASED INSULTS
    # ------------------------------------------------------------------
    "chó", "con chó", "thằng chó",
    "lợn", "con lợn",
    "súc vật",

    # ------------------------------------------------------------------
    # MORAL / CHARACTER ATTACKS
    # ------------------------------------------------------------------
    "khốn nạn", "mất dạy", "vô học",
    "bố láo", "láo chó",
    "bẩn thỉu", "đê tiện",
    "rác rưởi", "não tàn",
    "đầu đất", "não phẳng",
    "thiểu năng", "trí tuệ thấp",

    # ------------------------------------------------------------------
    # HATE / AGGRESSION MARKERS
    # ------------------------------------------------------------------
    "ghét", "ghét vcl",
    "chết đi", "đi chết đi",
    "cút đi", "cút xéo",
    "biến đi", "xéo đi",
    "đéo muốn nhìn mặt",

]
