"""Bilingual label definitions for agent internal strings.

config の language 設定 (ja / en) に応じてラベルを切り替える.
"""

from __future__ import annotations

from typing import Any

_LABELS: dict[str, dict[str, str]] = {
    # --- 占い/霊能の判定ラベル (co_divine_map / medium_result_map の値) ---
    "white": {"ja": "白(人間)", "en": "Innocent"},
    "black": {"ja": "黒(人狼)", "en": "Werewolf"},
    # --- 表示用マーカー ---
    "dead": {"ja": "(死亡)", "en": "(dead)"},
    "gray": {"ja": " ← グレー", "en": " <- gray"},
    "none": {"ja": "(なし)", "en": "(none)"},
    "none_yet": {"ja": "(まだなし)", "en": "(none yet)"},
    # --- 占い結果 color (テンプレートキー my_judgments 用) ---
    "color_white": {"ja": "白", "en": "innocent"},
    "color_black": {"ja": "黒", "en": "werewolf"},
    # --- 役職表示ラベル (player_intel_lines 用) ---
    "role_seer": {"ja": "占い師", "en": "Seer"},
    "role_seer_attacked": {"ja": "占い師（先に襲撃=真濃厚）", "en": "Seer (attacked=likely real)"},
    "role_seer_likely_fake": {"ja": "占い師（対抗が先に襲撃された=偽濃厚）", "en": "Seer (counter attacked first=likely fake)"},
    "role_possessed_fake_seer": {"ja": "狂人（偽占いCO）", "en": "Possessed (fake Seer)"},
    "role_medium": {"ja": "霊能者", "en": "Medium"},
    "role_medium_confirmed": {"ja": "霊能者（確定=白扱い）", "en": "Medium (confirmed villager)"},
    "role_bodyguard": {"ja": "騎士", "en": "Bodyguard"},
    "role_bodyguard_confirmed": {"ja": "騎士（確定=白扱い）", "en": "Bodyguard (confirmed villager)"},
    "role_bodyguard_likely": {"ja": "騎士（対抗無し・濃厚）", "en": "Bodyguard (no counter, likely)"},
    # --- intel 行フォーマット ---
    "intel_row": {
        "ja": "- {player}{marker}: 役職=[{role_str}], 占い受領=[{divine_str}]{gray_marker}",
        "en": "- {player}{marker}: role=[{role_str}], divination=[{divine_str}]{gray_marker}",
    },
    "intel_divine_item": {"ja": "{seer}より{color}", "en": "{color} by {seer}"},
    "intel_separator": {"ja": "、", "en": ", "},
    # --- 囲い候補ライン ---
    "kakoi_ww": {
        "ja": "- {seer}{sm} が「{target}{tm}」を白判定 → 囲い候補（黒塗り誘導の標的）",
        "en": "- {seer}{sm} gave {target}{tm} innocent -> sheltering candidate (framing target)",
    },
    "kakoi_village": {
        "ja": "- {seer}{sm} が「{target}{tm}」を白判定 → 囲い候補（人狼を匿うための白の可能性）",
        "en": "- {seer}{sm} gave {target}{tm} innocent -> sheltering candidate (possibly protecting a Werewolf)",
    },
    # --- 整合的支持 ---
    "support_gave_innocent": {
        "ja": "白出しした相手を擁護（自然）",
        "en": "supports the player they gave innocent (natural)",
    },
    "support_received_innocent": {
        "ja": "白判定してくれた相手に擁護を返す（自然）",
        "en": "supports the player who gave them innocent (natural)",
    },
    # --- seer.py 用 ---
    "seer_white": {"ja": "白(人間)", "en": "Innocent (Human)"},
    "seer_black": {"ja": "黒(人狼)", "en": "Werewolf"},
}


class Labels:
    """Language-aware label accessor."""

    def __init__(self, lang: str = "ja") -> None:
        self._lang = lang if lang in ("ja", "en") else "ja"

    def __call__(self, key: str) -> str:
        entry = _LABELS.get(key)
        if entry is None:
            return key
        return entry.get(self._lang, entry.get("ja", key))

    @property
    def lang(self) -> str:
        return self._lang


def get_labels(config: dict[str, Any]) -> Labels:
    """Create Labels instance from config dict."""
    lang = str(config.get("language", "ja"))
    return Labels(lang)
