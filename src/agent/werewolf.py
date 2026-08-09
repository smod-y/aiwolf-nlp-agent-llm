"""Module that defines the Werewolf agent class.

人狼のエージェントクラスを定義するモジュール.
"""

from __future__ import annotations

import random
from typing import Any

from aiwolf_nlp_common.packet import Role

from agent.agent import Agent

_MIN_PLAYERS_FOR_FAKE_CO = 9
"""偽占い CO を行う最小プレイヤー数."""

_DAY_FAKE_WHITE_PARTNER = 2
"""相方に偽白を出す日."""

_DAY_FAKE_BLACK_START = 3
"""偽黒を出し始める日."""


class Werewolf(Agent):
    """Werewolf agent class.

    人狼のエージェントクラス.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,  # noqa: ARG002
    ) -> None:
        """Initialize the werewolf agent.

        人狼のエージェントを初期化する.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role (ignored, always set to WEREWOLF) / 役職(無視され、常にWEREWOLFに設定)
        """
        super().__init__(config, name, game_id, Role.WEREWOLF)
        self.ww_seer_co: bool = False
        self.ww_want_co: bool = False
        self.ww_co_decided: bool = False
        self.fake_divine_results: dict[str, str] = {}
        self.ww_fake_today_target: str | None = None
        self.ww_fake_today_result: str | None = None

    def daily_initialize(self) -> None:
        """Perform processing for daily initialization request.

        昼開始リクエストに対する処理を行う.
        偽占い CO の判定と偽占い結果の生成を行った上で LLM へ送信する.
        """
        self._refresh_extractions()
        self._determine_ww_seer_co()
        if self.ww_seer_co:
            self._populate_fake_divine()
        self._send_message_to_llm(self.request)

    def _determine_ww_seer_co(self) -> None:
        """9人村以上の場合、whisper での合意に基づいて占い師COする人狼を決定する.

        Day 0: ww_want_co フラグを設定（名前順で先の人狼がCO希望）。
        Day 1+: Day 0 の whisper 履歴から相方がCOを主張したかを判定。
        相方がCO宣言していたら譲り、そうでなければ自分がCO。
        """
        player_num = self._player_num
        if player_num < _MIN_PLAYERS_FOR_FAKE_CO or self.info is None or not self.info.role_map:
            self.ww_seer_co = False
            self.ww_want_co = False
            return

        werewolf_names = sorted(
            name for name, role in self.info.role_map.items()
            if role == Role.WEREWOLF
        )
        is_first = bool(werewolf_names) and self.info.agent == werewolf_names[0]

        if self.info.day == 0:
            self.ww_want_co = is_first
            self.ww_seer_co = False
            return

        if self.ww_co_decided:
            return

        self.ww_co_decided = True

        partner_names = [n for n in werewolf_names if n != self.info.agent]
        partner_claims_co = self._partner_claimed_co(partner_names)

        if is_first:
            self.ww_seer_co = not partner_claims_co
        else:
            self.ww_seer_co = False

    def _partner_claimed_co(self, partner_names: list[str]) -> bool:
        """Day 0 の whisper 履歴から相方がCOを主張したかを判定する."""
        co_keywords = ["占い師CO", "自分がCO", "俺がCO", "私がCO", "僕がCO",
                       "あたしがCO", "わしがCO", "占い師をやる", "占い師やる",
                       "占い師として出る", "占い師に出る", "COする", "COします",
                       "COしたい", "COさせて"]
        for whisper in self.whisper_history:
            if whisper.agent in partner_names and whisper.day == 0:
                if any(kw in whisper.text for kw in co_keywords):
                    return True
        return False

    def _populate_fake_divine(self) -> None:
        """当日分の偽占い結果を生成する."""
        if self.info is None:
            return
        self.ww_fake_today_target = None
        self.ww_fake_today_result = None

        partners = [
            name for name, role in self.info.role_map.items()
            if role == Role.WEREWOLF and name != self.info.agent
        ]
        alive = self.get_alive_agents()

        if self.info.day == 1:
            # Day 1: white to random non-partner, non-self alive player
            candidates = [a for a in alive if a != self.info.agent and a not in partners]
            if candidates:
                target = random.choice(candidates)  # noqa: S311
                self.fake_divine_results[target] = self.L("seer_white")
                self.ww_fake_today_target = target
                self.ww_fake_today_result = self.L("color_white")

        elif self.info.day == _DAY_FAKE_WHITE_PARTNER:
            # Day 2: white to alive partner
            alive_partners = [p for p in partners if p in alive]
            if alive_partners:
                target = alive_partners[0]
            else:
                # Partner dead - white to random non-CO
                co_players = (
                    set(self.co_divine_map.keys())
                    | self.medium_co_set
                    | set(self.medium_result_map.keys())
                    | self.bodyguard_co_set
                )
                candidates = [
                    a for a in alive
                    if a != self.info.agent and a not in co_players
                    and a not in self.fake_divine_results
                ]
                target = random.choice(candidates) if candidates else None  # noqa: S311
            if target:
                self.fake_divine_results[target] = self.L("seer_white")
                self.ww_fake_today_target = target
                self.ww_fake_today_result = self.L("color_white")

        elif self.info.day >= _DAY_FAKE_BLACK_START:
            # Day 3+: black to someone without CO (unless max blacks reached)
            co_players = (
                set(self.co_divine_map.keys())
                | self.medium_co_set
                | set(self.medium_result_map.keys())
                | self.bodyguard_co_set
            )
            candidates = [
                a for a in alive
                if a != self.info.agent
                and a not in partners
                and a not in co_players
                and a not in self.fake_divine_results
            ]
            if candidates:
                existing_blacks = sum(
                    1 for r in self.fake_divine_results.values()
                    if r == self.L("seer_black")
                )
                target = random.choice(candidates)  # noqa: S311
                if existing_blacks < self._werewolf_total:
                    self.fake_divine_results[target] = self.L("seer_black")
                    self.ww_fake_today_target = target
                    self.ww_fake_today_result = self.L("color_black")
                else:
                    self.fake_divine_results[target] = self.L("seer_white")
                    self.ww_fake_today_target = target
                    self.ww_fake_today_result = self.L("color_white")

    def _get_template_keys(self) -> dict[str, Any]:
        """Get template keys including fake divine results.

        偽占い結果を含むテンプレートキーを取得する.

        Returns:
            dict[str, Any]: Template keys / テンプレートキー
        """
        keys = super()._get_template_keys()
        keys["ww_seer_co"] = self.ww_seer_co
        keys["ww_want_co"] = self.ww_want_co
        keys["ww_fake_divine_results"] = self.fake_divine_results
        keys["ww_fake_today_target"] = self.ww_fake_today_target
        keys["ww_fake_today_result"] = self.ww_fake_today_result
        return keys

    def whisper(self) -> str:
        """Return response to whisper request.

        囁きリクエストに対する応答を返す.

        Returns:
            str: Whisper message / 囁きメッセージ
        """
        return super().whisper()

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.

        Returns:
            str: Talk message / 発言メッセージ
        """
        return super().talk()

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        return super().vote()

    def attack(self) -> str:
        """Return response to attack request.

        襲撃リクエストに対する応答を返す.

        Returns:
            str: Agent name to attack / 襲撃対象のエージェント名
        """
        return super().attack()
