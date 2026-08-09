"""Module that defines the Possessed agent class.

狂人のエージェントクラスを定義するモジュール.
"""

from __future__ import annotations

import random
from typing import Any

from aiwolf_nlp_common.packet import Role

from agent.agent import Agent, _is_black, _is_white


class Possessed(Agent):
    """Possessed agent class.

    狂人のエージェントクラス.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,  # noqa: ARG002
    ) -> None:
        """Initialize the possessed agent.

        狂人のエージェントを初期化する.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role (ignored, always set to POSSESSED) / 役職(無視され、常にPOSSESSEDに設定)
        """
        super().__init__(config, name, game_id, Role.POSSESSED)
        self.possessed_black_target: str | None = None
        self.possessed_exposed: bool = False
        self.possessed_black_history: list[str] = []

    def daily_initialize(self) -> None:
        """Perform processing for daily initialization request.

        昼開始リクエストに対する処理を行う.
        Day 1 以降で黒出しターゲットを計算してからテンプレートキーに反映する.
        """
        self._refresh_extractions()
        self._check_exposed()
        if self.info and self.info.day >= 1 and not self.possessed_exposed:
            self._calculate_black_target()
        self._send_message_to_llm(self.request)

    def _check_exposed(self) -> None:
        """霊能結果と自分の占い結果を突き合わせて偽バレを検知する."""
        if self.possessed_exposed or self.info is None:
            return
        my_results = self.co_divine_map.get(self.info.agent, {})
        if not my_results:
            return
        for medium_results in self.medium_result_map.values():
            for target, medium_judgment in medium_results.items():
                my_judgment = my_results.get(target)
                if my_judgment is None:
                    continue
                my_black = _is_black(my_judgment)
                my_white = _is_white(my_judgment)
                medium_white = _is_white(medium_judgment)
                medium_black = _is_black(medium_judgment)
                if (my_black and medium_white) or (my_white and medium_black):
                    self.possessed_exposed = True
                    return

    def _calculate_black_target(self) -> None:
        """黒出しターゲットを決定する.

        優先順位:
        1. 自分以外の占い師CO者がいればその者に黒出し
        2. 霊能者CO者・騎士CO者・自分が白出し済みの対象は除外
        3. 残りのCO無し生存者からランダム

        ただし、総黒出し数がゲームの人狼数を超える場合は黒出ししない.
        """
        if self.info is None:
            return
        self.possessed_black_target = None

        my_results = self.co_divine_map.get(self.info.agent, {})
        co_map_blacks = sum(1 for r in my_results.values() if _is_black(r))
        total_blacks = max(len(self.possessed_black_history), co_map_blacks)
        if total_blacks >= self._werewolf_total:
            return

        all_mediums = self.medium_co_set | set(self.medium_result_map.keys())

        my_white_targets = {
            target
            for target, judgment in my_results.items()
            if _is_white(judgment)
        }

        alive = self.get_alive_agents()
        co_players = (
            set(self.co_divine_map.keys())
            | all_mediums
            | self.bodyguard_co_set
        )
        candidates = [
            a for a in alive
            if a != self.info.agent
            and a not in co_players
            and a not in my_white_targets
        ]
        if candidates:
            target = random.choice(candidates)  # noqa: S311
            self.possessed_black_target = target
            self.possessed_black_history.append(target)

    def _get_template_keys(self) -> dict[str, Any]:
        """Get template keys for Jinja2 rendering.

        Jinja2テンプレートに渡すキーを取得する.
        基底クラスのキーに加えて狂人固有のキーを追加する.

        Returns:
            dict[str, Any]: Template keys / テンプレートキー
        """
        keys = super()._get_template_keys()
        keys["possessed_black_target"] = self.possessed_black_target
        keys["possessed_exposed"] = self.possessed_exposed
        my_results = self.co_divine_map.get(self.info.agent, {}) if self.info else {}
        co_map_blacks = sum(1 for r in my_results.values() if _is_black(r))
        keys["possessed_black_count"] = max(len(self.possessed_black_history), co_map_blacks)
        return keys

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
