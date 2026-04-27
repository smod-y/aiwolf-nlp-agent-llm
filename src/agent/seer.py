"""Module that defines the Seer agent class.

占い師のエージェントクラスを定義するモジュール.
"""

from __future__ import annotations

from typing import Any

from aiwolf_nlp_common.packet import Role, Species

from agent.agent import Agent


class Seer(Agent):
    """Seer agent class.

    占い師のエージェントクラス.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,  # noqa: ARG002
    ) -> None:
        """Initialize the seer agent.

        占い師のエージェントを初期化する.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role (ignored, always set to SEER) / 役職(無視され、常にSEERに設定)
        """
        super().__init__(config, name, game_id, Role.SEER)
        self.divine_results: dict[str, str] = {}

    def daily_initialize(self) -> None:
        """Perform processing for daily initialization request.

        昼開始リクエストに対する処理を行う.
        first-person の占い結果を蓄積した上で, 親クラスで CO 抽出と LLM 送信を行う.
        """
        if self.info and self.info.divine_result:
            target = self.info.divine_result.target
            result = self.info.divine_result.result
            if result == Species.WEREWOLF:
                self.divine_results[target] = "黒(人狼)"
            else:
                self.divine_results[target] = "白(人間)"
        super().daily_initialize()

    def _get_template_keys(self) -> dict[str, Any]:
        """Get template keys including divine results map.

        占い結果マップを含むテンプレートキーを取得する.

        Returns:
            dict[str, Any]: Template keys / テンプレートキー
        """
        keys = super()._get_template_keys()
        keys["divine_results"] = self.divine_results
        return keys

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.

        Returns:
            str: Talk message / 発言メッセージ
        """
        return super().talk()

    def divine(self) -> str:
        """Return response to divine request.

        占いリクエストに対する応答を返す.

        Returns:
            str: Agent name to divine / 占い対象のエージェント名
        """
        return super().divine()

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        return super().vote()
