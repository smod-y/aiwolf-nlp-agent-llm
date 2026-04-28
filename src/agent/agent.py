"""Module that defines the base class for agents.

エージェントの基底クラスを定義するモジュール.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, ClassVar, ParamSpec, TypeVar, cast

from dotenv import load_dotenv
from jinja2 import Template
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage

from aiwolf_nlp_common.packet import Info, Packet, Request, Role, Setting, Status, Talk

from utils.agent_logger import AgentLogger
from utils.stoppable_thread import StoppableThread

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
T = TypeVar("T")


class Agent:
    """Base class for agents.

    エージェントの基底クラス.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,
    ) -> None:
        """Initialize the agent.

        エージェントの初期化を行う.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role / 役職
        """
        self.config = config
        self.agent_name = name
        self.agent_logger = AgentLogger(config, name, game_id)
        self.request: Request | None = None
        self.info: Info | None = None
        self.setting: Setting | None = None
        self.talk_history: list[Talk] = []
        self.whisper_history: list[Talk] = []
        self.role = role
        # グループチャット方式
        self.in_talk_phase = False
        self.in_whisper_phase = False

        self.sent_talk_count: int = 0
        self.sent_whisper_count: int = 0
        self.llm_model: BaseChatModel | None = None
        self.llm_message_history: list[BaseMessage] = []

        # 占い CO マップ: {占い CO したエージェント名: {対象: "白(人間)" | "黒(人狼)"}}
        # 全エージェントが talk_history からの抽出で蓄積し、矛盾検出に使う
        self.co_divine_map: dict[str, dict[str, str]] = {}
        self._last_co_scan_idx: int = 0

        # ライン精査用マップ:
        # {actor: {target: "strong_support|weak_support|flat|weak_oppose|strong_oppose"}}
        # talk_history から抽出された各プレイヤーの他プレイヤーに対する評価スタンス。
        # ライン切れ（同陣営らしさ判定）と思考曖昧者検出に使う。
        self.line_map: dict[str, dict[str, str]] = {}
        # 遷移履歴: {actor: {target: [(day, talk_idx, stance), ...]}}
        # 思考の矛盾を後から検出するため遷移を保存
        self.line_history: dict[str, dict[str, list[tuple[int, int, str]]]] = {}
        # 符号反転回数（support↔oppose の切り替わりのみカウント、flat 経由は除外）
        self.line_flip_count: dict[str, int] = {}
        self._last_line_scan_idx: int = 0

        # プロフィール（ペルソナ）は INITIALIZE パケットでのみ届くため、
        # 後続パケットで失われないよう独自にキャッシュする
        self.cached_profile: str | None = None

        load_dotenv(Path(__file__).parent.joinpath("./../../config/.env"))

    @staticmethod
    def timeout(func: Callable[P, T]) -> Callable[P, T]:
        """Decorator to set action timeout.

        アクションタイムアウトを設定するデコレータ.

        Args:
            func (Callable[P, T]): Function to be decorated / デコレート対象の関数

        Returns:
            Callable[P, T]: Function with timeout functionality / タイムアウト機能を追加した関数
        """

        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            res: T | Exception = Exception("No result")

            def execute_with_timeout() -> None:
                nonlocal res
                try:
                    res = func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    res = e

            thread = StoppableThread(target=execute_with_timeout)
            thread.start()
            self = args[0] if args else None
            if not isinstance(self, Agent):
                raise TypeError(self, " is not an Agent instance")
            timeout_value = (self.setting.timeout.action if hasattr(self, "setting") and self.setting else 0) // 1000
            if timeout_value > 0:
                thread.join(timeout=timeout_value)
                if thread.is_alive():
                    self.agent_logger.logger.warning(
                        "アクションがタイムアウトしました: %s",
                        self.request,
                    )
                    if bool(self.config["agent"]["kill_on_timeout"]):
                        thread.stop()
                        self.agent_logger.logger.warning(
                            "アクションを強制終了しました: %s",
                            self.request,
                        )
            else:
                thread.join()
            if isinstance(res, Exception):  # type: ignore[arg-type]
                raise res
            return res

        return _wrapper

    def set_packet(self, packet: Packet) -> None:
        """Set packet information.

        パケット情報をセットする.

        Args:
            packet (Packet): Received packet / 受信したパケット
        """
        self.request = packet.request
        if packet.info:
            # プロフィールは INITIALIZE 時のみ含まれる仕様なので、初回受信時にキャッシュして保持する
            if packet.info.profile is not None:
                self.cached_profile = packet.info.profile
            self.info = packet.info
        if packet.setting:
            self.setting = packet.setting
        if packet.talk_history:
            self.talk_history.extend(packet.talk_history)
        if packet.whisper_history:
            self.whisper_history.extend(packet.whisper_history)

        # グループチャット方式
        if packet.new_talk:
            self.talk_history.append(packet.new_talk)
            self.on_talk_received(packet.new_talk)
        if packet.new_whisper:
            self.whisper_history.append(packet.new_whisper)
            self.on_whisper_received(packet.new_whisper)

        if self.request == Request.INITIALIZE:
            self.talk_history: list[Talk] = []
            self.whisper_history: list[Talk] = []
            self.llm_message_history: list[BaseMessage] = []
            self.co_divine_map = {}
            self._last_co_scan_idx = 0
            self.line_map = {}
            self.line_history = {}
            self.line_flip_count = {}
            self._last_line_scan_idx = 0
        self.agent_logger.logger.debug(packet)

    def get_alive_agents(self) -> list[str]:
        """Get the list of alive agents.

        生存しているエージェントのリストを取得する.

        Returns:
            list[str]: List of alive agent names / 生存エージェント名のリスト
        """
        if not self.info:
            return []
        return [k for k, v in self.info.status_map.items() if v == Status.ALIVE]

    def on_talk_received(self, talk: Talk) -> None:
        """Called when a new talk is received (freeform mode).

        新しいトークを受信した時に呼ばれる (グループチャット方式用).

        Args:
            talk (Talk): Received talk / 受信したトーク
        """

    def on_whisper_received(self, whisper: Talk) -> None:
        """Called when a new whisper is received (freeform mode).

        新しい囁きを受信した時に呼ばれる (グループチャット方式用).

        Args:
            whisper (Talk): Received whisper / 受信した囁き
        """

    async def handle_talk_phase(self, send: Callable[[str], None]) -> None:
        """Handle talk phase in freeform mode.

        グループチャット方式でのトークフェーズ処理.

        Args:
            send (Callable[[str], None]): Send function / 送信関数
        """
        while self.in_talk_phase:
            if self.info and self.info.remain_count is not None and self.info.remain_count <= 0:
                break

            text = self.talk()
            if not self.in_talk_phase:
                break
            send(text)
            await asyncio.sleep(5)

    async def handle_whisper_phase(self, send: Callable[[str], None]) -> None:
        """Handle whisper phase in freeform mode.

        グループチャット方式での囁きフェーズ処理.

        Args:
            send (Callable[[str], None]): Send function / 送信関数
        """
        while self.in_whisper_phase:
            if self.info and self.info.remain_count is not None and self.info.remain_count <= 0:
                break

            text = self.whisper()
            if not self.in_whisper_phase:
                break
            send(text)
            await asyncio.sleep(5)

    def _resolve_prompt(self, key: str, *, merge_default: bool = False) -> str | None:
        """Resolve a prompt template by request key and role.

        リクエストキーと役職からプロンプトテンプレートを解決する.
        merge_default=False (既定): role固有 > default > 文字列そのまま(後方互換)
        merge_default=True: default と role固有 を連結して返す（system プロンプト用）

        Args:
            key (str): Prompt config key (e.g. "talk", "system") / プロンプト設定キー
            merge_default (bool): If True, concatenate default and role-specific
                sections instead of choosing one / Trueなら default と役職別を連結

        Returns:
            str | None: Resolved prompt template or None / 解決されたプロンプトテンプレートまたはNone
        """
        prompt_config: str | dict[str, str] | None = self.config["prompt"].get(key)
        if prompt_config is None:
            return None
        if isinstance(prompt_config, dict):
            role_key = self.role.value.lower()
            if merge_default:
                parts: list[str] = []
                default_part = prompt_config.get("default")
                if default_part:
                    parts.append(default_part)
                role_part = prompt_config.get(role_key)
                if role_part:
                    parts.append(role_part)
                return "\n\n".join(parts) if parts else None
            return prompt_config.get(role_key) or prompt_config.get("default")
        return prompt_config

    def _get_template_keys(self) -> dict[str, Any]:
        """Get template keys for Jinja2 rendering.

        Jinja2テンプレートに渡すキーを取得する.
        サブクラスでオーバーライドして追加のキーを提供できる.

        Returns:
            dict[str, Any]: Template keys / テンプレートキー
        """
        player_num = int(self.config["agent"]["num"])
        # 役職割当の想定: 5人=人狼1, 9人=人狼2, 13人=人狼3
        if player_num <= 5:
            werewolf_total = 1
        elif player_num <= 9:
            werewolf_total = 2
        else:
            werewolf_total = 3
        alive_count = 0
        if self.info is not None:
            alive_count = sum(
                1 for v in self.info.status_map.values() if v == Status.ALIVE
            )
        # 縄余裕: 村側が 1 回ミス吊りしても勝ち筋が残るかの簡易判定
        # alive > werewolf_total * 2 + 1 なら、ミス吊り 1 回を吸収できる
        rope_margin = alive_count > werewolf_total * 2 + 1
        # 残り吊り可能回数（最悪ケースで全ミスしたときに吊れる回数の最大値）
        rope_count = max(0, (alive_count - werewolf_total) // 2)
        # 自分のその日の最初の発話か（毎日リセットされる判定）
        # sent_talk_count は累積オフセットで毎日リセットされないため、別途算出する
        is_first_talk_today = True
        is_first_whisper_today = True
        if self.info is not None:
            for t in self.talk_history:
                if t.day == self.info.day and t.agent == self.info.agent:
                    is_first_talk_today = False
                    break
            for w in self.whisper_history:
                if w.day == self.info.day and w.agent == self.info.agent:
                    is_first_whisper_today = False
                    break
        # 占い CO 結果の集計
        # confirmed_white: 全占い CO から白判定 + 黒判定なし → 確定白（絶対吊らない）
        # partial_white  : 一部の占い CO から白判定（確定白を除く） → 9人村全員生存時のみ吊らない
        # black_judged   : 少なくとも 1 つの黒判定あり → 最優先で吊る
        confirmed_white_players: list[str] = []
        partial_white_players: list[str] = []
        black_judged_players: list[str] = []
        if self.co_divine_map:
            seer_co_count = len(self.co_divine_map)
            white_count: dict[str, int] = {}
            black_count: dict[str, int] = {}
            for results in self.co_divine_map.values():
                for target, result in results.items():
                    if "白" in result:
                        white_count[target] = white_count.get(target, 0) + 1
                    elif "黒" in result:
                        black_count[target] = black_count.get(target, 0) + 1
            for player, w in white_count.items():
                if black_count.get(player, 0) > 0:
                    continue
                if w == seer_co_count:
                    confirmed_white_players.append(player)
                else:
                    partial_white_players.append(player)
            black_judged_players = list(black_count.keys())
        # ライン精査の派生キー算出
        line_derived = self._compute_line_derived(
            confirmed_white_players, alive_count
        )
        return {
            "info": self.info,
            "setting": self.setting,
            "talk_history": self.talk_history,
            "whisper_history": self.whisper_history,
            "role": self.role,
            "sent_talk_count": self.sent_talk_count,
            "sent_whisper_count": self.sent_whisper_count,
            "player_num": player_num,
            "werewolf_total": werewolf_total,
            "alive_count": alive_count,
            "rope_count": rope_count,
            "rope_margin": rope_margin,
            "is_first_talk_today": is_first_talk_today,
            "is_first_whisper_today": is_first_whisper_today,
            "co_divine_map": self.co_divine_map,
            "profile": self.cached_profile,
            "confirmed_white_players": confirmed_white_players,
            "partial_white_players": partial_white_players,
            "black_judged_players": black_judged_players,
            "line_map": self.line_map,
            "my_lines": self.line_map.get(self.info.agent, {}) if self.info else {},
            "line_flip_count": self.line_flip_count,
            **line_derived,
        }

    def _compute_line_derived(
        self,
        confirmed_white_players: list[str],
        alive_count: int,
    ) -> dict[str, Any]:
        """ライン精査由来の派生キーを算出する.

        Returns:
            dict with keys:
            - suspected_via_white_anchor: list of (actor, stance) opposing 確定白
            - trusted_via_white_anchor  : list of (actor, stance) supporting 確定白
            - my_affinity_top           : (most_similar, most_opposite) self との親和
            - white_affinity_top        : 確定白との親和（最初の1人）
            - unstable_thinkers         : flip>=1 の actor 一覧
            - critical_unstable         : flip>=3 の actor 一覧
            - silent_players            : engagement が平均/2 未満の生存プレイヤー
        """
        # 1. 確定白アンカーでの分類
        suspected_via_white_anchor: list[tuple[str, str]] = []
        trusted_via_white_anchor: list[tuple[str, str]] = []
        self_name = self.info.agent if self.info else None
        for white in confirmed_white_players:
            for actor, lines in self.line_map.items():
                if actor == self_name or actor == white:
                    continue
                stance = lines.get(white)
                if stance is None:
                    continue
                if "oppose" in stance:
                    suspected_via_white_anchor.append((actor, stance))
                elif "support" in stance:
                    trusted_via_white_anchor.append((actor, stance))

        # 2. ペア親和度（共通対象による暗黙同盟）
        # affinity[A][B] = sum over targets X of sign(A[X])*sign(B[X])*|A[X]|*|B[X]|
        actors = list(self.line_map.keys())
        affinity: dict[tuple[str, str], int] = {}
        for i, a in enumerate(actors):
            for b in actors[i + 1 :]:
                score = 0
                shared_targets = set(self.line_map[a].keys()) & set(
                    self.line_map[b].keys(),
                )
                for x in shared_targets:
                    sa, sb = self.line_map[a][x], self.line_map[b][x]
                    score += (
                        self._line_sign(sa)
                        * self._line_sign(sb)
                        * self._line_magnitude(sa)
                        * self._line_magnitude(sb)
                    )
                if score != 0:
                    affinity[(a, b)] = score
                    affinity[(b, a)] = score

        # 自分との親和度 Top-2（最も近い・遠い）
        my_affinity_top: dict[str, tuple[str, int] | None] = {
            "closest": None,
            "farthest": None,
        }
        if self_name:
            self_pairs = [(b, sc) for (a, b), sc in affinity.items() if a == self_name]
            if self_pairs:
                self_pairs.sort(key=lambda x: x[1], reverse=True)
                if self_pairs[0][1] > 0:
                    my_affinity_top["closest"] = self_pairs[0]
                if self_pairs[-1][1] < 0:
                    my_affinity_top["farthest"] = self_pairs[-1]

        # 確定白との親和度 Top-2（複数いる場合は最初の確定白で代表）
        white_affinity_top: dict[str, tuple[str, int] | None] = {
            "closest": None,
            "farthest": None,
        }
        if confirmed_white_players:
            anchor = confirmed_white_players[0]
            anchor_pairs = [
                (b, sc) for (a, b), sc in affinity.items() if a == anchor
            ]
            if anchor_pairs:
                anchor_pairs.sort(key=lambda x: x[1], reverse=True)
                if anchor_pairs[0][1] > 0:
                    white_affinity_top["closest"] = anchor_pairs[0]
                if anchor_pairs[-1][1] < 0:
                    white_affinity_top["farthest"] = anchor_pairs[-1]

        # 3. 思考曖昧者
        unstable_thinkers = sorted(
            [a for a, n in self.line_flip_count.items() if n >= 1 and a != self_name],
        )
        critical_unstable = sorted(
            [a for a, n in self.line_flip_count.items() if n >= 3 and a != self_name],
        )

        # 4. 議論不参加者（engagement < 平均/2）
        engagement: dict[str, int] = {}
        for actor, lines in self.line_map.items():
            engagement[actor] = engagement.get(actor, 0) + len(lines)
            for target in lines:
                engagement[target] = engagement.get(target, 0) + 1
        silent_players: list[str] = []
        if engagement and self.info:
            alive_agents = [
                k for k, v in self.info.status_map.items() if v == Status.ALIVE
            ]
            scores = [engagement.get(p, 0) for p in alive_agents]
            avg = sum(scores) / len(scores) if scores else 0.0
            threshold = avg / 2.0
            silent_players = sorted(
                [
                    p
                    for p in alive_agents
                    if engagement.get(p, 0) < threshold and p != self_name
                ],
            )

        return {
            "suspected_via_white_anchor": suspected_via_white_anchor,
            "trusted_via_white_anchor": trusted_via_white_anchor,
            "my_affinity_top": my_affinity_top,
            "white_affinity_top": white_affinity_top,
            "unstable_thinkers": unstable_thinkers,
            "critical_unstable": critical_unstable,
            "silent_players": silent_players,
        }

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Strip leading/trailing ``` code fence lines if present."""
        text = text.strip()
        if not text.startswith("```"):
            return text
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)

    @staticmethod
    def _normalize_co_result(result: Any) -> str | None:  # noqa: ANN401
        """Normalize a divine result string to '白(人間)' / '黒(人狼)' or None."""
        if not result:
            return None
        s = str(result)
        s_upper = s.upper()
        if "黒" in s or "WEREWOLF" in s_upper or "BLACK" in s_upper:
            return "黒(人狼)"
        if "白" in s or "HUMAN" in s_upper or "WHITE" in s_upper:
            return "白(人間)"
        return None

    def _apply_co_extraction_items(self, items: list[Any]) -> None:
        """Apply parsed extraction items to co_divine_map."""
        for item in items:
            if not isinstance(item, dict):
                continue
            d = cast("dict[str, Any]", item)
            co_seer: Any = d.get("co_seer") or d.get("seer")
            target: Any = d.get("target")
            result: Any = d.get("result")
            if not co_seer or not target or result is None:
                continue
            normalized = self._normalize_co_result(result)
            if normalized is None:
                continue
            self.co_divine_map.setdefault(str(co_seer), {})[str(target)] = normalized

    def _extract_co_divine_results(self) -> None:
        """Extract seer-CO divine claims from talk_history via LLM and update co_divine_map.

        トーク履歴から占い CO の主張(誰を占って白/黒だったか)を LLM で抽出し,
        co_divine_map に蓄積する. 全エージェントが他者の占い CO を矛盾検出のために
        集計する用途で, 占い師自身の真の占い結果とは別に管理される(占い師の
        first-person 結果は talk_history の CO 発言経由でこの map にも反映される).
        """
        if self.llm_model is None:
            return
        new_talks = self.talk_history[self._last_co_scan_idx :]
        if not new_talks:
            return
        template = self._resolve_prompt("co_extraction")
        if template is None:
            return

        talks_text = "\n".join(f"Day{t.day} {t.agent}: {t.text}" for t in new_talks)
        if self.co_divine_map:
            existing_map_text = "\n".join(
                f"- {seer}: " + ", ".join(f"{tgt}={res}" for tgt, res in results.items())
                for seer, results in self.co_divine_map.items()
            )
        else:
            existing_map_text = "(まだなし)"

        rendered = (
            Template(template)
            .render(
                talks_text=talks_text,
                existing_map_text=existing_map_text,
                new_talks=new_talks,
                existing_map=self.co_divine_map,
            )
            .strip()
        )

        try:
            response = (self.llm_model | StrOutputParser()).invoke(
                [HumanMessage(content=rendered)],
            )
        except Exception:
            self.agent_logger.logger.exception("Failed to extract CO divine results")
            return

        # 同じ talks を二重に投げないよう、LLM 呼び出し成功時点でスキャン位置を進める
        self._last_co_scan_idx = len(self.talk_history)

        try:
            parsed: Any = json.loads(self._strip_code_fence(response))
        except json.JSONDecodeError:
            self.agent_logger.logger.warning(["CO_EXTRACTION_PARSE_ERROR", response])
            return
        if not isinstance(parsed, list):
            return

        self._apply_co_extraction_items(cast("list[Any]", parsed))
        self.agent_logger.logger.info(["CO_EXTRACTION", self.co_divine_map])

    _VALID_LINE_STANCES: ClassVar[set[str]] = {
        "strong_support",
        "weak_support",
        "flat",
        "weak_oppose",
        "strong_oppose",
    }

    @staticmethod
    def _line_sign(stance: str) -> int:
        """Return +1 / 0 / -1 sign for a line stance."""
        if "support" in stance:
            return 1
        if "oppose" in stance:
            return -1
        return 0

    @staticmethod
    def _line_magnitude(stance: str) -> int:
        """Return magnitude 0 / 1 / 2 for a line stance."""
        if stance.startswith("strong_"):
            return 2
        if stance.startswith("weak_"):
            return 1
        return 0

    def _is_line_flip(self, prev: str, new: str) -> bool:
        """支持↔反対の符号反転のみ flip としてカウント。flat 経由は除外。"""
        p, n = self._line_sign(prev), self._line_sign(new)
        if p == 0 or n == 0:
            return False
        return p != n

    def _apply_line_extraction_items(self, items: list[Any], scan_end_idx: int) -> None:
        """Apply parsed line items to line_map / line_history / line_flip_count."""
        if self.info is None:
            return
        current_day = self.info.day
        for item in items:
            if not isinstance(item, dict):
                continue
            d = cast("dict[str, Any]", item)
            actor: Any = d.get("actor")
            target: Any = d.get("target")
            stance: Any = d.get("stance")
            if not actor or not target or not stance:
                continue
            actor_s, target_s, stance_s = str(actor), str(target), str(stance)
            if stance_s not in self._VALID_LINE_STANCES:
                continue
            # 履歴に追加
            self.line_history.setdefault(actor_s, {}).setdefault(target_s, []).append(
                (current_day, scan_end_idx, stance_s),
            )
            # 既存スタンスと比較して flip 検出（最新採用方式）
            actor_lines = self.line_map.setdefault(actor_s, {})
            prev = actor_lines.get(target_s)
            if prev is not None and self._is_line_flip(prev, stance_s):
                self.line_flip_count[actor_s] = self.line_flip_count.get(actor_s, 0) + 1
            actor_lines[target_s] = stance_s

    def _extract_line_results(self) -> None:
        """talk_history からプレイヤー間のライン（支持/反対）を LLM で抽出する."""
        if self.llm_model is None:
            return
        new_talks = self.talk_history[self._last_line_scan_idx :]
        if not new_talks:
            return
        template = self._resolve_prompt("line_extraction")
        if template is None:
            return

        talks_text = "\n".join(f"Day{t.day} {t.agent}: {t.text}" for t in new_talks)
        if self.line_map:
            existing_map_text = "\n".join(
                f"- {actor}: " + ", ".join(f"{tgt}={st}" for tgt, st in lines.items())
                for actor, lines in self.line_map.items()
            )
        else:
            existing_map_text = "(まだなし)"

        rendered = (
            Template(template)
            .render(
                talks_text=talks_text,
                existing_map_text=existing_map_text,
                new_talks=new_talks,
                existing_map=self.line_map,
            )
            .strip()
        )

        try:
            response = (self.llm_model | StrOutputParser()).invoke(
                [HumanMessage(content=rendered)],
            )
        except Exception:
            self.agent_logger.logger.exception("Failed to extract line results")
            return

        scan_end_idx = len(self.talk_history)
        self._last_line_scan_idx = scan_end_idx

        try:
            parsed: Any = json.loads(self._strip_code_fence(response))
        except json.JSONDecodeError:
            self.agent_logger.logger.warning(["LINE_EXTRACTION_PARSE_ERROR", response])
            return
        if not isinstance(parsed, list):
            return

        self._apply_line_extraction_items(cast("list[Any]", parsed), scan_end_idx)
        self.agent_logger.logger.info(
            ["LINE_EXTRACTION", self.line_map, "flips", self.line_flip_count],
        )

    def _get_compression_config(self) -> dict[str, Any] | None:
        """Get history compression config for the current agent count.

        現在のエージェント数に対応する履歴圧縮設定を取得する.

        Returns:
            dict[str, Any] | None: Compression config or None if disabled /
                圧縮設定、無効の場合はNone
        """
        compression_configs: dict[int | str, Any] | None = self.config["llm"].get("history_compression")
        if compression_configs is None:
            return None
        agent_num = int(self.config["agent"]["num"])
        config: dict[str, Any] | None = compression_configs.get(agent_num) or compression_configs.get(
            str(agent_num),
        )
        if config is None or not config.get("enabled", False):
            return None
        return config

    def _compress_history(self) -> None:
        """Compress old message history by summarizing with LLM.

        古いメッセージ履歴をLLMで要約して圧縮する.
        """
        comp_config = self._get_compression_config()
        if comp_config is None:
            return
        threshold = int(comp_config["threshold"])
        keep_recent = int(comp_config["keep_recent"])
        if len(self.llm_message_history) < threshold:
            return
        if self.llm_model is None:
            return

        summary_template = self.config["prompt"].get("history_summary")
        if summary_template is None:
            return

        old_messages = self.llm_message_history[:-keep_recent]
        recent_messages = self.llm_message_history[-keep_recent:]

        msg_dicts: list[dict[str, str]] = [
            {"type": type(m).__name__, "content": cast("str", m.content)}  # pyright: ignore[reportUnknownMemberType]
            for m in old_messages
        ]
        summary_prompt = Template(summary_template).render(messages=msg_dicts).strip()

        try:
            summary = (self.llm_model | StrOutputParser()).invoke(
                [HumanMessage(content=summary_prompt)],
            )
        except Exception:
            self.agent_logger.logger.exception("Failed to compress history")
            return

        self.agent_logger.logger.info(
            ["HISTORY_COMPRESSION", f"{len(old_messages)} messages -> summary", summary],
        )
        self.llm_message_history = [
            HumanMessage(content=f"[これまでの要約]\n{summary}"),
            *recent_messages,
        ]

    def _send_message_to_llm(self, request: Request | None) -> str | None:
        """Send message to LLM and get response.

        LLMにメッセージを送信して応答を取得する.

        Args:
            request (Request | None): The request type to process / 処理するリクエストタイプ

        Returns:
            str | None: LLM response or None if error occurred / LLMの応答またはエラー時はNone
        """
        if request is None:
            return None
        prompt = self._resolve_prompt(request.lower())
        if prompt is None:
            return None
        if float(self.config["llm"]["sleep_time"]) > 0:
            sleep(float(self.config["llm"]["sleep_time"]))
        self._compress_history()
        key = self._get_template_keys()
        template: Template = Template(prompt)
        prompt = template.render(**key).strip()
        if self.llm_model is None:
            self.agent_logger.logger.error("LLM is not initialized")
            return None

        human_message = HumanMessage(content=prompt)
        messages: list[BaseMessage] = []
        system_template = self._resolve_prompt("system", merge_default=True)
        if system_template:
            system_content = Template(system_template).render(**key).strip()
            messages.append(SystemMessage(content=system_content))
        messages.extend(self.llm_message_history)
        messages.append(human_message)

        try:
            response = (self.llm_model | StrOutputParser()).invoke(messages)
        except Exception:
            self.agent_logger.logger.exception("Failed to send message to LLM")
            return None

        self.llm_message_history.append(human_message)
        self.llm_message_history.append(AIMessage(content=response))
        self.agent_logger.logger.info(["LLM", prompt, response])
        return response

    @timeout
    def name(self) -> str:
        """Return response to name request.

        名前リクエストに対する応答を返す.

        Returns:
            str: Agent name / エージェント名
        """
        return self.agent_name

    def initialize(self) -> None:
        """Perform initialization for game start request.

        ゲーム開始リクエストに対する初期化処理を行う.
        """
        if self.info is None:
            return

        model_type = str(self.config["llm"]["type"])
        match model_type:
            case "openai":
                self.llm_model = ChatOpenAI(
                    model=str(self.config["openai"]["model"]),
                    temperature=float(self.config["openai"]["temperature"]),
                    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
                )
            case "google":
                self.llm_model = ChatGoogleGenerativeAI(
                    model=str(self.config["google"]["model"]),
                    temperature=float(self.config["google"]["temperature"]),
                    api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
                )
            case "antohropic":
                self.llm_model = ChatGoogleGenerativeAI(
                    model=str(self.config["anthropic"]["model"]),
                    temperature=float(self.config["anthropic"]["temperature"]),
                    api_key=SecretStr(os.environ["ANTHROPIC_API_KEY"]),
                )
            case "ollama":
                self.llm_model = ChatOllama(
                    model=str(self.config["ollama"]["model"]),
                    temperature=float(self.config["ollama"]["temperature"]),
                    base_url=str(self.config["ollama"]["base_url"]),
                )
            case "vllm":
                self.llm_model = ChatOpenAI(
                    model=str(self.config["vllm"]["model"]),
                    temperature=float(self.config["vllm"]["temperature"]),
                    base_url=str(self.config["vllm"]["base_url"]),
                    api_key=SecretStr("EMPTY"),
                )
            case _:
                raise ValueError(model_type, "Unknown LLM type")
        self.llm_model = self.llm_model

    def _refresh_extractions(self) -> None:
        """各アクション直前に co_divine_map と line_map を最新の talk_history で更新."""
        self._extract_co_divine_results()
        self._extract_line_results()

    def daily_initialize(self) -> None:
        """Perform processing for daily initialization request.

        昼開始リクエストに対する処理を行う.
        前日のトーク履歴から占い CO・ライン情報を抽出した上で LLM へ送信する.
        """
        self._refresh_extractions()
        self._send_message_to_llm(self.request)

    def whisper(self) -> str:
        """Return response to whisper request.

        囁きリクエストに対する応答を返す.

        Returns:
            str: Whisper message / 囁きメッセージ
        """
        self._refresh_extractions()
        response = self._send_message_to_llm(self.request)
        self.sent_whisper_count = len(self.whisper_history)
        return response or ""

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.

        Returns:
            str: Talk message / 発言メッセージ
        """
        self._refresh_extractions()
        response = self._send_message_to_llm(Request.TALK)
        self.sent_talk_count = len(self.talk_history)
        return response or ""

    def daily_finish(self) -> None:
        """Perform processing for daily finish request.

        昼終了リクエストに対する処理を行う.
        """
        self._refresh_extractions()
        self._send_message_to_llm(self.request)

    def divine(self) -> str:
        """Return response to divine request.

        占いリクエストに対する応答を返す.

        Returns:
            str: Agent name to divine / 占い対象のエージェント名
        """
        self._refresh_extractions()
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def guard(self) -> str:
        """Return response to guard request.

        護衛リクエストに対する応答を返す.

        Returns:
            str: Agent name to guard / 護衛対象のエージェント名
        """
        self._refresh_extractions()
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        self._refresh_extractions()
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def attack(self) -> str:
        """Return response to attack request.

        襲撃リクエストに対する応答を返す.

        Returns:
            str: Agent name to attack / 襲撃対象のエージェント名
        """
        self._refresh_extractions()
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def finish(self) -> None:
        """Perform processing for game finish request.

        ゲーム終了リクエストに対する処理を行う.
        """

    @timeout
    def action(self) -> str | None:  # noqa: C901, PLR0911
        """Execute action according to request type.

        リクエストの種類に応じたアクションを実行する.

        Returns:
            str | None: Action result string or None / アクションの結果文字列またはNone
        """
        match self.request:
            case Request.NAME:
                return self.name()
            case Request.TALK:
                return self.talk()
            case Request.WHISPER:
                return self.whisper()
            case Request.VOTE:
                return self.vote()
            case Request.DIVINE:
                return self.divine()
            case Request.GUARD:
                return self.guard()
            case Request.ATTACK:
                return self.attack()
            case Request.INITIALIZE:
                self.initialize()
            case Request.DAILY_INITIALIZE:
                self.daily_initialize()
            case Request.DAILY_FINISH:
                self.daily_finish()
            case Request.FINISH:
                self.finish()
            case _:
                pass
        return None
