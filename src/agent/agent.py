"""Module that defines the base class for agents.

エージェントの基底クラスを定義するモジュール.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
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

        # 霊能 CO マップ: {霊能 CO したエージェント名: {追放対象: "白(人間)" | "黒(人狼)"}}
        # 霊能は追放済みプレイヤーの判定であり、占い CO 結果との矛盾検出で偽占い師を確定できる
        self.medium_result_map: dict[str, dict[str, str]] = {}
        self._last_medium_scan_idx: int = 0

        # 霊能 CO セット: 霊能者を CO したエージェント名の集合
        # 結果未報告の段階（Day 1 等）でも CO 宣言だけで蓄積するため、medium_result_map とは独立に保持する.
        # 単独 CO・生存中なら確定霊能者 = 確定白として扱う判定材料に使う.
        self.medium_co_set: set[str] = set()
        self._last_medium_co_scan_idx: int = 0

        # 騎士 CO セット: 騎士を CO したエージェント名の集合
        # 護衛・襲撃の判断、player_intel 表示に使う
        self.bodyguard_co_set: set[str] = set()
        self._last_bodyguard_scan_idx: int = 0

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

        # 襲撃された (= 夜中に人狼に噛まれた) プレイヤーの順序付きリスト.
        # info.attacked_agent は前日の襲撃結果のみ。累積で保持して、
        # 「先に襲撃された占い師 = 真占い」の推論材料として使う.
        # 順序が必要なため list で保持 (重複は受信時にチェック).
        self.attacked_players: list[str] = []

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
            # 前夜の襲撃情報を順序付きで蓄積（先に襲撃された占い師=真の判定に使う）
            if packet.info.attacked_agent and packet.info.attacked_agent not in self.attacked_players:
                self.attacked_players.append(packet.info.attacked_agent)
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
            self.medium_result_map = {}
            self._last_medium_scan_idx = 0
            self.medium_co_set = set()
            self._last_medium_co_scan_idx = 0
            self.bodyguard_co_set = set()
            self._last_bodyguard_scan_idx = 0
            self.line_map = {}
            self.line_history = {}
            self.line_flip_count = {}
            self._last_line_scan_idx = 0
            self.attacked_players = []
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
        # 騎士 CO の単独確定判定
        # 騎士 CO が1人だけ（=対抗 CO 無し）の場合の扱い。日数で確度を変える:
        #   Day 1: 真騎士が必ず生存 → 確定騎士 = 確定白（黒判定よりも優先）
        #   Day 2: 真騎士が既に死亡している可能性が残るが、基本的に確定扱い（黒判定よりも優先）
        #   Day 3: ライン精査・他情報と合わせて LLM が最終判断 → 推定どまり
        #   Day 4 以降: 残り人数が少なくライン精査が主軸 → 特別扱いしない
        # 自分自身や、人狼陣営から見て相方人狼が CO した場合は偽 CO 確定なので除外する.
        confirmed_knight_player: str | None = None
        presumed_knight_player: str | None = None
        if self.info is not None and len(self.bodyguard_co_set) == 1:
            single_knight = next(iter(self.bodyguard_co_set))
            if self.info.status_map.get(single_knight) == Status.ALIVE:
                is_self = single_knight == self.info.agent
                is_partner_ww = (
                    self.role == Role.WEREWOLF
                    and self.info.role_map.get(single_knight) == Role.WEREWOLF
                )
                # 自分が本物の騎士なら、他者の騎士 CO は 100% 偽 → 確定騎士に昇格させない
                is_real_bodyguard = self.role == Role.BODYGUARD
                if not is_self and not is_partner_ww and not is_real_bodyguard:
                    day = self.info.day
                    if day <= 2:
                        confirmed_knight_player = single_knight
                    elif day == 3:
                        presumed_knight_player = single_knight
        # 確定騎士は確定白扱いに昇格させる（黒判定があっても優先 = 黒出しした占い師は偽）
        if confirmed_knight_player is not None:
            if confirmed_knight_player not in confirmed_white_players:
                confirmed_white_players.append(confirmed_knight_player)
            if confirmed_knight_player in partial_white_players:
                partial_white_players.remove(confirmed_knight_player)
            if confirmed_knight_player in black_judged_players:
                black_judged_players.remove(confirmed_knight_player)

        # 霊能 CO の単独確定判定
        # 霊能 CO が1人だけ・生存中なら確定霊能 = 確定白。
        # 霊能を騙る人狼/狂人は稀かつリスクが高く、対抗 CO が無ければまず真と扱うのが定石.
        # 自分自身や、人狼陣営から見て相方人狼が CO した場合は偽 CO 確定なので除外する.
        # 結果報告 (medium_result_map) を待たず CO 宣言 (medium_co_set) の段階で判定する
        # → Day 1 で霊能 CO だけが出ているケースでも確定白として扱える.
        confirmed_medium_player: str | None = None
        all_medium_cos = self.medium_co_set | set(self.medium_result_map.keys())
        if self.info is not None and len(all_medium_cos) == 1:
            single_medium = next(iter(all_medium_cos))
            if self.info.status_map.get(single_medium) == Status.ALIVE:
                is_self = single_medium == self.info.agent
                is_partner_ww = (
                    self.role == Role.WEREWOLF
                    and self.info.role_map.get(single_medium) == Role.WEREWOLF
                )
                if not is_self and not is_partner_ww:
                    confirmed_medium_player = single_medium
        if confirmed_medium_player is not None:
            if confirmed_medium_player not in confirmed_white_players:
                confirmed_white_players.append(confirmed_medium_player)
            if confirmed_medium_player in partial_white_players:
                partial_white_players.remove(confirmed_medium_player)
            if confirmed_medium_player in black_judged_players:
                black_judged_players.remove(confirmed_medium_player)

        # 「先に襲撃された占い師 = 真占い師」の推論。
        # 人狼陣営は真占いを潰しにかかるため、最初に噛まれた占い CO は真の可能性が極めて高い.
        # その他の占い CO は (後に同じく噛まれていても) 偏狂人候補として扱う.
        # 両占いが噛まれた場合に「両方真」とは判断できないため、必ず最初の1人のみを真候補に絞る.
        first_attacked_seer: str | None = None
        likely_fake_seers_via_attack: list[str] = []
        if self.attacked_players and self.co_divine_map:
            for attacked in self.attacked_players:
                if attacked in self.co_divine_map:
                    first_attacked_seer = attacked
                    break
            if first_attacked_seer is not None:
                for seer_co in self.co_divine_map:
                    if seer_co != first_attacked_seer:
                        likely_fake_seers_via_attack.append(seer_co)
        likely_real_seers_via_attack: list[str] = (
            [first_attacked_seer] if first_attacked_seer is not None else []
        )
        # ライン精査の派生キー算出
        line_derived = self._compute_line_derived(
            confirmed_white_players, alive_count
        )
        # 生存状況に応じた表示マーカー（死亡時のみ "(死亡)" を付ける。生存はマーカーなし）
        def _alive_marker(name: str) -> str:
            if self.info is None:
                return ""
            return "" if self.info.status_map.get(name) == Status.ALIVE else "(死亡)"

        # 占い CO 一覧の bullet 文字列を precompute（複数テンプレートで重複していた Jinja ループを排除）
        # actor/target 共に死亡時は "(死亡)" を付与して、LLM が死亡プレイヤーを行動対象に選ばないよう支援
        co_divine_lines = "\n".join(
            f"- {seer}{_alive_marker(seer)}: "
            + ", ".join(
                f"{tgt}{_alive_marker(tgt)}={res}" for tgt, res in results.items()
            )
            for seer, results in self.co_divine_map.items()
        )
        # 自分への占い判定を [{"seer": ..., "color": "白" or "黒"}, ...] 形式で precompute
        my_judgments: list[dict[str, str]] = []
        if self.info is not None:
            self_name = self.info.agent
            for seer, results in self.co_divine_map.items():
                if self_name not in results:
                    continue
                result = results[self_name]
                if "白" in result:
                    my_judgments.append({"seer": seer, "color": "白"})
                elif "黒" in result:
                    my_judgments.append({"seer": seer, "color": "黒"})
        # 霊能 CO 一覧の bullet 文字列を precompute
        medium_result_lines = "\n".join(
            f"- {medium}: " + ", ".join(f"{tgt}={res}" for tgt, res in results.items())
            for medium, results in self.medium_result_map.items()
        )
        # 確定偽占い師: 霊能結果と占い結果に矛盾がある占い師 (例: 占い師 X が「Y は白」、霊能が「Y は黒」)
        # 霊能 CO が複数あると真偽不明なので、1 件のときのみ判定（安全側に倒す）
        confirmed_fake_seers: list[str] = []
        if len(self.medium_result_map) == 1:
            medium_results = next(iter(self.medium_result_map.values()))
            for seer, seer_results in self.co_divine_map.items():
                for target, seer_color in seer_results.items():
                    medium_color = medium_results.get(target)
                    if medium_color is None:
                        continue
                    # 白/黒のラベルが一致しなければ矛盾
                    seer_white = "白" in seer_color
                    medium_white = "白" in medium_color
                    if seer_white != medium_white:
                        if seer not in confirmed_fake_seers:
                            confirmed_fake_seers.append(seer)
                        break
        # プレイヤーごとの統合インテル: 役職主張 + 受領占い結果 + グレー判定
        # 全プレイヤー（生存・死亡含む）を対象に、自分視点で集約した情報を1行ずつまとめる
        # 死亡者には "(死亡)" マーカーを付与
        player_intel_lines = ""
        if self.info is not None:
            all_players_for_intel = list(self.info.status_map.keys())
            # 受領占い: target -> [(seer, color)]
            incoming_divine: dict[str, list[tuple[str, str]]] = {}
            for seer, seer_results in self.co_divine_map.items():
                for target, color in seer_results.items():
                    incoming_divine.setdefault(target, []).append((seer, color))
            intel_rows: list[str] = []
            for p in all_players_for_intel:
                # 役職主張
                roles: list[str] = []
                if p in self.co_divine_map:
                    if p in confirmed_fake_seers:
                        roles.append("狂人（偽占いCO）")
                    elif p == first_attacked_seer:
                        roles.append("占い師（先に襲撃=真濃厚）")
                    elif p in likely_fake_seers_via_attack:
                        roles.append("占い師（対抗が先に襲撃された=偽濃厚）")
                    else:
                        roles.append("占い師")
                if p in self.medium_co_set or p in self.medium_result_map:
                    if p == confirmed_medium_player:
                        roles.append("霊能者（確定=白扱い）")
                    else:
                        roles.append("霊能者")
                if p in self.bodyguard_co_set:
                    if p == confirmed_knight_player:
                        roles.append("騎士（確定=白扱い）")
                    elif p == presumed_knight_player:
                        roles.append("騎士（対抗無し・濃厚）")
                    else:
                        roles.append("騎士")
                role_str = "、".join(roles) if roles else ""
                # 受領占い
                divines = incoming_divine.get(p, [])
                divine_str = (
                    "、".join(f"{s}より{c}" for s, c in divines) if divines else ""
                )
                gray_marker = " ← グレー" if not roles and not divines else ""
                intel_rows.append(
                    f"- {p}{_alive_marker(p)}: 役職=[{role_str}], 占い受領=[{divine_str}]{gray_marker}"
                )
            player_intel_lines = "\n".join(intel_rows)
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
            "co_divine_lines": co_divine_lines,
            "my_judgments": my_judgments,
            "medium_result_map": self.medium_result_map,
            "medium_result_lines": medium_result_lines,
            "medium_co_set": self.medium_co_set,
            "confirmed_fake_seers": confirmed_fake_seers,
            "bodyguard_co_set": self.bodyguard_co_set,
            "confirmed_knight_player": confirmed_knight_player,
            "presumed_knight_player": presumed_knight_player,
            "confirmed_medium_player": confirmed_medium_player,
            "attacked_players": self.attacked_players,
            "first_attacked_seer": first_attacked_seer,
            "likely_real_seers_via_attack": likely_real_seers_via_attack,
            "likely_fake_seers_via_attack": likely_fake_seers_via_attack,
            "player_intel_lines": player_intel_lines,
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

        # 5. 整合的支持の判定（ライン support が CO 白判定で説明できるか）
        # consistent_supports: (supporter, supported, 説明文)
        consistent_supports: list[tuple[str, str, str]] = []
        for supporter, lines in self.line_map.items():
            for supported, stance in lines.items():
                if "support" not in stance:
                    continue
                # ① supporter が supported を白出ししている → 自分が白と判定した相手を擁護
                if "白" in self.co_divine_map.get(supporter, {}).get(supported, ""):
                    consistent_supports.append(
                        (supporter, supported, "白出しした相手を擁護（自然）"),
                    )
                # ② supported が supporter を白出ししている → 白判定された側が擁護を返す
                elif "白" in self.co_divine_map.get(supported, {}).get(supporter, ""):
                    consistent_supports.append(
                        (supporter, supported, "白判定してくれた相手に擁護を返す（自然）"),
                    )

        # 6. 陣営構図（alliance_blocks）: union-find で正の関係（support / 白判定）を連結
        alliance_blocks = self._compute_alliance_blocks()

        # 7. 人狼チーム情報（WW のみ意味を持つが、テンプレ側で role 判定するので全員に渡す）
        # role_map は WW には他 WW も見える（仕様）
        werewolf_partners: list[str] = []
        partners_alive_count = 0
        if self.info is not None and self.info.role_map:
            for agent, r in self.info.role_map.items():
                if r == Role.WEREWOLF and agent != self.info.agent:
                    werewolf_partners.append(agent)
                    if self.info.status_map.get(agent) == Status.ALIVE:
                        partners_alive_count += 1
        # 相方を黒判定した占い師（=真占い師確定の有力候補）
        partner_black_judged_by: list[str] = []
        # 相方を白判定した占い師（=狂人/偽占い師確定の有力候補。味方として扱う）
        partner_white_judged_by: list[str] = []
        for seer, results in self.co_divine_map.items():
            for partner in werewolf_partners:
                partner_result = results.get(partner, "")
                if "黒" in partner_result and seer not in partner_black_judged_by:
                    partner_black_judged_by.append(seer)
                if "白" in partner_result and seer not in partner_white_judged_by:
                    partner_white_judged_by.append(seer)
        # 自分を黒判定した占い師（黒判定された WW 用）
        my_black_judged_by: list[str] = []
        if self_name:
            for seer, results in self.co_divine_map.items():
                if "黒" in results.get(self_name, ""):
                    my_black_judged_by.append(seer)

        return {
            "suspected_via_white_anchor": suspected_via_white_anchor,
            "trusted_via_white_anchor": trusted_via_white_anchor,
            "my_affinity_top": my_affinity_top,
            "white_affinity_top": white_affinity_top,
            "unstable_thinkers": unstable_thinkers,
            "critical_unstable": critical_unstable,
            "silent_players": silent_players,
            "consistent_supports": consistent_supports,
            "alliance_blocks": alliance_blocks,
            "werewolf_partners": werewolf_partners,
            "partners_alive_count": partners_alive_count,
            "partner_black_judged_by": partner_black_judged_by,
            "partner_white_judged_by": partner_white_judged_by,
            "my_black_judged_by": my_black_judged_by,
        }

    def _compute_alliance_blocks(self) -> list[list[str]]:
        """正の関係（line support / co 白判定）で連結されたグループを抽出する.

        Union-Find で:
        - line_map で support 関係にあるペア → 同陣営
        - co_divine_map で白判定が出ているペア → 同陣営
        として連結成分を計算し、サイズ 2 以上のグループのみ返す.

        Returns:
            list[list[str]]: 陣営グループ（各グループ内のメンバー名はソート済み）
        """
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def add(p: str) -> None:
            if p not in parent:
                parent[p] = p

        def union(a: str, b: str) -> None:
            add(a)
            add(b)
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for actor, lines in self.line_map.items():
            for target, stance in lines.items():
                if "support" in stance:
                    union(actor, target)
        for seer, results in self.co_divine_map.items():
            for target, result in results.items():
                if "白" in result:
                    union(seer, target)

        groups: dict[str, list[str]] = {}
        for p in parent:
            groups.setdefault(find(p), []).append(p)
        return [sorted(g) for g in groups.values() if len(g) >= 2]

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
        """Apply parsed extraction items to co_divine_map.

        役職マップ（medium_result_map）を見て、既に霊能者として CO 済みのプレイヤーの
        判定発言は霊能結果と判断して占い CO マップには取り込まない（誤分類防止）.
        """
        for item in items:
            if not isinstance(item, dict):
                continue
            d = cast("dict[str, Any]", item)
            co_seer: Any = d.get("co_seer") or d.get("seer")
            target: Any = d.get("target")
            result: Any = d.get("result")
            if not co_seer or not target or result is None:
                continue
            co_seer_str = str(co_seer)
            # 役職マップによるルーティング: 既に霊能者 CO 済みなら、その発言は霊能結果として扱い、占いマップには入れない
            if co_seer_str in self.medium_co_set or co_seer_str in self.medium_result_map:
                continue
            normalized = self._normalize_co_result(result)
            if normalized is None:
                continue
            self.co_divine_map.setdefault(co_seer_str, {})[str(target)] = normalized

    def _apply_medium_extraction_items(self, items: list[Any]) -> None:
        """Apply parsed extraction items to medium_result_map.

        霊能結果を主張したプレイヤーは霊能 CO 済みでもあるため、medium_co_set にも同期して登録する.
        """
        for item in items:
            if not isinstance(item, dict):
                continue
            d = cast("dict[str, Any]", item)
            medium: Any = d.get("medium") or d.get("co_medium")
            target: Any = d.get("target")
            result: Any = d.get("result")
            if not medium or not target or result is None:
                continue
            normalized = self._normalize_co_result(result)
            if normalized is None:
                continue
            medium_str = str(medium)
            self.medium_result_map.setdefault(medium_str, {})[str(target)] = normalized
            self.medium_co_set.add(medium_str)

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

        # 既に霊能者 CO 済みのプレイヤー名を列挙 → これらの発言は霊能結果なので占い CO 抽出対象外
        known_mediums = self.medium_co_set | set(self.medium_result_map.keys())
        known_mediums_text = (
            "、".join(sorted(known_mediums)) if known_mediums else "(なし)"
        )

        rendered = (
            Template(template)
            .render(
                talks_text=talks_text,
                existing_map_text=existing_map_text,
                known_mediums_text=known_mediums_text,
                new_talks=new_talks,
                existing_map=self.co_divine_map,
            )
            .strip()
        )

        try:
            response = (
                self.llm_model.bind(temperature=self._get_temperature("co_extraction"))
                | StrOutputParser()
            ).invoke([HumanMessage(content=rendered)])
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

    def _extract_medium_results(self) -> None:
        """Extract medium-CO claims from talk_history via LLM and update medium_result_map.

        トーク履歴から霊能 CO の主張(誰が追放されて人狼/人間だったか)を LLM で抽出し,
        medium_result_map に蓄積する. 占い CO 結果と矛盾する霊能結果が出現したら
        該当占い師を偽（=狂人）として確定できる.
        """
        if self.llm_model is None:
            return
        new_talks = self.talk_history[self._last_medium_scan_idx :]
        if not new_talks:
            return
        template = self._resolve_prompt("medium_extraction")
        if template is None:
            return

        talks_text = "\n".join(f"Day{t.day} {t.agent}: {t.text}" for t in new_talks)
        if self.medium_result_map:
            existing_map_text = "\n".join(
                f"- {medium}: " + ", ".join(f"{tgt}={res}" for tgt, res in results.items())
                for medium, results in self.medium_result_map.items()
            )
        else:
            existing_map_text = "(まだなし)"

        rendered = (
            Template(template)
            .render(
                talks_text=talks_text,
                existing_map_text=existing_map_text,
                new_talks=new_talks,
                existing_map=self.medium_result_map,
            )
            .strip()
        )

        try:
            response = (
                self.llm_model.bind(temperature=self._get_temperature("medium_extraction"))
                | StrOutputParser()
            ).invoke([HumanMessage(content=rendered)])
        except Exception:
            self.agent_logger.logger.exception("Failed to extract medium results")
            return

        self._last_medium_scan_idx = len(self.talk_history)

        try:
            parsed: Any = json.loads(self._strip_code_fence(response))
        except json.JSONDecodeError:
            self.agent_logger.logger.warning(["MEDIUM_EXTRACTION_PARSE_ERROR", response])
            return
        if not isinstance(parsed, list):
            return

        self._apply_medium_extraction_items(cast("list[Any]", parsed))
        self.agent_logger.logger.info(["MEDIUM_EXTRACTION", self.medium_result_map])

    def _apply_bodyguard_extraction_items(self, items: list[Any]) -> None:
        """Apply parsed extraction items to bodyguard_co_set."""
        for item in items:
            if not isinstance(item, dict):
                continue
            d = cast("dict[str, Any]", item)
            player: Any = d.get("player") or d.get("bodyguard")
            if not player:
                continue
            self.bodyguard_co_set.add(str(player))

    def _extract_bodyguard_co(self) -> None:
        """Extract bodyguard CO declarations from talk_history via LLM and update bodyguard_co_set.

        トーク履歴から騎士 CO の宣言を LLM で抽出し, bodyguard_co_set に蓄積する.
        """
        if self.llm_model is None:
            return
        new_talks = self.talk_history[self._last_bodyguard_scan_idx :]
        if not new_talks:
            return
        template = self._resolve_prompt("bodyguard_extraction")
        if template is None:
            return

        talks_text = "\n".join(f"Day{t.day} {t.agent}: {t.text}" for t in new_talks)
        existing_text = (
            "、".join(sorted(self.bodyguard_co_set)) if self.bodyguard_co_set else "(まだなし)"
        )

        rendered = (
            Template(template)
            .render(
                talks_text=talks_text,
                existing_set_text=existing_text,
                new_talks=new_talks,
                existing_set=self.bodyguard_co_set,
            )
            .strip()
        )

        try:
            response = (
                self.llm_model.bind(temperature=self._get_temperature("bodyguard_extraction"))
                | StrOutputParser()
            ).invoke([HumanMessage(content=rendered)])
        except Exception:
            self.agent_logger.logger.exception("Failed to extract bodyguard CO")
            return

        self._last_bodyguard_scan_idx = len(self.talk_history)

        try:
            parsed: Any = json.loads(self._strip_code_fence(response))
        except json.JSONDecodeError:
            self.agent_logger.logger.warning(["BODYGUARD_EXTRACTION_PARSE_ERROR", response])
            return
        if not isinstance(parsed, list):
            return

        self._apply_bodyguard_extraction_items(cast("list[Any]", parsed))
        self.agent_logger.logger.info(["BODYGUARD_EXTRACTION", sorted(self.bodyguard_co_set)])

    def _apply_medium_co_extraction_items(self, items: list[Any]) -> None:
        """Apply parsed extraction items to medium_co_set."""
        for item in items:
            if not isinstance(item, dict):
                continue
            d = cast("dict[str, Any]", item)
            player: Any = d.get("player") or d.get("medium")
            if not player:
                continue
            self.medium_co_set.add(str(player))

    def _extract_medium_co(self) -> None:
        """Extract medium-CO declarations from talk_history via LLM and update medium_co_set.

        トーク履歴から霊能者 CO の宣言を LLM で抽出し, medium_co_set に蓄積する.
        霊能結果の有無に関わらず宣言だけで追加するため, Day 1 でも確定霊能者として扱える.
        """
        if self.llm_model is None:
            return
        new_talks = self.talk_history[self._last_medium_co_scan_idx :]
        if not new_talks:
            return
        template = self._resolve_prompt("medium_co_extraction")
        if template is None:
            return

        talks_text = "\n".join(f"Day{t.day} {t.agent}: {t.text}" for t in new_talks)
        existing_text = (
            "、".join(sorted(self.medium_co_set)) if self.medium_co_set else "(まだなし)"
        )

        rendered = (
            Template(template)
            .render(
                talks_text=talks_text,
                existing_set_text=existing_text,
                new_talks=new_talks,
                existing_set=self.medium_co_set,
            )
            .strip()
        )

        try:
            response = (
                self.llm_model.bind(temperature=self._get_temperature("medium_co_extraction"))
                | StrOutputParser()
            ).invoke([HumanMessage(content=rendered)])
        except Exception:
            self.agent_logger.logger.exception("Failed to extract medium CO")
            return

        self._last_medium_co_scan_idx = len(self.talk_history)

        try:
            parsed: Any = json.loads(self._strip_code_fence(response))
        except json.JSONDecodeError:
            self.agent_logger.logger.warning(["MEDIUM_CO_EXTRACTION_PARSE_ERROR", response])
            return
        if not isinstance(parsed, list):
            return

        self._apply_medium_co_extraction_items(cast("list[Any]", parsed))
        self.agent_logger.logger.info(["MEDIUM_CO_EXTRACTION", sorted(self.medium_co_set)])

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
            response = (
                self.llm_model.bind(temperature=self._get_temperature("line_extraction"))
                | StrOutputParser()
            ).invoke([HumanMessage(content=rendered)])
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

    def _get_temperature(self, task_name: str | None = None) -> float:
        """Resolve temperature for the given task from the active LLM provider config.

        現在の LLM プロバイダ設定からタスク別 temperature を取得する.
        temperature が単一の数値なら全タスクで同値、辞書なら task_name をキーに
        引き、見つからなければ "default" にフォールバックする.

        Args:
            task_name (str | None): タスク名（"line_extraction" / "talk" 等）.
                None の場合は default 値を返す.

        Returns:
            float: 解決された temperature 値.
        """
        model_type = str(self.config["llm"]["type"])
        temp_config: Any = self.config.get(model_type, {}).get("temperature", 0.7)
        if isinstance(temp_config, dict):
            cfg = cast("dict[str, Any]", temp_config)
            if task_name is not None and task_name in cfg:
                return float(cfg[task_name])
            return float(cfg.get("default", 0.7))
        return float(temp_config)

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
            summary = (
                self.llm_model.bind(temperature=self._get_temperature("history_summary"))
                | StrOutputParser()
            ).invoke([HumanMessage(content=summary_prompt)])
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
            response = (
                self.llm_model.bind(temperature=self._get_temperature(request.lower()))
                | StrOutputParser()
            ).invoke(messages)
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
        default_temperature = self._get_temperature()
        match model_type:
            case "openai":
                self.llm_model = ChatOpenAI(
                    model=str(self.config["openai"]["model"]),
                    temperature=default_temperature,
                    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
                )
            case "google":
                self.llm_model = ChatGoogleGenerativeAI(
                    model=str(self.config["google"]["model"]),
                    temperature=default_temperature,
                    api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
                )
            case "antohropic":
                self.llm_model = ChatGoogleGenerativeAI(
                    model=str(self.config["anthropic"]["model"]),
                    temperature=default_temperature,
                    api_key=SecretStr(os.environ["ANTHROPIC_API_KEY"]),
                )
            case "ollama":
                self.llm_model = ChatOllama(
                    model=str(self.config["ollama"]["model"]),
                    temperature=default_temperature,
                    base_url=str(self.config["ollama"]["base_url"]),
                )
            case "vllm":
                self.llm_model = ChatOpenAI(
                    model=str(self.config["vllm"]["model"]),
                    temperature=default_temperature,
                    base_url=str(self.config["vllm"]["base_url"]),
                    api_key=SecretStr("EMPTY"),
                )
            case _:
                raise ValueError(model_type, "Unknown LLM type")
        self.llm_model = self.llm_model

    def _refresh_extractions(self) -> None:
        """各アクション直前に CO・霊能・ライン情報を最新の talk_history で更新.

        実行順: medium_co → medium_results → bodyguard → co → line
        medium_co / medium_results / bodyguard を先に走らせて役職マップを構築し、
        co_extraction の応用時に既知の霊能者・騎士の発言を弾けるようにする.
        """
        self._extract_medium_co()
        self._extract_medium_results()
        self._extract_bodyguard_co()
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
        if self.info and self.info.status_map.get(self.info.agent) != Status.ALIVE:
            return
        self._refresh_extractions()
        self._send_message_to_llm(self.request)

    def _validate_alive_target(self, response: str | None) -> str:
        """Validate LLM response is a name of an alive agent; fallback to random alive.

        LLM の応答が生存しているエージェント名であることを検証し、無効ならランダムな生存者を返す.
        死亡プレイヤー名・空文字・無効文字列を含む応答に対する安全網.
        """
        alive = self.get_alive_agents()
        if response and response.strip() in alive:
            return response.strip()
        return random.choice(alive)  # noqa: S311

    def divine(self) -> str:
        """Return response to divine request.

        占いリクエストに対する応答を返す.

        Returns:
            str: Agent name to divine / 占い対象のエージェント名
        """
        self._refresh_extractions()
        return self._validate_alive_target(self._send_message_to_llm(self.request))

    def guard(self) -> str:
        """Return response to guard request.

        護衛リクエストに対する応答を返す.

        Returns:
            str: Agent name to guard / 護衛対象のエージェント名
        """
        self._refresh_extractions()
        return self._validate_alive_target(self._send_message_to_llm(self.request))

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        self._refresh_extractions()
        return self._validate_alive_target(self._send_message_to_llm(self.request))

    def attack(self) -> str:
        """Return response to attack request.

        襲撃リクエストに対する応答を返す.

        Returns:
            str: Agent name to attack / 襲撃対象のエージェント名
        """
        self._refresh_extractions()
        return self._validate_alive_target(self._send_message_to_llm(self.request))

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
