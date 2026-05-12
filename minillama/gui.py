"""View layer for the interactive experiment. It renders the dialog transcript, live metrics, route tables, station/line data, and transit graph.
"""
import queue
import re
import tkinter as tk
import time
from difflib import SequenceMatcher
import customtkinter as ctk
from tkinter import ttk

from minillama.config import (
    GUI_WIDTH,
    GUI_HEIGHT,
    GUI_DIALOG_MIN_WIDTH,
    GUI_MAP_MIN_WIDTH,
    GUI_MIN_WIDTH,
    GUI_MIN_HEIGHT,
    GUI_EQUAL_PANE_MIN_WIDTH,
    GUI_REFRESH_MS,
    GUI_FONT_FAMILY,
    GUI_MONO_FONT_FAMILY,
    GUI_FONT_SMALL,
    GUI_FONT_NORMAL,
    GUI_FONT_SECTION,
    GUI_TABLE_FONT_SIZE,
    GUI_TABLE_HEADER_FONT_SIZE,
    GUI_TABLE_ROW_HEIGHT,
    GUI_SECTION_CORNER_RADIUS,
    GUI_SECTION_BORDER_WIDTH,
    GUI_MAIN_PAD,
    GUI_SECTION_PAD_Y,
    GUI_SECTION_HEADER_PAD_X,
    GUI_SECTION_HEADER_PAD_Y,
    GUI_SECTION_CONTENT_PAD_Y,
    GUI_TOGGLE_WIDTH,
    GUI_SELECTOR_HEIGHT,
    GUI_LEGACY_SELECTOR_WIDTH,
    GUI_LEGACY_SELECTOR_HEIGHT,
    GUI_LEGACY_SELECTOR_BUTTON_WIDTH,
    GUI_LEGACY_SELECTOR_BUTTON_HEIGHT,
    GUI_LEGACY_SELECTOR_BUTTON_RADIUS,
    GUI_TEXTBOX_HEIGHT,
    GUI_STATION_TABS_HEIGHT,
    GUI_DEFAULT_TABS_HEIGHT,
    GUI_COLORS,
    GUI_ROUTE_TABLE_COLUMNS,
    GUI_LINE_TABLE_COLUMNS,
    GUI_STATION_LINE_TABLE_COLUMNS,
    GUI_STATION_TIME_TABLE_COLUMNS,
    GUI_ROUTE_TABLE_HEIGHT,
    GUI_LINE_TABLE_HEIGHT,
    GUI_STATION_TABLE_HEIGHT,
    MAP_MIN_WIDTH,
    MAP_MIN_HEIGHT,
    MAP_PADDING_X,
    MAP_PADDING_Y,
    MAP_MIN_SCALE,
    MAP_PARALLEL_LINE_SPACING,
    MAP_PARALLEL_LINE_MIN_SPACING,
    MAP_LINE_CASING_WIDTH,
    MAP_LINE_CASING_MIN_WIDTH,
    MAP_LINE_WIDTH,
    MAP_LINE_MIN_WIDTH,
    MAP_ROUTE_LINE_WIDTH,
    MAP_ROUTE_LINE_MIN_WIDTH,
    MAP_START_DEST_RADIUS,
    MAP_ROUTE_RADIUS,
    MAP_STATION_RADIUS,
    MAP_MIN_STATION_RADIUS,
    MAP_STATION_FONT_SCALE,
    MAP_STATION_MIN_FONT,
    MAP_ROLE_FONT_SCALE,
    MAP_ROLE_MIN_FONT,
    MAP_EDGE_LABEL_MIN_SCALE,
    MAP_EDGE_LABEL_OFFSET,
    MAP_EDGE_LABEL_MIN_OFFSET,
    MAP_EDGE_LABEL_FONT_SCALE,
    MAP_ROUTE_LABEL_FONT_SCALE,
    MAP_EDGE_LABEL_MIN_FONT,
    MAP_LABEL_PAD,
    MAP_LEGEND_RIGHT_OFFSET,
    MAP_LEGEND_TOP,
    MAP_LEGEND_ROW_GAP,
)
from minillama.metrics import COMPARISON_TERMS, COOPERATION_TERMS, TASK_TERMS
from minillama.metro_data import ADJACENCY, LINES, STATION_POS, line_stop_pairs
from minillama.route_planner import (
    estimate_route_time,
    fmt_time,
    line_direction_sequences,
    route_is_valid,
    segment_travel,
)


class DialogWindow:
    """GUI view for one live dialog experiment. It receives controller events and renders transcript, metrics, tables, and graph state.
    """
    def __init__(self, event_queue, scenario):
        """  init   method for this module's MVC responsibility.
        
        Args:
            event_queue: Input value used by `__init__`; see the function signature and caller context for the expected type.
            scenario: Input value used by `__init__`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.event_queue = event_queue
        self.scenario = scenario
        self.current_route = []
        self.snapshot_values = {}
        self.summary_values = {}
        self.metric_values = {}
        self.live_metric_values = {}
        self.stage_metric_values = {}
        self.section_widgets = {}
        self.section_visibility = {}
        self.dialog_started_at = time.time()
        self.metric_buffers = {
            "agent_a_history": [],
            "agent_b_tokens": [],
            "agent_b_bigrams": [],
        }
        self.live_stats = {
            "messages": 0,
            "agent_a_messages": 0,
            "agent_b_messages": 0,
            "words": 0,
            "agent_a_words": 0,
            "agent_b_words": 0,
            "task_terms": 0,
            "station_mentions": 0,
            "comparison_terms": 0,
            "cooperation_terms": 0,
            "questions": 0,
            "agent_a_questions": 0,
            "agent_b_questions": 0,
            "candidate_routes": 0,
            "route_revisions": 0,
            "best_duration": None,
            "warnings": 0,
            "recovered_warnings": 0,
            "pending_warning_recovery": 0,
            "asr_ref_words": 0,
            "asr_ref_chars": 0,
            "asr_word_substitutions": 0,
            "asr_word_deletions": 0,
            "asr_word_insertions": 0,
            "asr_char_substitutions": 0,
            "asr_char_deletions": 0,
            "asr_char_insertions": 0,
            "asr_utterances": 0,
            "asr_sentence_errors": 0,
            "asr_keyword_tp": 0,
            "asr_keyword_fp": 0,
            "asr_keyword_fn": 0,
            "semantic_attempts": 0,
            "semantic_failures": 0,
            "semantic_frame_hits": 0,
            "slot_hits": 0,
            "slot_total": 0,
            "state_updates": 0,
            "valid_state_updates": 0,
            "agent_b_latency_total": 0.0,
            "agent_b_latency_count": 0,
            "agent_a_latency_total": 0.0,
            "agent_a_latency_count": 0,
            "reformulations": 0,
            "finished": False,
        }

        self.root = ctk.CTk()
        self.root.title("Speech Dialog Navigation Evaluation")
        self.root.geometry(f"{GUI_WIDTH}x{GUI_HEIGHT}")
        self.root.minsize(GUI_MIN_WIDTH, GUI_MIN_HEIGHT)
        self.maximize_startup_window()

        self.configure_style()
        self.build_layout()
        self.update_live_dialog_metrics()
        self.draw_network()

        self.canvas.bind("<Configure>", lambda _: self.draw_network())
        self.root.after(GUI_REFRESH_MS, self.process_events)

    def maximize_startup_window(self):
        """Maximize startup window method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        width = self.root.winfo_screenwidth()
        height = self.root.winfo_screenheight()
        self.root.geometry(f"{width}x{height}+0+0")
        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass

    def configure_style(self):
        """Configure style method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Data.Treeview",
            background=GUI_COLORS["table_bg"],
            fieldbackground=GUI_COLORS["table_bg"],
            foreground=GUI_COLORS["text"],
            bordercolor=GUI_COLORS["table_border"],
            lightcolor=GUI_COLORS["table_border"],
            darkcolor=GUI_COLORS["table_border"],
            font=(GUI_MONO_FONT_FAMILY, GUI_TABLE_FONT_SIZE),
            rowheight=GUI_TABLE_ROW_HEIGHT,
        )
        style.configure(
            "Data.Treeview.Heading",
            background=GUI_COLORS["table_heading_bg"],
            foreground=GUI_COLORS["text"],
            bordercolor=GUI_COLORS["table_border"],
            font=(GUI_FONT_FAMILY, GUI_TABLE_HEADER_FONT_SIZE, "bold"),
        )
        style.map("Data.Treeview", background=[("selected", GUI_COLORS["table_selected"])])

    def build_layout(self):
        """Build layout method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.main = ctk.CTkFrame(
            self.root,
            fg_color=GUI_COLORS["app_bg"],
            corner_radius=0,
        )
        self.main.pack(fill=tk.BOTH, expand=True, padx=GUI_MAIN_PAD, pady=GUI_MAIN_PAD)
        left_min_width = max(GUI_DIALOG_MIN_WIDTH, GUI_EQUAL_PANE_MIN_WIDTH)
        right_min_width = MAP_MIN_WIDTH
        self.main.grid_columnconfigure(0, weight=2, minsize=left_min_width, uniform="workspace")
        self.main.grid_columnconfigure(1, weight=1, minsize=right_min_width, uniform="workspace")
        self.main.grid_rowconfigure(0, weight=1)

        self.left = ctk.CTkFrame(self.main, fg_color="transparent")
        self.right = ctk.CTkFrame(self.main, fg_color="transparent")
        self.left.grid(row=0, column=0, sticky="nsew", padx=(0, GUI_MAIN_PAD))
        self.right.grid(row=0, column=1, sticky="nsew")

        self.build_left_panel()
        self.build_right_panel()

    def make_section(self, parent, title, row, sticky="nsew", padx=0, pady=None, key=None, column=0, columnspan=1):
        """Make section method for this module's MVC responsibility.
        
        Args:
            parent: Input value used by `make_section`; see the function signature and caller context for the expected type.
            title: Input value used by `make_section`; see the function signature and caller context for the expected type.
            row: Input value used by `make_section`; see the function signature and caller context for the expected type.
            sticky: Input value used by `make_section`; see the function signature and caller context for the expected type.
            padx: Input value used by `make_section`; see the function signature and caller context for the expected type.
            pady: Input value used by `make_section`; see the function signature and caller context for the expected type.
            key: Input value used by `make_section`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        if pady is None:
            pady = (0, GUI_SECTION_PAD_Y)
        outer = ctk.CTkFrame(
            parent,
            fg_color=GUI_COLORS["panel_bg"],
            border_color=GUI_COLORS["table_border"],
            border_width=GUI_SECTION_BORDER_WIDTH,
            corner_radius=GUI_SECTION_CORNER_RADIUS,
        )
        outer.grid(row=row, column=column, columnspan=columnspan, sticky=sticky, padx=padx, pady=pady)
        outer.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            outer,
            text=title,
            anchor="w",
            font=(GUI_FONT_FAMILY, GUI_FONT_SECTION, "bold"),
            text_color=GUI_COLORS["text"],
        )
        header.grid(row=0, column=0, sticky="ew", padx=GUI_SECTION_HEADER_PAD_X, pady=GUI_SECTION_HEADER_PAD_Y)

        content = ctk.CTkFrame(outer, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew", padx=0, pady=GUI_SECTION_CONTENT_PAD_Y)
        content.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(1, weight=1)
        if key:
            self.section_widgets[key] = outer
            self.section_visibility[key] = tk.BooleanVar(value=True)
        return content

    def build_toggle_bar(self, parent, sections):
        """Build toggle bar method for this module's MVC responsibility.
        
        Args:
            parent: Input value used by `build_toggle_bar`; see the function signature and caller context for the expected type.
            sections: Input value used by `build_toggle_bar`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", pady=(0, GUI_SECTION_PAD_Y))
        for column, (key, label) in enumerate(sections):
            checkbox = ctk.CTkCheckBox(
                bar,
                text=label,
                variable=self.section_visibility[key],
                command=lambda section_key=key: self.toggle_section(section_key),
                width=GUI_TOGGLE_WIDTH,
                font=(GUI_FONT_FAMILY, GUI_FONT_NORMAL),
            )
            checkbox.grid(row=0, column=column, sticky="w", padx=(0, 4))

    def build_snapshot_grid(self, parent):
        """Build a compact KPI grid for the most important live conversation metrics."""
        items = [
            ("run_state", "State"),
            ("elapsed", "Elapsed"),
            ("route_status", "Route"),
            ("live_duration", "Live Time"),
            ("best_duration", "Best Time"),
            ("candidate_routes", "Candidates"),
            ("route_revisions", "Revisions"),
            ("question_rate", "Q Rate"),
            ("messages", "Messages"),
            ("words", "Words"),
            ("task_focus", "Task Focus"),
            ("warnings", "Warnings"),
        ]
        columns = 4
        for column in range(columns):
            parent.grid_columnconfigure(column, weight=1)

        for index, (key, label) in enumerate(items):
            row = index // columns
            column = index % columns
            tile = ctk.CTkFrame(
                parent,
                fg_color=GUI_COLORS["tab_bg"],
                corner_radius=6,
                border_width=1,
                border_color=GUI_COLORS["table_border"],
            )
            tile.grid(row=row, column=column, sticky="nsew", padx=(0, 2), pady=(0, 2))
            tile.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                tile,
                text=label,
                anchor="w",
                font=(GUI_FONT_FAMILY, GUI_FONT_SMALL),
                text_color=GUI_COLORS["muted_text"],
            ).grid(row=0, column=0, sticky="ew", padx=6, pady=(3, 0))

            value = ctk.CTkLabel(
                tile,
                text="-",
                anchor="w",
                font=(GUI_MONO_FONT_FAMILY, GUI_FONT_NORMAL, "bold"),
                text_color=GUI_COLORS["text"],
            )
            value.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 3))
            self.snapshot_values[key] = value

    def build_stage_metrics_panel(self, parent):
        """Build a scrollable metrics panel grouped by speech-dialog processing stage."""
        specs = [
            ("ASR", [("asr_wer", "WER"), ("asr_cer", "CER"), ("asr_ser", "SER"), ("asr_keyword_f1", "Keyword F1"), ("asr_rtf", "RTF")]),
            ("NLU / SLU", [("nlu_slot_accuracy", "Slot Acc"), ("nlu_frame_accuracy", "Frame Acc"), ("nlu_semantic_error_rate", "Semantic Err"), ("nlu_intent_accuracy", "Intent Acc")]),
            ("DST", [("dst_joint_goal_accuracy", "Joint Goal"), ("dst_slot_accuracy", "Slot Acc"), ("dst_state_update_accuracy", "State Update")]),
            ("DM", [("dm_task_success_rate", "Task Success"), ("dm_dialogue_length", "Dialogue Length"), ("dm_recovery_rate", "Recovery Rate"), ("dm_avg_latency", "Avg B Latency")]),
            ("NLG", [("nlg_distinct_1", "Distinct-1"), ("nlg_distinct_2", "Distinct-2"), ("nlg_repetition_rate", "Repetition"), ("nlg_avg_words", "Avg B Words")]),
            ("TTS", [("tts_mos", "MOS"), ("tts_predicted_mos", "Pred MOS"), ("tts_intelligibility", "Intelligibility"), ("tts_speaker_similarity", "Speaker Sim")]),
            ("End-to-End", [("e2e_task_completion_rate", "Completion"), ("e2e_dialogue_success", "Dialogue Success"), ("e2e_latency", "Latency"), ("e2e_turns_per_dialogue", "Turns/Dialog"), ("e2e_error_recovery", "Error Recovery")]),
            ("UX", [("ux_reformulation_rate", "Reformulation"), ("ux_abandonment_rate", "Abandonment"), ("ux_predicted_satisfaction", "Pred Satisfaction"), ("ux_warning_pressure", "Warning Pressure")]),
            ("Robustness", [("robustness_noise", "Noise Robustness"), ("robustness_asr_degradation", "ASR Degradation"), ("robustness_speech_pattern", "Speech Pattern"), ("robustness_bias", "Bias/Fairness")]),
        ]

        scroller = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroller.grid(row=0, column=0, sticky="nsew")
        scroller.grid_columnconfigure(0, weight=1)

        for row, (title, metrics) in enumerate(specs):
            block = ctk.CTkFrame(
                scroller,
                fg_color=GUI_COLORS["tab_bg"],
                border_width=1,
                border_color=GUI_COLORS["table_border"],
                corner_radius=6,
            )
            block.grid(row=row, column=0, sticky="ew", pady=(0, 2))
            block.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(
                block,
                text=title,
                anchor="w",
                font=(GUI_FONT_FAMILY, GUI_FONT_NORMAL, "bold"),
                text_color=GUI_COLORS["text"],
            ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=(3, 1))

            for metric_row, (key, label) in enumerate(metrics, start=1):
                ctk.CTkLabel(
                    block,
                    text=label,
                    anchor="w",
                    font=(GUI_FONT_FAMILY, GUI_FONT_SMALL),
                    text_color=GUI_COLORS["muted_text"],
                ).grid(row=metric_row, column=0, sticky="w", padx=(6, 4), pady=0)
                value = ctk.CTkLabel(
                    block,
                    text="n/a",
                    anchor="e",
                    font=(GUI_MONO_FONT_FAMILY, GUI_FONT_SMALL, "bold"),
                    text_color=GUI_COLORS["text"],
                )
                value.grid(row=metric_row, column=1, sticky="ew", padx=(0, 6), pady=0)
                self.stage_metric_values[key] = value

    def toggle_section(self, key):
        """Toggle section method for this module's MVC responsibility.
        
        Args:
            key: Input value used by `toggle_section`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        widget = self.section_widgets[key]
        if self.section_visibility[key].get():
            widget.grid()
        else:
            widget.grid_remove()

    def build_left_panel(self):
        """Build left panel method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.left.grid_columnconfigure(0, weight=1)
        self.left.grid_rowconfigure(2, weight=1)

        dashboard = ctk.CTkFrame(self.left, fg_color="transparent")
        dashboard.grid(row=1, column=0, sticky="ew", pady=(0, GUI_SECTION_PAD_Y))
        dashboard.grid_columnconfigure(0, weight=1)
        dashboard.grid_columnconfigure(1, weight=1)
        dashboard.grid_columnconfigure(2, weight=2)

        self.snapshot_frame = self.make_section(
            dashboard,
            "Live Snapshot",
            row=0,
            column=0,
            columnspan=3,
            sticky="ew",
            key="snapshot",
        )
        self.build_snapshot_grid(self.snapshot_frame)

        self.summary_frame = self.make_section(
            dashboard,
            "Run Context",
            row=1,
            column=0,
            sticky="nsew",
            key="run",
        )
        self.summary_frame.grid_columnconfigure((1, 3), weight=1)

        summary_rows = [
            (("test_case", "Test"), ("persona", "Persona")),
            (("scenario", "Scenario"), ("speech", "Speech")),
            (("model", "Model"), ("device", "Device")),
            (("agent_a", "Agent A"), ("settings", "Settings")),
            (("run_state", "State"), ("elapsed", "Elapsed")),
            (("warnings", "Warnings"), ("", "")),
        ]
        for row, pair in enumerate(summary_rows):
            for column_offset, (key, label) in enumerate(pair):
                if not key:
                    continue
                label_column = column_offset * 2
                value_column = label_column + 1
                ctk.CTkLabel(
                    self.summary_frame,
                    text=label,
                    anchor="w",
                    font=(GUI_FONT_FAMILY, GUI_FONT_SMALL),
                    text_color=GUI_COLORS["muted_text"],
                ).grid(row=row, column=label_column, sticky="w", padx=(0, 4), pady=0)
                value = ctk.CTkLabel(
                    self.summary_frame,
                    text="-",
                    anchor="w",
                    font=(GUI_MONO_FONT_FAMILY, GUI_FONT_NORMAL, "bold"),
                    text_color=GUI_COLORS["text"],
                    wraplength=170,
                    justify="left",
                )
                value.grid(row=row, column=value_column, sticky="ew", padx=(0, 10), pady=0)
                self.summary_values[key] = value

        self.metrics_frame = self.make_section(
            dashboard,
            "Outcome Metrics",
            row=1,
            column=1,
            sticky="nsew",
            key="evaluation",
        )
        self.metrics_frame.grid_columnconfigure((1, 3), weight=1)

        metric_rows = [
            (("route", "Route"), ("duration", "Duration")),
            (("breakdown", "Breakdown"), ("reference_duration", "Reference")),
            (("reference_route", "Reference Route"), ("best_turn", "Best Turn")),
            (("candidate_routes", "Candidates"), ("route_revisions", "Revisions")),
            (("valid", "Valid"), ("goal", "Goal")),
            (("correct", "Correct"), ("runtime", "Runtime")),
        ]
        for row, pair in enumerate(metric_rows):
            for column_offset, (key, label) in enumerate(pair):
                label_column = column_offset * 2
                value_column = label_column + 1
                ctk.CTkLabel(
                    self.metrics_frame,
                    text=label,
                    anchor="w",
                    font=(GUI_FONT_FAMILY, GUI_FONT_NORMAL),
                    text_color=GUI_COLORS["muted_text"],
                ).grid(row=row, column=label_column, sticky="nw", padx=(0, 4), pady=0)
                value = ctk.CTkLabel(
                    self.metrics_frame,
                    text="-",
                    anchor="w",
                    font=(GUI_MONO_FONT_FAMILY, GUI_FONT_NORMAL, "bold"),
                    text_color=GUI_COLORS["text"],
                    wraplength=190,
                    justify="left",
                )
                value.grid(row=row, column=value_column, sticky="ew", padx=(0, 10), pady=0)
                self.metric_values[key] = value

        self.live_metrics_frame = self.make_section(
            dashboard,
            "Conversation Metrics",
            row=1,
            column=2,
            sticky="nsew",
            key="dialog_metrics",
        )
        self.live_metrics_frame.grid_columnconfigure((1, 3), weight=1)
        live_metric_rows = [
            (("messages", "Messages"), ("agent_split", "A/B Turns")),
            (("words", "Words"), ("agent_words", "A/B Words")),
            (("avg_words", "Avg Words"), ("agent_avg_words", "A/B Avg")),
            (("questions", "Questions"), ("agent_questions", "A/B Questions")),
            (("question_rate", "Question Rate"), ("station_mentions", "Station Mentions")),
            (("station_density", "Station Density"), ("task_terms", "Task Terms")),
            (("task_focus", "Task Focus"), ("comparison_terms", "Compare Terms")),
            (("comparison_rate", "Compare Rate"), ("cooperation_terms", "Coop Terms")),
            (("cooperation_rate", "Coop Rate"), ("last_speaker", "Last Speaker")),
        ]
        for row, pair in enumerate(live_metric_rows):
            for column_offset, (key, label) in enumerate(pair):
                label_column = column_offset * 2
                value_column = label_column + 1
                ctk.CTkLabel(
                    self.live_metrics_frame,
                    text=label,
                    anchor="w",
                    font=(GUI_FONT_FAMILY, GUI_FONT_NORMAL),
                    text_color=GUI_COLORS["muted_text"],
                ).grid(row=row, column=label_column, sticky="w", padx=(0, 4), pady=0)
                value = ctk.CTkLabel(
                    self.live_metrics_frame,
                    text="-",
                    anchor="w",
                    font=(GUI_MONO_FONT_FAMILY, GUI_FONT_NORMAL, "bold"),
                    text_color=GUI_COLORS["text"],
                )
                value.grid(row=row, column=value_column, sticky="ew", padx=(0, 10), pady=0)
                self.live_metric_values[key] = value

        self.transcript_frame = self.make_section(
            self.left,
            "Conversation",
            row=2,
            sticky="nsew",
            key="transcript",
        )
        self.transcript_frame.grid_rowconfigure(0, weight=1)
        self.transcript_frame.grid_columnconfigure(0, weight=1)

        self.textbox = ctk.CTkTextbox(
            self.transcript_frame,
            wrap=tk.WORD,
            font=(GUI_FONT_FAMILY, GUI_FONT_NORMAL),
            height=GUI_TEXTBOX_HEIGHT,
            border_width=0,
            fg_color=GUI_COLORS["tab_bg"],
            text_color=GUI_COLORS["text"],
        )
        self.textbox.grid(row=0, column=0, sticky="nsew")
        self.textbox.tag_config("Agent A", foreground=GUI_COLORS["agent_a"])
        self.textbox.tag_config("Agent B", foreground=GUI_COLORS["agent_b"])
        self.textbox.tag_config("system", foreground=GUI_COLORS["subtle_text"])
        self.textbox.tag_config("warning", foreground=GUI_COLORS["warning"])

        self.route_frame = self.make_section(
            self.left,
            "Route Candidate Comparison",
            row=3,
            sticky="ew",
            pady=0,
            key="route",
        )
        self.route_frame.grid_columnconfigure(0, weight=1)

        self.candidate_table = ttk.Treeview(
            self.route_frame,
            columns=("turn", "duration", "delta", "decision", "route"),
            show="headings",
            height=4,
            style="Data.Treeview",
        )
        candidate_columns = [
            ("turn", "Turn", 42, "center", False),
            ("duration", "Time", 54, "center", False),
            ("delta", "Delta", 54, "center", False),
            ("decision", "Result", 72, "center", False),
            ("route", "Candidate route", 320, "w", True),
        ]
        for col, label, width, anchor, stretch in candidate_columns:
            self.candidate_table.heading(col, text=label)
            self.candidate_table.column(col, width=width, minwidth=width, anchor=anchor, stretch=stretch)
        self.candidate_table.grid(row=0, column=0, sticky="ew", pady=(0, 2))

        self.route_table = ttk.Treeview(
            self.route_frame,
            columns=("from", "to", "line", "fullness", "depart", "arrive", "ride", "wait", "transfer"),
            show="headings",
            height=GUI_ROUTE_TABLE_HEIGHT,
            style="Data.Treeview",
        )
        for col, label, width, anchor, stretch in GUI_ROUTE_TABLE_COLUMNS:
            self.route_table.heading(col, text=label)
            self.route_table.column(col, width=width, minwidth=width, anchor=anchor, stretch=stretch)

        self.route_table.grid(row=1, column=0, sticky="ew")
        self.build_toggle_bar(
            self.left,
            (
                ("snapshot", "Live"),
                ("run", "Run"),
                ("evaluation", "Eval"),
                ("dialog_metrics", "Dialog"),
                ("route", "Route"),
                ("transcript", "Conversation"),
            ),
        )

    def build_right_panel(self):
        """Build right panel method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.right.grid_columnconfigure(0, weight=1)
        self.right.grid_rowconfigure(1, weight=1)
        self.right.grid_rowconfigure(3, weight=1)

        self.stage_metrics_frame = self.make_section(
            self.right,
            "Stage Metrics",
            row=1,
            sticky="nsew",
            key="stage_metrics",
        )
        self.stage_metrics_frame.grid_rowconfigure(0, weight=1)
        self.stage_metrics_frame.grid_columnconfigure(0, weight=1)
        self.build_stage_metrics_panel(self.stage_metrics_frame)

        schedule_hour = self.scenario["start_time_min"] // 60 * 60
        self.reference_frame = self.make_section(
            self.right,
            f"Transit Reference ({fmt_time(schedule_hour)}-{fmt_time(schedule_hour + 60)})",
            row=2,
            sticky="ew",
            key="reference",
        )
        self.reference_tabs = self.make_tabs(self.reference_frame, height=GUI_DEFAULT_TABS_HEIGHT)
        self.reference_tabs.add("Lines")
        self.reference_tabs.add("Stations")
        self.build_line_tabs(self.reference_tabs.tab("Lines"))
        self.build_station_tabs(self.reference_tabs.tab("Stations"))

        self.map_frame = self.make_section(self.right, "Network Map", row=3, sticky="nsew", pady=0, key="map")
        self.map_frame.grid_rowconfigure(0, weight=1)
        self.map_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self.map_frame,
            bg=GUI_COLORS["map_bg"],
            highlightthickness=1,
            highlightbackground=GUI_COLORS["map_border"],
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.build_toggle_bar(
            self.right,
            (("stage_metrics", "Stages"), ("reference", "Reference"), ("map", "Map")),
        )

    def build_line_tabs(self, parent):
        """Build line tabs method for this module's MVC responsibility.
        
        Args:
            parent: Input value used by `build_line_tabs`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.line_content = ctk.CTkFrame(parent, fg_color="transparent")
        self.line_content.grid(row=1, column=0, sticky="ew")
        self.line_content.grid_columnconfigure(0, weight=1)
        self.line_selector_values = {
            self.line_tab_label(line_name): line_name
            for line_name in LINES
        }
        self.line_selector = ctk.CTkSegmentedButton(
            parent,
            values=list(self.line_selector_values),
            command=lambda label: self.show_line(self.line_selector_values[label]),
            height=GUI_SELECTOR_HEIGHT,
            font=(GUI_FONT_FAMILY, GUI_FONT_NORMAL, "bold"),
        )
        self.line_selector.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 2))
        first_label = next(iter(self.line_selector_values))
        self.line_selector.set(first_label)
        self.show_line(next(iter(LINES)))

    def build_station_tabs(self, parent):
        """Build station tabs method for this module's MVC responsibility.
        
        Args:
            parent: Input value used by `build_station_tabs`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.station_content = ctk.CTkFrame(parent, fg_color="transparent")
        self.station_content.grid(row=1, column=0, sticky="ew")
        self.station_content.grid_columnconfigure(0, weight=1)
        self.station_selector_values = {
            self.station_label(station): station
            for station in STATION_POS
        }
        self.station_selector = ctk.CTkSegmentedButton(
            parent,
            values=list(self.station_selector_values),
            command=lambda label: self.show_station(self.station_selector_values[label]),
            height=GUI_SELECTOR_HEIGHT,
            font=(GUI_FONT_FAMILY, GUI_FONT_NORMAL, "bold"),
        )
        self.station_selector.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 2))
        self.station_selector.set(self.station_label(self.scenario["start_station"]))
        self.show_station(self.scenario["start_station"])

    def build_selector_area(self, parent, items, command):
        """Build selector area method for this module's MVC responsibility.
        
        Args:
            parent: Input value used by `build_selector_area`; see the function signature and caller context for the expected type.
            items: Input value used by `build_selector_area`; see the function signature and caller context for the expected type.
            command: Input value used by `build_selector_area`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        parent.grid_columnconfigure(1, weight=1)
        selector = ctk.CTkScrollableFrame(
            parent,
            width=GUI_LEGACY_SELECTOR_WIDTH,
            height=GUI_LEGACY_SELECTOR_HEIGHT,
            fg_color=GUI_COLORS["app_bg"],
            corner_radius=GUI_SECTION_CORNER_RADIUS,
        )
        selector.grid(row=0, column=0, sticky="nsw", padx=(0, 2), pady=0)

        content = ctk.CTkFrame(parent, fg_color="transparent")
        content.grid(row=0, column=1, sticky="nsew", pady=0)
        content.grid_columnconfigure(0, weight=1)

        for row, (label, value) in enumerate(items):
            ctk.CTkButton(
                selector,
                text=label,
                width=GUI_LEGACY_SELECTOR_BUTTON_WIDTH,
                height=GUI_LEGACY_SELECTOR_BUTTON_HEIGHT,
                corner_radius=GUI_LEGACY_SELECTOR_BUTTON_RADIUS,
                font=(GUI_FONT_FAMILY, GUI_FONT_SMALL, "bold"),
                command=lambda selected=value: command(selected),
            ).grid(row=row, column=0, sticky="ew", padx=0, pady=(0, 3))

        return content

    def clear_frame(self, frame):
        """Clear frame method for this module's MVC responsibility.
        
        Args:
            frame: Input value used by `clear_frame`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        for child in frame.winfo_children():
            child.destroy()

    def show_line(self, line_name):
        """Show line method for this module's MVC responsibility.
        
        Args:
            line_name: Input value used by `show_line`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.clear_frame(self.line_content)
        self.populate_line_tab(self.line_content, line_name)

    def show_station(self, station):
        """Show station method for this module's MVC responsibility.
        
        Args:
            station: Input value used by `show_station`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.clear_frame(self.station_content)
        self.populate_station_tab(self.station_content, station)

    def make_tabs(self, parent, height=GUI_DEFAULT_TABS_HEIGHT):
        """Make tabs method for this module's MVC responsibility.
        
        Args:
            parent: Input value used by `make_tabs`; see the function signature and caller context for the expected type.
            height: Input value used by `make_tabs`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        tabs = ctk.CTkTabview(
            parent,
            height=height,
            fg_color=GUI_COLORS["tab_bg"],
            segmented_button_fg_color=GUI_COLORS["tab_button_bg"],
            segmented_button_selected_color=GUI_COLORS["tab_selected"],
            segmented_button_selected_hover_color=GUI_COLORS["tab_selected_hover"],
            segmented_button_unselected_color=GUI_COLORS["tab_button_bg"],
            segmented_button_unselected_hover_color=GUI_COLORS["tab_unselected_hover"],
        )
        tabs.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        return tabs

    def populate_line_tab(self, parent, line_name):
        """Populate line tab method for this module's MVC responsibility.
        
        Args:
            parent: Input value used by `populate_line_tab`; see the function signature and caller context for the expected type.
            line_name: Input value used by `populate_line_tab`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        parent.grid_columnconfigure(0, weight=1)
        data = LINES[line_name]
        direction_text = self.line_direction_text(line_name)
        ctk.CTkLabel(
            parent,
            text=f"{self.short_line_label(line_name)}  {direction_text}  every {data['headway']}m  {data.get('fullness', 0)}% full",
            anchor="w",
            font=(GUI_FONT_FAMILY, GUI_FONT_NORMAL + 1, "bold"),
            text_color=GUI_COLORS["text"],
        ).grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 3))

        line_table = ttk.Treeview(
            parent,
            columns=("order", "station", "fullness", "previous", "next", "ride", "elapsed"),
            show="headings",
            height=GUI_LINE_TABLE_HEIGHT,
            style="Data.Treeview",
        )
        for col, label, width, anchor, stretch in GUI_LINE_TABLE_COLUMNS:
            line_table.heading(col, text=label)
            line_table.column(col, width=width, minwidth=width, anchor=anchor, stretch=stretch)

        elapsed = 0
        sequence = self.line_display_sequence(line_name)
        for index, station in enumerate(sequence):
            previous_station = sequence[index - 1] if index > 0 else None
            next_station = sequence[index + 1] if index + 1 < len(sequence) else None
            ride = segment_travel(previous_station, station) if previous_station else 0
            if previous_station:
                elapsed += ride
            line_table.insert(
                "",
                tk.END,
                values=(
                    index + 1,
                    self.station_label(station),
                    f"{data.get('fullness', 0)}%",
                    self.station_label(previous_station) if previous_station else "-",
                    self.station_label(next_station) if next_station else "-",
                    f"{ride}m" if previous_station else "start",
                    f"{elapsed}m",
                ),
            )

        line_table.grid(row=1, column=0, sticky="ew", padx=0, pady=(0, 2))

    def populate_station_tab(self, parent, station):
        """Populate station tab method for this module's MVC responsibility.
        
        Args:
            parent: Input value used by `populate_station_tab`; see the function signature and caller context for the expected type.
            station: Input value used by `populate_station_tab`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        parent.grid_columnconfigure(0, weight=1)
        station_tabs = self.make_tabs(parent, height=GUI_STATION_TABS_HEIGHT)
        station_tabs.add("Lines")
        station_tabs.add("Times")
        self.populate_station_lines_tab(station_tabs.tab("Lines"), station)
        self.populate_station_times_tab(station_tabs.tab("Times"), station)

    def populate_station_lines_tab(self, parent, station):
        """Populate station lines tab method for this module's MVC responsibility.
        
        Args:
            parent: Input value used by `populate_station_lines_tab`; see the function signature and caller context for the expected type.
            station: Input value used by `populate_station_lines_tab`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        parent.grid_columnconfigure(0, weight=1)
        station_line_table = ttk.Treeview(
            parent,
            columns=("line", "fullness", "route", "stop", "neighbors", "travel"),
            show="headings",
            height=GUI_STATION_TABLE_HEIGHT,
            style="Data.Treeview",
        )
        for col, label, width, anchor, stretch in GUI_STATION_LINE_TABLE_COLUMNS:
            station_line_table.heading(col, text=label)
            station_line_table.column(col, width=width, minwidth=width, anchor=anchor, stretch=stretch)

        for line_name in self.lines_serving_station(station):
            previous_station, next_station = self.line_neighbors_at_station(line_name, station)
            travel_parts = []
            if previous_station:
                travel_parts.append(f"{self.station_label(previous_station)}-{self.station_label(station)}:{segment_travel(previous_station, station)}m")
            if next_station:
                travel_parts.append(f"{self.station_label(station)}-{self.station_label(next_station)}:{segment_travel(station, next_station)}m")
            neighbors = (
                f"{self.station_label(previous_station) if previous_station else '-'}"
                f"  |  {self.station_label(next_station) if next_station else '-'}"
            )
            station_line_table.insert(
                "",
                tk.END,
                values=(
                    line_name,
                    f"{LINES[line_name].get('fullness', 0)}%",
                    self.line_direction_text(line_name),
                    self.station_label(station),
                    neighbors,
                    "; ".join(travel_parts),
                ),
            )
        station_line_table.grid(row=0, column=0, sticky="ew", padx=0, pady=0)

    def populate_station_times_tab(self, parent, station):
        """Populate station times tab method for this module's MVC responsibility.
        
        Args:
            parent: Input value used by `populate_station_times_tab`; see the function signature and caller context for the expected type.
            station: Input value used by `populate_station_times_tab`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        parent.grid_columnconfigure(0, weight=1)
        schedule_table = ttk.Treeview(
            parent,
            columns=("line", "movement", "times"),
            show="headings",
            height=GUI_STATION_TABLE_HEIGHT,
            style="Data.Treeview",
        )
        for col, label, width, anchor, stretch in GUI_STATION_TIME_TABLE_COLUMNS:
            schedule_table.heading(col, text=label)
            schedule_table.column(col, width=width, minwidth=width, anchor=anchor, stretch=stretch)

        hour_start = self.scenario["start_time_min"] // 60 * 60
        hour_end = hour_start + 60
        for line_name, movement, times in self.station_hour_train_events(station, hour_start, hour_end):
            schedule_table.insert(
                "",
                tk.END,
                values=(
                    line_name,
                    movement,
                    self.minute_list(times),
                ),
            )
        schedule_table.grid(row=0, column=0, sticky="ew", padx=0, pady=0)

    def station_hour_train_events(self, station, hour_start, hour_end):
        """Station hour train events method for this module's MVC responsibility.
        
        Args:
            station: Input value used by `station_hour_train_events`; see the function signature and caller context for the expected type.
            hour_start: Input value used by `station_hour_train_events`; see the function signature and caller context for the expected type.
            hour_end: Input value used by `station_hour_train_events`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        rows = []
        seen = set()
        for line_name in sorted(LINES):
            for sequence in line_direction_sequences(line_name):
                station_count = len(sequence) - 1 if sequence[0] == sequence[-1] else len(sequence)
                for index in range(station_count):
                    current = sequence[index]
                    if current != station:
                        continue

                    previous_station = sequence[index - 1] if index > 0 else None
                    next_station = sequence[index + 1] if index + 1 < len(sequence) else None
                    movement = self.station_movement_text(previous_station, next_station)
                    key = (line_name, movement)
                    if key in seen:
                        continue
                    seen.add(key)

                    offset = self.station_direction_offset(sequence, index)
                    rows.append((line_name, movement, self.train_times_in_hour(line_name, offset, hour_start, hour_end)))
        return rows

    def line_display_sequence(self, line_name):
        """Line display sequence method for this module's MVC responsibility.
        
        Args:
            line_name: Input value used by `line_display_sequence`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        stops = LINES[line_name]["stops"]
        if LINES[line_name].get("kind") == "Ring" and len(stops) > 2:
            return stops + [stops[0]]
        return stops

    def line_direction_text(self, line_name):
        """Line direction text method for this module's MVC responsibility.
        
        Args:
            line_name: Input value used by `line_direction_text`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        sequence = self.line_display_sequence(line_name)
        return f"{self.station_label(sequence[0])} <-> {self.station_label(sequence[-1])}"

    def lines_serving_station(self, station):
        """Lines serving station method for this module's MVC responsibility.
        
        Args:
            station: Input value used by `lines_serving_station`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return sorted(
            [
                line_name
                for line_name, data in LINES.items()
                if station in data["stops"]
            ],
            key=self.short_line_label,
        )

    def line_neighbors_at_station(self, line_name, station):
        """Line neighbors at station method for this module's MVC responsibility.
        
        Args:
            line_name: Input value used by `line_neighbors_at_station`; see the function signature and caller context for the expected type.
            station: Input value used by `line_neighbors_at_station`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        stops = LINES[line_name]["stops"]
        index = stops.index(station)
        previous_station = stops[index - 1] if index > 0 else None
        next_station = stops[index + 1] if index + 1 < len(stops) else None

        if LINES[line_name].get("kind") == "Ring" and len(stops) > 2:
            previous_station = stops[index - 1]
            next_station = stops[(index + 1) % len(stops)]

        return previous_station, next_station

    def station_movement_text(self, previous_station, next_station):
        """Station movement text method for this module's MVC responsibility.
        
        Args:
            previous_station: Input value used by `station_movement_text`; see the function signature and caller context for the expected type.
            next_station: Input value used by `station_movement_text`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        previous_label = self.station_label(previous_station) if previous_station else "start"
        next_label = self.station_label(next_station) if next_station else "end"
        return f"{previous_label} -> {next_label}"

    def station_direction_offset(self, sequence, station_index):
        """Station direction offset method for this module's MVC responsibility.
        
        Args:
            sequence: Input value used by `station_direction_offset`; see the function signature and caller context for the expected type.
            station_index: Input value used by `station_direction_offset`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        elapsed = 0
        for a, b in zip(sequence[:station_index], sequence[1:station_index + 1]):
            elapsed += segment_travel(a, b)
        return elapsed

    def train_times_in_hour(self, line_name, offset, hour_start, hour_end):
        """Train times in hour method for this module's MVC responsibility.
        
        Args:
            line_name: Input value used by `train_times_in_hour`; see the function signature and caller context for the expected type.
            offset: Input value used by `train_times_in_hour`; see the function signature and caller context for the expected type.
            hour_start: Input value used by `train_times_in_hour`; see the function signature and caller context for the expected type.
            hour_end: Input value used by `train_times_in_hour`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        headway = LINES[line_name]["headway"]
        if hour_start <= offset:
            first = offset
        else:
            elapsed_since_offset = hour_start - offset
            remainder = elapsed_since_offset % headway
            first = hour_start if remainder == 0 else hour_start + headway - remainder
        return list(range(first, hour_end, headway))

    def minute_list(self, times):
        """Minute list method for this module's MVC responsibility.
        
        Args:
            times: Input value used by `minute_list`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        by_hour = {}
        for time in times:
            by_hour.setdefault(time // 60, []).append(time % 60)

        return " ".join(
            f"{hour:02d}: {' '.join(f'{minute:02d}' for minute in minutes)}"
            for hour, minutes in sorted(by_hour.items())
        )

    @staticmethod
    def station_label(station):
        """Station label method for this module's MVC responsibility.
        
        Args:
            station: Input value used by `station_label`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return station[:1].upper()

    def route_label(self, route):
        """Route label method for this module's MVC responsibility.
        
        Args:
            route: Input value used by `route_label`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return " -> ".join(self.station_label(station) for station in route)

    def line_stop_labels(self, line_name):
        """Line stop labels method for this module's MVC responsibility.
        
        Args:
            line_name: Input value used by `line_stop_labels`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        data = LINES[line_name]
        stops = data["stops"]
        if data.get("kind") == "Ring" and len(stops) > 2:
            stops = stops + [stops[0]]
        return " -> ".join(self.station_label(station) for station in stops)

    def compact_line_segment_text(self, line_name):
        """Compact line segment text method for this module's MVC responsibility.
        
        Args:
            line_name: Input value used by `compact_line_segment_text`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return "; ".join(
            f"{self.station_label(a)}-{self.station_label(b)}:{self.segment_minutes(a, b)}m"
            for a, b in line_stop_pairs(line_name, LINES[line_name])
        )

    def segment_minutes(self, a, b):
        """Segment minutes method for this module's MVC responsibility.
        
        Args:
            a: Input value used by `segment_minutes`; see the function signature and caller context for the expected type.
            b: Input value used by `segment_minutes`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        for nxt, _, travel in self.adjacent_segments(a):
            if nxt == b:
                return travel
        return "?"

    def adjacent_segments(self, station):
        """Adjacent segments method for this module's MVC responsibility.
        
        Args:
            station: Input value used by `adjacent_segments`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return ADJACENCY[station]

    def canvas_transform(self):
        """Canvas transform method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        width = max(self.canvas.winfo_width(), MAP_MIN_WIDTH)
        height = max(self.canvas.winfo_height(), MAP_MIN_HEIGHT)

        xs = [x for x, _ in STATION_POS.values()]
        ys = [y for _, y in STATION_POS.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        source_w = max(max_x - min_x, 1)
        source_h = max(max_y - min_y, 1)
        scale = max(
            MAP_MIN_SCALE,
            min(
                (width - 2 * MAP_PADDING_X) / source_w,
                (height - 2 * MAP_PADDING_Y) / source_h,
            ),
        )

        used_w = source_w * scale
        used_h = source_h * scale
        offset_x = (width - used_w) / 2
        offset_y = (height - used_h) / 2

        def transform(point):
            """Transform function for this module's MVC responsibility.
            
            Args:
                point: Input value used by `transform`; see the function signature and caller context for the expected type.
            
            Returns:
                The computed value or side effect documented by the implementation.
            """
            x, y = point
            return (
                offset_x + (x - min_x) * scale,
                offset_y + (y - min_y) * scale,
            )

        return transform, scale

    def draw_network(self):
        """Draw network method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        if not hasattr(self, "canvas"):
            return

        self.canvas.delete("all")
        transform, scale = self.canvas_transform()

        route_steps = self.current_route_steps()
        route_edge_lines = {
            (tuple(sorted((step["from"], step["to"]))), step["line"])
            for step in route_steps
        }
        edge_groups = self.line_edge_groups()
        drawn_edge_labels = set()

        for line_name, data in LINES.items():
            for a, b in line_stop_pairs(line_name, data):
                x1, y1 = transform(STATION_POS[a])
                x2, y2 = transform(STATION_POS[b])
                x1, y1, x2, y2 = self.offset_edge_points(
                    a,
                    b,
                    line_name,
                    x1,
                    y1,
                    x2,
                    y2,
                    scale,
                    edge_groups,
                )
                self.canvas.create_line(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=GUI_COLORS["panel_bg"],
                    width=max(MAP_LINE_CASING_MIN_WIDTH, int(MAP_LINE_CASING_WIDTH * scale)),
                    capstyle=tk.ROUND,
                )

        for line_name, data in LINES.items():
            for a, b in line_stop_pairs(line_name, data):
                x1, y1 = transform(STATION_POS[a])
                x2, y2 = transform(STATION_POS[b])
                x1, y1, x2, y2 = self.offset_edge_points(
                    a,
                    b,
                    line_name,
                    x1,
                    y1,
                    x2,
                    y2,
                    scale,
                    edge_groups,
                )
                self.canvas.create_line(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=data["color"],
                    width=max(MAP_LINE_MIN_WIDTH, int(MAP_LINE_WIDTH * scale)),
                    capstyle=tk.ROUND,
                )
                self.draw_edge_label(
                    a,
                    b,
                    line_name,
                    x1,
                    y1,
                    x2,
                    y2,
                    scale,
                    drawn_edge_labels,
                    is_route_edge=(tuple(sorted((a, b))), line_name) in route_edge_lines,
                )

        for step in route_steps:
            a = step["from"]
            b = step["to"]
            line_name = step["line"]
            x1, y1 = transform(STATION_POS[a])
            x2, y2 = transform(STATION_POS[b])
            x1, y1, x2, y2 = self.offset_edge_points(
                a,
                b,
                line_name,
                x1,
                y1,
                x2,
                y2,
                scale,
                edge_groups,
            )
            self.canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill=GUI_COLORS["route_line"],
                width=max(MAP_ROUTE_LINE_MIN_WIDTH, int(MAP_ROUTE_LINE_WIDTH * scale)),
                capstyle=tk.ROUND,
            )
            self.draw_edge_label(
                a,
                b,
                line_name,
                x1,
                y1,
                x2,
                y2,
                scale,
                drawn_edge_labels,
                is_route_edge=True,
            )

        start = self.scenario["start_station"]
        destination = self.scenario["destination_station"]

        for station, pos in STATION_POS.items():
            x, y = transform(pos)

            if station == start:
                fill = GUI_COLORS["start_station"]
                radius = MAP_START_DEST_RADIUS
                role = "S"
            elif station == destination:
                fill = GUI_COLORS["destination_station"]
                radius = MAP_START_DEST_RADIUS
                role = "D"
            elif station in self.current_route:
                fill = GUI_COLORS["route_station"]
                radius = MAP_ROUTE_RADIUS
                role = ""
            else:
                fill = GUI_COLORS["panel_bg"]
                radius = MAP_STATION_RADIUS
                role = ""

            radius = max(MAP_MIN_STATION_RADIUS, int(radius * scale))
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=fill,
                outline=GUI_COLORS["map_outline"],
                width=1,
            )
            self.canvas.create_text(
                x,
                y,
                text=self.station_label(station),
                font=(GUI_FONT_FAMILY, max(MAP_STATION_MIN_FONT, int(MAP_STATION_FONT_SCALE * scale)), "bold"),
                fill=GUI_COLORS["map_outline"],
            )
            if role:
                self.canvas.create_text(
                    x,
                    y + radius + max(MAP_ROLE_MIN_FONT + 2, int(MAP_STATION_FONT_SCALE * scale)),
                    text=role,
                    font=(GUI_FONT_FAMILY, max(MAP_ROLE_MIN_FONT, int(MAP_ROLE_FONT_SCALE * scale)), "bold"),
                    fill=GUI_COLORS["muted_text"],
                )

        self.draw_line_color_index()
        self.update_route_metric()
        self.update_route_table()

    def current_route_steps(self):
        """Current route steps method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        if len(self.current_route) < 2:
            return []

        estimate = estimate_route_time(
            self.current_route,
            self.scenario["start_time_min"],
            self.scenario["transfer_time_min"],
        )
        if not estimate:
            return []

        return estimate[1]

    def line_edge_groups(self):
        """Line edge groups method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        groups = {}
        for line_name, data in LINES.items():
            for a, b in line_stop_pairs(line_name, data):
                groups.setdefault(tuple(sorted((a, b))), []).append(line_name)
        return groups

    def offset_edge_points(self, a, b, line_name, x1, y1, x2, y2, scale, edge_groups):
        """Offset edge points method for this module's MVC responsibility.
        
        Args:
            a: Input value used by `offset_edge_points`; see the function signature and caller context for the expected type.
            b: Input value used by `offset_edge_points`; see the function signature and caller context for the expected type.
            line_name: Input value used by `offset_edge_points`; see the function signature and caller context for the expected type.
            x1: Input value used by `offset_edge_points`; see the function signature and caller context for the expected type.
            y1: Input value used by `offset_edge_points`; see the function signature and caller context for the expected type.
            x2: Input value used by `offset_edge_points`; see the function signature and caller context for the expected type.
            y2: Input value used by `offset_edge_points`; see the function signature and caller context for the expected type.
            scale: Input value used by `offset_edge_points`; see the function signature and caller context for the expected type.
            edge_groups: Input value used by `offset_edge_points`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        lines = edge_groups.get(tuple(sorted((a, b))), [line_name])
        if len(lines) <= 1 or line_name not in lines:
            return x1, y1, x2, y2

        # Use a canonical geometric direction so parallel lines stay on the
        # same side even when two services define the shared edge in opposite
        # station order, e.g. Ring T-S and East-West S-T.
        if (x1, y1) <= (x2, y2):
            dx = x2 - x1
            dy = y2 - y1
        else:
            dx = x1 - x2
            dy = y1 - y2

        length = max((dx * dx + dy * dy) ** 0.5, 1)
        nx = -dy / length
        ny = dx / length
        spacing = max(MAP_PARALLEL_LINE_MIN_SPACING, int(MAP_PARALLEL_LINE_SPACING * scale))
        index = lines.index(line_name)
        centered_index = index - (len(lines) - 1) / 2
        offset = centered_index * spacing

        return (
            x1 + nx * offset,
            y1 + ny * offset,
            x2 + nx * offset,
            y2 + ny * offset,
        )

    def draw_edge_label(
        self,
        a,
        b,
        line_name,
        x1,
        y1,
        x2,
        y2,
        scale,
        drawn_edge_labels,
        is_route_edge=False,
    ):
        """Draw edge label method for this module's MVC responsibility.
        
        Args:
            a: Input value used by `draw_edge_label`; see the function signature and caller context for the expected type.
            b: Input value used by `draw_edge_label`; see the function signature and caller context for the expected type.
            line_name: Input value used by `draw_edge_label`; see the function signature and caller context for the expected type.
            x1: Input value used by `draw_edge_label`; see the function signature and caller context for the expected type.
            y1: Input value used by `draw_edge_label`; see the function signature and caller context for the expected type.
            x2: Input value used by `draw_edge_label`; see the function signature and caller context for the expected type.
            y2: Input value used by `draw_edge_label`; see the function signature and caller context for the expected type.
            scale: Input value used by `draw_edge_label`; see the function signature and caller context for the expected type.
            drawn_edge_labels: Input value used by `draw_edge_label`; see the function signature and caller context for the expected type.
            is_route_edge: Input value used by `draw_edge_label`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        edge_key = tuple(sorted((a, b)))
        label_key = (edge_key, line_name)
        if label_key in drawn_edge_labels and not is_route_edge:
            return
        if not is_route_edge and scale < MAP_EDGE_LABEL_MIN_SCALE:
            return

        drawn_edge_labels.add(label_key)
        travel = self.segment_minutes(a, b)
        label = f"{self.short_line_label(line_name)} {travel}m"

        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        dx = x2 - x1
        dy = y2 - y1
        length = max((dx * dx + dy * dy) ** 0.5, 1)
        offset = max(MAP_EDGE_LABEL_MIN_OFFSET, int(MAP_EDGE_LABEL_OFFSET * scale))
        nx = -dy / length
        ny = dx / length

        lx = mx + nx * offset
        ly = my + ny * offset
        font_scale = MAP_ROUTE_LABEL_FONT_SCALE if is_route_edge else MAP_EDGE_LABEL_FONT_SCALE
        font_size = max(MAP_EDGE_LABEL_MIN_FONT, int(font_scale * scale))
        bg = GUI_COLORS["route_label_bg"] if is_route_edge else GUI_COLORS["map_bg"]
        fg = GUI_COLORS["warning"] if is_route_edge else GUI_COLORS["muted_text"]

        text_id = self.canvas.create_text(
            lx,
            ly,
            text=label,
            font=(GUI_MONO_FONT_FAMILY, font_size, "bold" if is_route_edge else "normal"),
            fill=fg,
        )
        bbox = self.canvas.bbox(text_id)
        if bbox:
            pad = MAP_LABEL_PAD
            rect_id = self.canvas.create_rectangle(
                bbox[0] - pad,
                bbox[1] - pad,
                bbox[2] + pad,
                bbox[3] + pad,
                fill=bg,
                outline=GUI_COLORS["map_border"],
            )
            self.canvas.tag_lower(rect_id, text_id)

    def draw_line_color_index(self):
        """Draw line color index method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        width = max(self.canvas.winfo_width(), MAP_MIN_WIDTH)
        x = width - MAP_LEGEND_RIGHT_OFFSET
        y = MAP_LEGEND_TOP
        row_gap = MAP_LEGEND_ROW_GAP

        self.canvas.create_rectangle(
            x - 10,
            y - 10,
            x + 80,
            y + row_gap * len(LINES) + 6,
            fill=GUI_COLORS["map_bg"],
            outline=GUI_COLORS["table_border"],
        )
        self.canvas.create_text(
            x,
            y,
            text="Lines",
            anchor="w",
            font=(GUI_FONT_FAMILY, GUI_FONT_SMALL - 1, "bold"),
            fill=GUI_COLORS["text"],
        )

        for index, (line_name, data) in enumerate(LINES.items(), start=1):
            cy = y + index * row_gap
            self.canvas.create_oval(
                x,
                cy - 6,
                x + 12,
                cy + 6,
                fill=data["color"],
                outline=GUI_COLORS["map_outline"],
            )
            self.canvas.create_text(
                x + 18,
                cy,
                text=self.line_tab_label(line_name),
                anchor="w",
                font=(GUI_FONT_FAMILY, GUI_FONT_SMALL - 1, "bold"),
                fill=GUI_COLORS["text"],
            )

    def short_line_label(self, line_name):
        """Short line label method for this module's MVC responsibility.
        
        Args:
            line_name: Input value used by `short_line_label`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        parts = line_name.split("-")
        if line_name == "Ring":
            return "R"
        return "".join(part[:1] for part in parts[:2]).upper()

    def line_tab_label(self, line_name):
        """Line tab label method for this module's MVC responsibility.
        
        Args:
            line_name: Input value used by `line_tab_label`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        if line_name == "Ring":
            return "R"
        parts = line_name.split("-")
        suffix = parts[-1] if parts[-1].isdigit() else ""
        prefix = "".join(part[:1] for part in parts[:-1] if part)
        return f"{prefix}{suffix}".upper()

    def route_edge_line(self, a, b):
        """Route edge line method for this module's MVC responsibility.
        
        Args:
            a: Input value used by `route_edge_line`; see the function signature and caller context for the expected type.
            b: Input value used by `route_edge_line`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        for nxt, line, _ in self.adjacent_segments(a):
            if nxt == b:
                return line
        return "?"

    def process_events(self):
        """Process events method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        try:
            while True:
                event = self.event_queue.get_nowait()
                kind = event[0]

                if kind == "message":
                    _, speaker, message = event
                    self.add_message(speaker, message)
                elif kind == "system":
                    _, message = event
                    self.add_system(message)
                elif kind == "warning":
                    _, message = event
                    self.add_warning(message)
                elif kind == "route":
                    _, route = event
                    self.current_route = route
                    self.draw_network()
                    self.update_route_table()
                elif kind == "candidate":
                    _, candidate = event
                    self.add_candidate(candidate)
                elif kind == "telemetry":
                    _, telemetry_type, payload = event
                    self.handle_telemetry(telemetry_type, payload)
                elif kind == "metrics":
                    _, metrics = event
                    self.apply_metrics(metrics)
                elif kind == "done":
                    self.live_stats["finished"] = True
                    self.update_live_dialog_metrics()
                    self.add_system("Finished")

        except queue.Empty:
            pass

        self.update_live_dialog_metrics()
        self.root.after(GUI_REFRESH_MS, self.process_events)

    def add_message(self, speaker, message):
        """Add message method for this module's MVC responsibility.
        
        Args:
            speaker: Input value used by `add_message`; see the function signature and caller context for the expected type.
            message: Input value used by `add_message`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.record_dialog_message(speaker, message)
        self.textbox.insert(tk.END, f"{speaker}: ", speaker)
        self.textbox.insert(tk.END, f"{message}\n")
        self.textbox.see(tk.END)

    def add_system(self, message):
        """Add system method for this module's MVC responsibility.
        
        Args:
            message: Input value used by `add_system`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        key, value = self.parse_system_message(message)
        if key:
            self.set_summary(key, value)

    def add_warning(self, message):
        """Add warning method for this module's MVC responsibility.
        
        Args:
            message: Input value used by `add_warning`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.live_stats["warnings"] += 1
        self.live_stats["pending_warning_recovery"] += 1
        self.update_live_dialog_metrics()

    def handle_telemetry(self, telemetry_type, payload):
        """Route controller telemetry events into stage-specific metric collectors."""
        if telemetry_type == "speech":
            self.record_speech_telemetry(payload)
        elif telemetry_type == "timing":
            self.record_timing_telemetry(payload)
        elif telemetry_type == "nlu":
            self.record_nlu_telemetry(payload)

    def record_speech_telemetry(self, payload):
        """Update ASR-facing metrics from source text versus transcript."""
        source_text = payload.get("source_text", "")
        transcript = payload.get("transcript", "")
        ref_words = self.dialog_words(source_text)
        hyp_words = self.dialog_words(transcript)
        word_subs, word_dels, word_ins = self.edit_counts(ref_words, hyp_words)
        char_subs, char_dels, char_ins = self.edit_counts(list(self.normalized_text(source_text)), list(self.normalized_text(transcript)))
        ref_keywords = self.station_mentions_set(source_text)
        hyp_keywords = self.station_mentions_set(transcript)

        self.live_stats["asr_utterances"] += 1
        self.live_stats["asr_ref_words"] += len(ref_words)
        self.live_stats["asr_ref_chars"] += len(self.normalized_text(source_text))
        self.live_stats["asr_word_substitutions"] += word_subs
        self.live_stats["asr_word_deletions"] += word_dels
        self.live_stats["asr_word_insertions"] += word_ins
        self.live_stats["asr_char_substitutions"] += char_subs
        self.live_stats["asr_char_deletions"] += char_dels
        self.live_stats["asr_char_insertions"] += char_ins
        if self.normalized_text(source_text) != self.normalized_text(transcript):
            self.live_stats["asr_sentence_errors"] += 1
        self.live_stats["asr_keyword_tp"] += len(ref_keywords & hyp_keywords)
        self.live_stats["asr_keyword_fp"] += len(hyp_keywords - ref_keywords)
        self.live_stats["asr_keyword_fn"] += len(ref_keywords - hyp_keywords)

    def record_timing_telemetry(self, payload):
        """Update turn-latency metrics."""
        speaker = payload.get("speaker")
        latency = float(payload.get("turn_latency_sec", 0.0) or 0.0)
        if speaker == "Agent B":
            self.live_stats["agent_b_latency_total"] += latency
            self.live_stats["agent_b_latency_count"] += 1
        elif speaker == "Agent A":
            self.live_stats["agent_a_latency_total"] += latency
            self.live_stats["agent_a_latency_count"] += 1

    def record_nlu_telemetry(self, payload):
        """Update semantic and state-tracking metrics from parsed route attempts."""
        if payload.get("speaker") != "Agent B":
            return

        has_station_mentions = payload.get("has_station_mentions", False)
        if not has_station_mentions:
            return

        parsed_route = payload.get("parsed_route") or []
        route_valid_flag = bool(payload.get("route_valid"))
        route_goal_flag = bool(payload.get("route_reaches_goal"))

        self.live_stats["semantic_attempts"] += 1
        self.live_stats["state_updates"] += 1

        if not parsed_route:
            self.live_stats["semantic_failures"] += 1
            return

        start_hit = 1 if parsed_route and parsed_route[0] == self.scenario["start_station"] else 0
        goal_hit = 1 if parsed_route and parsed_route[-1] == self.scenario["destination_station"] else 0
        self.live_stats["slot_hits"] += start_hit + goal_hit
        self.live_stats["slot_total"] += 2

        if route_valid_flag:
            self.live_stats["valid_state_updates"] += 1
        else:
            self.live_stats["semantic_failures"] += 1

        if route_valid_flag and route_goal_flag:
            self.live_stats["semantic_frame_hits"] += 1

    def apply_metrics(self, metrics):
        """Apply metrics method for this module's MVC responsibility.
        
        Args:
            metrics: Input value used by `apply_metrics`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        parsed = self.parse_metrics(metrics)
        mapping = {
            "Displayed route": "route",
            "Displayed duration": "duration",
            "Duration breakdown": "breakdown",
            "Reference route": "reference_route",
            "Reference duration": "reference_duration",
            "Candidate routes": "candidate_routes",
            "Route revisions": "route_revisions",
            "Best candidate turn": "best_turn",
            "Route valid": "valid",
            "Route reaches goal": "goal",
            "Route correct": "correct",
            "Runtime": "runtime",
        }

        for source_key, metric_key in mapping.items():
            if source_key in parsed:
                self.metric_values[metric_key].configure(text=parsed[source_key])

        if "Displayed route" in parsed and parsed["Displayed route"] != "None":
            route = [station.strip() for station in parsed["Displayed route"].split("->")]
            self.metric_values["route"].configure(text=self.route_label(route))

    def update_route_metric(self):
        """Update route metric method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        route_text = self.route_label(self.current_route) if self.current_route else "No inferred route"
        self.metric_values["route"].configure(text=route_text)
        self.update_live_dialog_metrics()

    def record_dialog_message(self, speaker, message):
        """Record dialog message method for this module's MVC responsibility.
        
        Args:
            speaker: Input value used by `record_dialog_message`; see the function signature and caller context for the expected type.
            message: Input value used by `record_dialog_message`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.live_stats["messages"] += 1
        self.live_stats["last_speaker"] = speaker
        words = self.dialog_words(message)
        if speaker == "Agent A":
            self.live_stats["agent_a_messages"] += 1
            self.live_stats["agent_a_words"] += len(words)
        elif speaker == "Agent B":
            self.live_stats["agent_b_messages"] += 1
            self.live_stats["agent_b_words"] += len(words)

        self.live_stats["words"] += len(words)
        self.live_stats["task_terms"] += sum(1 for word in words if word in TASK_TERMS)
        self.live_stats["comparison_terms"] += sum(1 for word in words if word in COMPARISON_TERMS)
        self.live_stats["cooperation_terms"] += sum(1 for word in words if word in COOPERATION_TERMS)
        self.live_stats["station_mentions"] += self.count_station_mentions(message)
        self.live_stats["questions"] += message.count("?")
        if speaker == "Agent A":
            self.live_stats["agent_a_questions"] += message.count("?")
            prior_agent_a = self.metric_buffers["agent_a_history"][-1] if self.metric_buffers["agent_a_history"] else None
            normalized = self.normalized_text(message)
            if prior_agent_a and normalized:
                similarity = SequenceMatcher(None, prior_agent_a, normalized).ratio()
                if 0.72 <= similarity < 0.999:
                    self.live_stats["reformulations"] += 1
            if normalized:
                self.metric_buffers["agent_a_history"].append(normalized)
        elif speaker == "Agent B":
            self.live_stats["agent_b_questions"] += message.count("?")
            self.metric_buffers["agent_b_tokens"].extend(words)
            self.metric_buffers["agent_b_bigrams"].extend(list(zip(words, words[1:])))
        self.update_live_dialog_metrics()

    def update_live_dialog_metrics(self):
        """Update live dialog metrics method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        if not self.live_metric_values and not self.snapshot_values and not self.stage_metric_values:
            return

        elapsed = time.time() - self.dialog_started_at
        words = self.live_stats["words"]
        task_terms = self.live_stats["task_terms"]
        comparison_terms = self.live_stats["comparison_terms"]
        cooperation_terms = self.live_stats["cooperation_terms"]
        questions = self.live_stats["questions"]
        task_focus = task_terms / words if words else 0.0
        comparison_rate = comparison_terms / words if words else 0.0
        cooperation_rate = cooperation_terms / words if words else 0.0
        station_density = self.live_stats["station_mentions"] / words if words else 0.0
        question_rate = questions / self.live_stats["messages"] if self.live_stats["messages"] else 0.0
        avg_words = words / self.live_stats["messages"] if self.live_stats["messages"] else 0.0
        agent_a_avg = (
            self.live_stats["agent_a_words"] / self.live_stats["agent_a_messages"]
            if self.live_stats["agent_a_messages"]
            else 0.0
        )
        agent_b_avg = (
            self.live_stats["agent_b_words"] / self.live_stats["agent_b_messages"]
            if self.live_stats["agent_b_messages"]
            else 0.0
        )
        state = "Finished" if self.live_stats["finished"] else "Running"
        route_status, live_duration = self.live_route_status()
        best_duration = self.live_stats["best_duration"]
        best_duration_text = f"{best_duration}m" if best_duration is not None else "-"
        route_success = 1.0 if route_status == "Correct" else 0.0
        asr_wer = self.safe_ratio(
            self.live_stats["asr_word_substitutions"] + self.live_stats["asr_word_deletions"] + self.live_stats["asr_word_insertions"],
            self.live_stats["asr_ref_words"],
        )
        asr_cer = self.safe_ratio(
            self.live_stats["asr_char_substitutions"] + self.live_stats["asr_char_deletions"] + self.live_stats["asr_char_insertions"],
            self.live_stats["asr_ref_chars"],
        )
        asr_ser = self.safe_ratio(self.live_stats["asr_sentence_errors"], self.live_stats["asr_utterances"])
        asr_keyword_precision = self.safe_ratio(
            self.live_stats["asr_keyword_tp"],
            self.live_stats["asr_keyword_tp"] + self.live_stats["asr_keyword_fp"],
        )
        asr_keyword_recall = self.safe_ratio(
            self.live_stats["asr_keyword_tp"],
            self.live_stats["asr_keyword_tp"] + self.live_stats["asr_keyword_fn"],
        )
        asr_keyword_f1 = self.f1_score(asr_keyword_precision, asr_keyword_recall)
        slot_accuracy = self.safe_ratio(self.live_stats["slot_hits"], self.live_stats["slot_total"])
        frame_accuracy = self.safe_ratio(self.live_stats["semantic_frame_hits"], self.live_stats["semantic_attempts"])
        semantic_error_rate = self.safe_ratio(self.live_stats["semantic_failures"], self.live_stats["semantic_attempts"])
        state_update_accuracy = self.safe_ratio(self.live_stats["valid_state_updates"], self.live_stats["state_updates"])
        recovery_rate = (
            self.safe_ratio(self.live_stats["recovered_warnings"], self.live_stats["warnings"])
            if self.live_stats["warnings"]
            else 1.0
        )
        agent_b_avg_latency = self.safe_ratio(self.live_stats["agent_b_latency_total"], self.live_stats["agent_b_latency_count"])
        distinct_1 = self.safe_ratio(
            len(set(self.metric_buffers["agent_b_tokens"])),
            len(self.metric_buffers["agent_b_tokens"]),
        )
        distinct_2 = self.safe_ratio(
            len(set(self.metric_buffers["agent_b_bigrams"])),
            len(self.metric_buffers["agent_b_bigrams"]),
        )
        repetition_rate = 1.0 - distinct_1 if self.metric_buffers["agent_b_tokens"] else 0.0
        reformulation_rate = self.safe_ratio(
            self.live_stats["reformulations"],
            max(self.live_stats["agent_a_messages"] - 1, 0),
        )
        warning_pressure = self.safe_ratio(self.live_stats["warnings"], self.live_stats["messages"])
        predicted_satisfaction = self.clamp(
            0.45
            + 0.35 * route_success
            + 0.10 * task_focus
            + 0.05 * distinct_1
            - 0.20 * reformulation_rate
            - 0.15 * asr_wer
            - 0.10 * warning_pressure,
            0.0,
            1.0,
        )
        speech_profile = self.summary_values["speech"].cget("text") if "speech" in self.summary_values else "-"
        robustness_noise = 1.0 - asr_wer

        values = {
            "messages": str(self.live_stats["messages"]),
            "agent_split": f"{self.live_stats['agent_a_messages']} / {self.live_stats['agent_b_messages']}",
            "words": str(words),
            "agent_words": f"{self.live_stats['agent_a_words']} / {self.live_stats['agent_b_words']}",
            "avg_words": f"{avg_words:.1f}",
            "agent_avg_words": f"{agent_a_avg:.1f} / {agent_b_avg:.1f}",
            "questions": str(questions),
            "agent_questions": f"{self.live_stats['agent_a_questions']} / {self.live_stats['agent_b_questions']}",
            "question_rate": f"{question_rate:.2f}/msg",
            "task_focus": f"{task_focus:.2f}",
            "task_terms": str(task_terms),
            "station_mentions": str(self.live_stats["station_mentions"]),
            "station_density": f"{station_density:.2f}",
            "comparison_terms": str(comparison_terms),
            "comparison_rate": f"{comparison_rate:.2f}",
            "cooperation_terms": str(cooperation_terms),
            "cooperation_rate": f"{cooperation_rate:.2f}",
            "last_speaker": self.live_stats.get("last_speaker", "-"),
        }
        for key, value in values.items():
            if key in self.live_metric_values:
                self.live_metric_values[key].configure(text=value)

        snapshot_values = {
            "run_state": state,
            "elapsed": f"{elapsed:.1f}s",
            "route_status": route_status,
            "live_duration": live_duration,
            "best_duration": best_duration_text,
            "candidate_routes": str(self.live_stats["candidate_routes"]),
            "route_revisions": str(self.live_stats["route_revisions"]),
            "question_rate": f"{question_rate:.2f}/msg",
            "messages": str(self.live_stats["messages"]),
            "words": str(words),
            "task_focus": f"{task_focus:.2f}",
            "warnings": str(self.live_stats["warnings"]),
        }
        for key, value in snapshot_values.items():
            if key in self.snapshot_values:
                self.snapshot_values[key].configure(text=value)

        stage_values = {
            "asr_wer": self.format_ratio(asr_wer),
            "asr_cer": self.format_ratio(asr_cer),
            "asr_ser": self.format_ratio(asr_ser),
            "asr_keyword_f1": self.format_ratio(asr_keyword_f1),
            "asr_rtf": "n/a",
            "nlu_slot_accuracy": self.format_ratio(slot_accuracy),
            "nlu_frame_accuracy": self.format_ratio(frame_accuracy),
            "nlu_semantic_error_rate": self.format_ratio(semantic_error_rate),
            "nlu_intent_accuracy": "n/a",
            "dst_joint_goal_accuracy": self.format_ratio(route_success),
            "dst_slot_accuracy": self.format_ratio(slot_accuracy),
            "dst_state_update_accuracy": self.format_ratio(state_update_accuracy),
            "dm_task_success_rate": self.format_ratio(route_success),
            "dm_dialogue_length": str(self.live_stats["messages"]),
            "dm_recovery_rate": self.format_ratio(recovery_rate),
            "dm_avg_latency": self.format_seconds(agent_b_avg_latency),
            "nlg_distinct_1": self.format_ratio(distinct_1),
            "nlg_distinct_2": self.format_ratio(distinct_2),
            "nlg_repetition_rate": self.format_ratio(repetition_rate),
            "nlg_avg_words": f"{agent_b_avg:.1f}",
            "tts_mos": "n/a",
            "tts_predicted_mos": "n/a",
            "tts_intelligibility": "n/a",
            "tts_speaker_similarity": "n/a",
            "e2e_task_completion_rate": self.format_ratio(route_success),
            "e2e_dialogue_success": self.format_ratio(route_success),
            "e2e_latency": self.format_seconds(elapsed),
            "e2e_turns_per_dialogue": str(self.live_stats["messages"]),
            "e2e_error_recovery": self.format_ratio(recovery_rate),
            "ux_reformulation_rate": self.format_ratio(reformulation_rate),
            "ux_abandonment_rate": "n/a",
            "ux_predicted_satisfaction": self.format_ratio(predicted_satisfaction),
            "ux_warning_pressure": self.format_ratio(warning_pressure),
            "robustness_noise": self.format_ratio(robustness_noise),
            "robustness_asr_degradation": self.format_ratio(asr_wer),
            "robustness_speech_pattern": speech_profile,
            "robustness_bias": "n/a",
        }
        for key, value in stage_values.items():
            if key in self.stage_metric_values:
                self.stage_metric_values[key].configure(text=value)

        self.set_summary("elapsed", f"{elapsed:.1f}s")
        self.set_summary("warnings", str(self.live_stats["warnings"]))
        self.set_summary("run_state", state)

    def add_candidate(self, candidate):
        """Add one inferred route candidate to the comparison table."""
        if not hasattr(self, "candidate_table"):
            return

        previous_best = candidate.get("previous_best")
        duration = candidate["duration"]
        if previous_best is None:
            delta = "-"
        else:
            diff = duration - previous_best
            delta = f"{diff:+d}m"

        if candidate["decision"] == "repeat":
            self.candidate_table.insert(
                "",
                tk.END,
                values=(
                    candidate["turn"],
                    f"{duration}m",
                    delta,
                    "repeat",
                    self.route_label(candidate["route"]),
                ),
            )
            self.update_live_dialog_metrics()
            return

        if candidate["decision"] == "improved":
            self.live_stats["route_revisions"] += 1
        self.live_stats["candidate_routes"] += 1
        self.live_stats["best_duration"] = candidate.get("best_duration")
        if self.live_stats["pending_warning_recovery"]:
            self.live_stats["recovered_warnings"] += self.live_stats["pending_warning_recovery"]
            self.live_stats["pending_warning_recovery"] = 0

        self.candidate_table.insert(
            "",
            tk.END,
            values=(
                candidate["turn"],
                f"{duration}m",
                delta,
                candidate["decision"],
                self.route_label(candidate["route"]),
            ),
        )
        self.update_live_dialog_metrics()

    def live_route_status(self):
        """Live route status method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        if len(self.current_route) < 2:
            return "No route", "-"

        valid = route_is_valid(self.current_route)
        reaches_goal = self.route_reaches_goal(self.current_route)
        status = "Correct" if valid and reaches_goal else "Partial" if valid else "Invalid"
        estimate = estimate_route_time(
            self.current_route,
            self.scenario["start_time_min"],
            self.scenario["transfer_time_min"],
        )
        if not estimate:
            return status, "-"

        arrival, _ = estimate
        duration = arrival - self.scenario["start_time_min"]
        return status, f"{duration}m"

    def route_reaches_goal(self, route):
        """Route reaches goal method for this module's MVC responsibility.
        
        Args:
            route: Input value used by `route_reaches_goal`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return (
            bool(route)
            and route[0] == self.scenario["start_station"]
            and route[-1] == self.scenario["destination_station"]
        )

    @staticmethod
    def clamp(value, lower, upper):
        """Clamp a numeric value to a closed interval."""
        return max(lower, min(upper, value))

    @staticmethod
    def safe_ratio(numerator, denominator):
        """Return a stable ratio for GUI metrics."""
        return numerator / denominator if denominator else 0.0

    @staticmethod
    def f1_score(precision, recall):
        """Compute F1 from precision and recall."""
        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    @staticmethod
    def format_ratio(value):
        """Format a ratio as a compact percentage."""
        return f"{value * 100:.1f}%"

    @staticmethod
    def format_seconds(value):
        """Format seconds compactly for dashboard display."""
        return f"{value:.2f}s"

    @staticmethod
    def normalized_text(message):
        """Normalize text for compact metric comparisons."""
        return " ".join(re.findall(r"[a-z0-9]+", message.lower()))

    @classmethod
    def edit_counts(cls, reference_tokens, hypothesis_tokens):
        """Return substitution, deletion, and insertion counts via Levenshtein alignment."""
        rows = len(reference_tokens) + 1
        cols = len(hypothesis_tokens) + 1
        dp = [[(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)]

        for i in range(1, rows):
            cost, subs, dels, ins = dp[i - 1][0]
            dp[i][0] = (cost + 1, subs, dels + 1, ins)
        for j in range(1, cols):
            cost, subs, dels, ins = dp[0][j - 1]
            dp[0][j] = (cost + 1, subs, dels, ins + 1)

        for i in range(1, rows):
            for j in range(1, cols):
                if reference_tokens[i - 1] == hypothesis_tokens[j - 1]:
                    candidates = [dp[i - 1][j - 1]]
                else:
                    cost, subs, dels, ins = dp[i - 1][j - 1]
                    candidates = [(cost + 1, subs + 1, dels, ins)]

                cost, subs, dels, ins = dp[i - 1][j]
                candidates.append((cost + 1, subs, dels + 1, ins))
                cost, subs, dels, ins = dp[i][j - 1]
                candidates.append((cost + 1, subs, dels, ins + 1))
                dp[i][j] = min(candidates, key=lambda item: (item[0], item[1] + item[2] + item[3], item[1], item[2], item[3]))

        _, substitutions, deletions, insertions = dp[-1][-1]
        return substitutions, deletions, insertions

    @staticmethod
    def station_mentions_set(message):
        """Return the set of station names mentioned in a message."""
        mentions = set()
        for station in STATION_POS:
            if re.search(rf"\b{re.escape(station)}\b", message, flags=re.IGNORECASE):
                mentions.add(station)
        return mentions

    @staticmethod
    def dialog_words(message):
        """Dialog words method for this module's MVC responsibility.
        
        Args:
            message: Input value used by `dialog_words`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return [
            word
            for word in re.findall(r"[A-Za-z]+", message.lower())
            if word
        ]

    @staticmethod
    def count_station_mentions(message):
        """Count station mentions method for this module's MVC responsibility.
        
        Args:
            message: Input value used by `count_station_mentions`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        count = 0
        for station in STATION_POS:
            count += len(re.findall(rf"\b{re.escape(station)}\b", message, flags=re.IGNORECASE))
        return count

    def update_route_table(self):
        """Update route table method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        if not hasattr(self, "route_table"):
            return

        for item in self.route_table.get_children():
            self.route_table.delete(item)

        if len(self.current_route) < 2:
            return

        estimate = estimate_route_time(
            self.current_route,
            self.scenario["start_time_min"],
            self.scenario["transfer_time_min"],
        )
        if not estimate:
            return

        _, steps = estimate
        for step in steps:
            self.route_table.insert(
                "",
                tk.END,
                values=(
                    self.station_label(step["from"]),
                    self.station_label(step["to"]),
                    self.short_line_label(step["line"]),
                    f"{step.get('fullness', 0)}%",
                    fmt_time(step["depart"]),
                    fmt_time(step["arrive"]),
                    f"{step['travel']}m",
                    f"{step['wait']}m",
                    f"{step['transfer']}m",
                ),
            )

    def parse_metrics(self, metrics):
        """Parse metrics method for this module's MVC responsibility.
        
        Args:
            metrics: Input value used by `parse_metrics`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        parsed = {}
        for line in metrics.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
        return parsed

    def parse_system_message(self, message):
        """Parse system message method for this module's MVC responsibility.
        
        Args:
            message: Input value used by `parse_system_message`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        mapping = {
            "Test case": "test_case",
            "Persona": "persona",
            "Scenario": "scenario",
            "Speech transport": "speech",
            "Model": "model",
            "Device": "device",
            "Agent A": "agent_a",
        }
        if message.strip().startswith("Turns="):
            return "settings", message.strip()
        if ":" not in message:
            return None, None

        key, value = message.split(":", 1)
        return mapping.get(key.strip()), value.strip()

    def set_summary(self, key, value):
        """Set summary method for this module's MVC responsibility.
        
        Args:
            key: Input value used by `set_summary`; see the function signature and caller context for the expected type.
            value: Input value used by `set_summary`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        if key in self.summary_values:
            self.summary_values[key].configure(text=value)

    def label_for(self, key):
        """Label for method for this module's MVC responsibility.
        
        Args:
            key: Input value used by `label_for`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        labels = {
            "test_case": "Test",
            "persona": "Persona",
            "scenario": "Scenario",
            "speech": "Speech",
            "route": "Route",
            "duration": "Duration",
            "reference_duration": "Reference",
            "reference_route": "Ref Route",
            "best_turn": "Best Turn",
            "candidate_routes": "Candidates",
            "route_revisions": "Revisions",
            "breakdown": "Breakdown",
            "valid": "Valid",
            "goal": "Goal",
            "correct": "Correct",
            "runtime": "Runtime",
        }
        return labels.get(key, key)

    def run(self):
        """Run method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.root.mainloop()
