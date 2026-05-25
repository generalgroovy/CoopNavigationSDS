"""View layer for the interactive experiment conversation transcript."""
import queue
import re
import tkinter as tk
import time
from difflib import SequenceMatcher
from tkinter import ttk

from minillama.view.config import (
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
    GUI_TEXTBOX_HEIGHT,
    GUI_STATION_TABS_HEIGHT,
    GUI_DEFAULT_TABS_HEIGHT,
    GUI_COLORS,
    GUI_ROUTE_TABLE_COLUMNS,
    GUI_LINE_TABLE_COLUMNS,
    GUI_STATION_LINE_TABLE_COLUMNS,
    GUI_STATION_TIME_TABLE_COLUMNS,
    GUI_NETWORK_LINE_COLUMNS,
    GUI_NETWORK_STATION_COLUMNS,
    GUI_ROUTE_TABLE_HEIGHT,
    GUI_LINE_TABLE_HEIGHT,
    GUI_STATION_TABLE_HEIGHT,
    GUI_NETWORK_TABLE_HEIGHT,
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
from minillama.evaluation.metrics import COMPARISON_TERMS, COOPERATION_TERMS, METRIC_FAMILY_SPECS, TASK_TERMS
from minillama.model.metro_data import (
    ADJACENCY,
    LINES,
    STATION_POS,
    line_fullness_percent,
    line_stop_pairs,
    station_fullness_percent,
)
from minillama.model.network_overview import build_network_overview
from minillama.model.route_planner import (
    estimate_route_time,
    fmt_time,
    line_direction_sequences,
    route_line_sequence,
    route_is_valid,
    segment_travel,
)

SECTION_PAD_X = 8
SECTION_PAD_Y = 8
SECTION_HEADER_PAD_X_LOCAL = 10
SECTION_HEADER_PAD_Y_LOCAL = (8, 4)
SECTION_CONTENT_PAD_Y_LOCAL = (0, 8)
BODY_FONT_SIZE = GUI_FONT_NORMAL + 1
SECTION_TITLE_SIZE = GUI_FONT_SECTION + 1
TABLE_FONT_SIZE = GUI_TABLE_FONT_SIZE + 1
TABLE_HEADER_SIZE = GUI_TABLE_HEADER_FONT_SIZE + 1
TABLE_ROW_HEIGHT_LOCAL = GUI_TABLE_ROW_HEIGHT + 4
TRANSCRIPT_FONT_SIZE = GUI_FONT_NORMAL + 2
TRANSCRIPT_SPACING = 6
SUMMARY_WRAP = 260
METRIC_WRAP = 280
REFERENCE_LIST_FONT = GUI_FONT_NORMAL
REFERENCE_DETAIL_FONT = GUI_FONT_NORMAL


class NotebookTabs(tk.Frame):
    """Small notebook wrapper with the add/tab API used by the existing view code."""

    def __init__(self, parent):
        super().__init__(parent, bg=GUI_COLORS["app_bg"], bd=0, highlightthickness=0)
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill=tk.BOTH, expand=True)
        self._tabs = {}

    def add(self, title):
        frame = tk.Frame(self._notebook, bg=GUI_COLORS["app_bg"], bd=0, highlightthickness=0)
        self._tabs[title] = frame
        self._notebook.add(frame, text=title)
        return frame

    def tab(self, title):
        return self._tabs[title]


class StartupConfigDialog:
    """Compact startup form for interactive run configuration."""

    def __init__(self, choices, defaults):
        self.choices = choices
        self.defaults = defaults
        self.result = None
        self.root = tk.Tk()
        self.root.title("MiniLlama Run Configuration")
        self.root.configure(bg=GUI_COLORS["app_bg"])
        self.root.resizable(False, False)
        self.vars = {
            "test_case_key": tk.StringVar(value=defaults["test_case_key"]),
            "agent_b_plugin": tk.StringVar(value=defaults["agent_b_plugin"]),
            "speech_pattern_key": tk.StringVar(value=defaults["speech_pattern_key"]),
            "speech_engine": tk.StringVar(value=defaults["speech_engine"]),
            "tts_engine": tk.StringVar(value=defaults.get("tts_engine", defaults["speech_engine"])),
            "asr_engine": tk.StringVar(value=defaults.get("asr_engine", defaults["speech_engine"])),
            "speech_audio_dir": tk.StringVar(value=defaults["speech_audio_dir"]),
            "speech_scope": tk.StringVar(value=defaults["speech_scope"]),
            "speech_incoming_enabled": tk.BooleanVar(value=defaults["speech_incoming_enabled"]),
            "speech_outgoing_enabled": tk.BooleanVar(value=defaults["speech_outgoing_enabled"]),
            "speech_playback_enabled": tk.BooleanVar(value=defaults.get("speech_playback_enabled", False)),
            "speech_realtime_enabled": tk.BooleanVar(value=defaults.get("speech_realtime_enabled", False)),
            "gui_enabled": tk.BooleanVar(value=defaults.get("gui_enabled", True)),
        }
        self.build()
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)

    def build(self):
        frame = tk.Frame(self.root, bg=GUI_COLORS["panel_bg"], bd=1, relief="solid")
        frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        frame.grid_columnconfigure(1, weight=1)

        rows = [
            ("Test case", "test_case_key", self.choices["test_case_keys"]),
            ("Agent B", "agent_b_plugin", self.choices["agent_b_plugins"]),
            ("Speech pattern", "speech_pattern_key", self.choices["speech_patterns"]),
            ("Speech engine", "speech_engine", self.choices["speech_engines"]),
            ("TTS engine", "tts_engine", self.choices["tts_engines"]),
            ("ASR engine", "asr_engine", self.choices["asr_engines"]),
            ("Speech scope", "speech_scope", self.choices["speech_scopes"]),
        ]
        for row, (label, key, values) in enumerate(rows):
            tk.Label(
                frame,
                text=label,
                anchor="w",
                font=(GUI_FONT_FAMILY, GUI_FONT_NORMAL),
                bg=GUI_COLORS["panel_bg"],
                fg=GUI_COLORS["muted_text"],
            ).grid(row=row, column=0, sticky="w", padx=(10, 8), pady=(8, 0))
            combo = ttk.Combobox(
                frame,
                textvariable=self.vars[key],
                values=values,
                state="normal" if key == "agent_b_plugin" else "readonly",
                width=36,
            )
            combo.grid(row=row, column=1, sticky="ew", padx=(0, 10), pady=(8, 0))

        audio_row = len(rows)
        tk.Label(
            frame,
            text="Speech files",
            anchor="w",
            font=(GUI_FONT_FAMILY, GUI_FONT_NORMAL),
            bg=GUI_COLORS["panel_bg"],
            fg=GUI_COLORS["muted_text"],
        ).grid(row=audio_row, column=0, sticky="w", padx=(10, 8), pady=(8, 0))
        tk.Entry(
            frame,
            textvariable=self.vars["speech_audio_dir"],
            width=36,
        ).grid(row=audio_row, column=1, sticky="ew", padx=(0, 10), pady=(8, 0))

        toggle_row = audio_row + 1
        tk.Checkbutton(
            frame,
            text="Incoming ASR",
            variable=self.vars["speech_incoming_enabled"],
            bg=GUI_COLORS["panel_bg"],
            fg=GUI_COLORS["text"],
            selectcolor=GUI_COLORS["tab_bg"],
            activebackground=GUI_COLORS["panel_bg"],
            activeforeground=GUI_COLORS["text"],
        ).grid(row=toggle_row, column=0, sticky="w", padx=(10, 8), pady=(10, 0))
        tk.Checkbutton(
            frame,
            text="Outgoing TTS",
            variable=self.vars["speech_outgoing_enabled"],
            bg=GUI_COLORS["panel_bg"],
            fg=GUI_COLORS["text"],
            selectcolor=GUI_COLORS["tab_bg"],
            activebackground=GUI_COLORS["panel_bg"],
            activeforeground=GUI_COLORS["text"],
        ).grid(row=toggle_row, column=1, sticky="w", padx=(0, 10), pady=(10, 0))

        gui_row = toggle_row + 1
        tk.Checkbutton(
            frame,
            text="Play generated audio",
            variable=self.vars["speech_playback_enabled"],
            bg=GUI_COLORS["panel_bg"],
            fg=GUI_COLORS["text"],
            selectcolor=GUI_COLORS["tab_bg"],
            activebackground=GUI_COLORS["panel_bg"],
            activeforeground=GUI_COLORS["text"],
        ).grid(row=gui_row, column=0, sticky="w", padx=(10, 8), pady=(10, 0))
        tk.Checkbutton(
            frame,
            text="Real-time listening",
            variable=self.vars["speech_realtime_enabled"],
            bg=GUI_COLORS["panel_bg"],
            fg=GUI_COLORS["text"],
            selectcolor=GUI_COLORS["tab_bg"],
            activebackground=GUI_COLORS["panel_bg"],
            activeforeground=GUI_COLORS["text"],
        ).grid(row=gui_row, column=1, sticky="w", padx=(0, 10), pady=(10, 0))

        gui_row += 1
        tk.Checkbutton(
            frame,
            text="Conversation GUI",
            variable=self.vars["gui_enabled"],
            bg=GUI_COLORS["panel_bg"],
            fg=GUI_COLORS["text"],
            selectcolor=GUI_COLORS["tab_bg"],
            activebackground=GUI_COLORS["panel_bg"],
            activeforeground=GUI_COLORS["text"],
        ).grid(row=gui_row, column=0, columnspan=2, sticky="w", padx=(10, 8), pady=(10, 0))

        button_row = gui_row + 1
        buttons = tk.Frame(frame, bg=GUI_COLORS["panel_bg"])
        buttons.grid(row=button_row, column=0, columnspan=2, sticky="e", padx=10, pady=12)
        tk.Button(
            buttons,
            text="Start",
            command=self.start,
            bg=GUI_COLORS["tab_selected"],
            fg="#ffffff",
            activebackground=GUI_COLORS["tab_selected"],
            activeforeground="#ffffff",
            bd=0,
            padx=14,
            pady=5,
        ).grid(row=0, column=0, padx=(0, 8))
        tk.Button(
            buttons,
            text="Cancel",
            command=self.cancel,
            bg=GUI_COLORS["tab_button_bg"],
            fg=GUI_COLORS["text"],
            activebackground=GUI_COLORS["tab_unselected_hover"],
            activeforeground=GUI_COLORS["text"],
            bd=0,
            padx=14,
            pady=5,
        ).grid(row=0, column=1)

    def start(self):
        self.result = {
            key: var.get()
            for key, var in self.vars.items()
        }
        self.root.destroy()

    def cancel(self):
        self.result = None
        self.root.destroy()

    def show(self):
        self.root.mainloop()
        return self.result


def make_scrollable_frame(parent, bg, padx=0, pady=0, sticky="nsew"):
    """Create a vertically scrollable frame and return the content container."""
    container = tk.Frame(parent, bg=bg, bd=0, highlightthickness=0)
    container.grid(sticky=sticky, padx=padx, pady=pady)
    container.grid_columnconfigure(0, weight=1)
    container.grid_rowconfigure(0, weight=1)

    canvas = tk.Canvas(container, bg=bg, bd=0, highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")

    content = tk.Frame(canvas, bg=bg, bd=0, highlightthickness=0)
    window_id = canvas.create_window((0, 0), window=content, anchor="nw")

    def _sync_scrollregion(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _sync_width(event):
        canvas.itemconfigure(window_id, width=event.width)

    content.bind("<Configure>", _sync_scrollregion)
    canvas.bind("<Configure>", _sync_width)
    content._scrollable_container = container
    content._scrollable_canvas = canvas
    return content


class DialogWindow:
    """GUI view for one live dialog experiment conversation."""

    def __init__(self, event_queue, scenario, minimal=True):
        """  init   method for this module's MVC responsibility.

        Args:
            event_queue: Input value used by `__init__`; see the function signature and caller context for the expected type.
            scenario: Input value used by `__init__`; see the function signature and caller context for the expected type.

        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.event_queue = event_queue
        self.scenario = scenario
        self.minimal = minimal
        self.current_route = []
        self.snapshot_values = {}
        self.summary_values = {}
        self.metric_values = {}
        self.live_metric_values = {}
        self.stage_metric_values = {}
        self.metric_windows = {}
        self.latest_metrics_text = ""
        self.pending_speech_traces = {}
        self.section_widgets = {}
        self.section_visibility = {}
        self.section_toggle_buttons = {}
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
            "tts_ref_words": 0,
            "tts_word_substitutions": 0,
            "tts_word_deletions": 0,
            "tts_word_insertions": 0,
            "tts_utterances": 0,
            "speech_incoming_enabled_count": 0,
            "speech_outgoing_enabled_count": 0,
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
            "first_agent_a_generation_sec": None,
            "first_agent_a_audio_sec": None,
            "first_agent_b_generation_sec": None,
            "first_agent_b_audio_sec": None,
            "reformulations": 0,
            "finished": False,
        }

        self.root = tk.Tk()
        self.root.title("MiniLlama Conversation")
        self.root.geometry("960x720" if self.minimal else f"{GUI_WIDTH}x{GUI_HEIGHT}")
        self.root.minsize(620, 420)
        self.root.configure(bg=GUI_COLORS["app_bg"])
        if not self.minimal:
            self.maximize_startup_window()

        self.configure_style()
        if self.minimal:
            self.build_minimal_metric_menu()
        self.build_layout()
        self.update_live_dialog_metrics()
        self.draw_network()

        if hasattr(self, "canvas"):
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
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda _: self.root.attributes("-fullscreen", False))
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
        style.configure("TNotebook", background=GUI_COLORS["app_bg"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(10, 5), background=GUI_COLORS["tab_button_bg"], foreground=GUI_COLORS["text"])
        style.map("TNotebook.Tab", background=[("selected", GUI_COLORS["tab_selected"]), ("active", GUI_COLORS["tab_unselected_hover"])], foreground=[("selected", "#ffffff"), ("active", GUI_COLORS["text"])])
        style.configure(
            "Data.Treeview",
            background=GUI_COLORS["table_bg"],
            fieldbackground=GUI_COLORS["table_bg"],
            foreground=GUI_COLORS["text"],
            bordercolor=GUI_COLORS["table_border"],
            lightcolor=GUI_COLORS["table_border"],
            darkcolor=GUI_COLORS["table_border"],
            font=(GUI_MONO_FONT_FAMILY, TABLE_FONT_SIZE),
            rowheight=TABLE_ROW_HEIGHT_LOCAL,
        )
        style.configure(
            "Data.Treeview.Heading",
            background=GUI_COLORS["table_heading_bg"],
            foreground=GUI_COLORS["text"],
            bordercolor=GUI_COLORS["table_border"],
            font=(GUI_FONT_FAMILY, TABLE_HEADER_SIZE, "bold"),
        )
        style.map("Data.Treeview", background=[("selected", GUI_COLORS["table_selected"])])

    def build_layout(self):
        """Build the conversation-only workspace layout."""
        self.main = tk.Frame(self.root, bg=GUI_COLORS["app_bg"], bd=0, highlightthickness=0)
        self.main.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(0, weight=1)

        if self.minimal:
            self.build_plain_conversation(self.main)
            return

        self.workspace_tabs = self.make_tabs(self.main, height=GUI_HEIGHT - (GUI_MAIN_PAD * 2))
        self.workspace_tabs.add("Metric Data")
        self.workspace_tabs.add("Network Data")

        self.build_metric_data_tab(self.workspace_tabs.tab("Metric Data"))
        self.build_network_data_tab(self.workspace_tabs.tab("Network Data"))

    def build_page_scroller(self, parent):
        """Build a scrollable page shell that lets cards fill the available width."""
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        scroller = make_scrollable_frame(parent, GUI_COLORS["app_bg"], padx=0, pady=0)
        scroller.grid_columnconfigure(0, weight=1)
        return scroller

    def build_conversation_card(self, parent, row=0, text_height=None):
        """Build the conversation transcript card."""
        self.transcript_frame = self.make_section(
            parent,
            "Conversation",
            row=row,
            sticky="nsew",
            key="transcript",
        )
        self.transcript_frame.grid_rowconfigure(0, weight=1)
        self.transcript_frame.grid_columnconfigure(0, weight=1)
        self.transcript_frame.grid_columnconfigure(1, weight=0)

        self.textbox = tk.Text(
            self.transcript_frame,
            wrap=tk.WORD,
            font=(GUI_FONT_FAMILY, TRANSCRIPT_FONT_SIZE),
            height=text_height or GUI_TEXTBOX_HEIGHT,
            bd=0,
            highlightthickness=1,
            highlightbackground=GUI_COLORS["table_border"],
            bg=GUI_COLORS["tab_bg"],
            fg=GUI_COLORS["text"],
            insertbackground=GUI_COLORS["text"],
        )
        transcript_scrollbar = ttk.Scrollbar(self.transcript_frame, orient="vertical", command=self.textbox.yview)
        self.textbox.configure(yscrollcommand=transcript_scrollbar.set)
        self.textbox.grid(row=0, column=0, sticky="nsew")
        transcript_scrollbar.grid(row=0, column=1, sticky="ns")
        self.textbox.tag_config("Agent A", foreground=GUI_COLORS["agent_a"], font=(GUI_FONT_FAMILY, TRANSCRIPT_FONT_SIZE, "bold"), spacing1=TRANSCRIPT_SPACING)
        self.textbox.tag_config("Agent B", foreground=GUI_COLORS["agent_b"], font=(GUI_FONT_FAMILY, TRANSCRIPT_FONT_SIZE, "bold"), spacing1=TRANSCRIPT_SPACING)
        self.textbox.tag_config("Agent A body", foreground=GUI_COLORS["text"], lmargin1=16, lmargin2=16, spacing3=TRANSCRIPT_SPACING)
        self.textbox.tag_config("Agent B body", foreground=GUI_COLORS["text"], lmargin1=16, lmargin2=16, spacing3=TRANSCRIPT_SPACING)
        self.textbox.tag_config("speech_label", foreground=GUI_COLORS["muted_text"], lmargin1=16, lmargin2=16, font=(GUI_FONT_FAMILY, GUI_FONT_SMALL, "bold"))
        self.textbox.tag_config("spoken_text", foreground=GUI_COLORS["subtle_text"], lmargin1=16, lmargin2=16)
        self.textbox.tag_config("understood_text", foreground=GUI_COLORS["text"], lmargin1=16, lmargin2=16, font=(GUI_FONT_FAMILY, TRANSCRIPT_FONT_SIZE, "bold"), spacing3=TRANSCRIPT_SPACING)
        self.textbox.tag_config("system", foreground=GUI_COLORS["subtle_text"])
        self.textbox.tag_config("warning", foreground=GUI_COLORS["warning"])

    def build_plain_conversation(self, parent):
        """Build the minimal conversation-only transcript surface."""
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=0)
        parent.grid_rowconfigure(0, weight=1)
        self.textbox = tk.Text(
            parent,
            wrap=tk.WORD,
            font=(GUI_FONT_FAMILY, TRANSCRIPT_FONT_SIZE),
            bd=0,
            highlightthickness=1,
            highlightbackground=GUI_COLORS["table_border"],
            bg=GUI_COLORS["tab_bg"],
            fg=GUI_COLORS["text"],
            insertbackground=GUI_COLORS["text"],
        )
        transcript_scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.textbox.yview)
        self.textbox.configure(yscrollcommand=transcript_scrollbar.set)
        self.textbox.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        transcript_scrollbar.grid(row=0, column=1, sticky="ns", pady=6)
        self.textbox.tag_config("Agent A", foreground=GUI_COLORS["agent_a"], font=(GUI_FONT_FAMILY, TRANSCRIPT_FONT_SIZE, "bold"), spacing1=TRANSCRIPT_SPACING)
        self.textbox.tag_config("Agent B", foreground=GUI_COLORS["agent_b"], font=(GUI_FONT_FAMILY, TRANSCRIPT_FONT_SIZE, "bold"), spacing1=TRANSCRIPT_SPACING)
        self.textbox.tag_config("Agent A body", foreground=GUI_COLORS["text"], lmargin1=16, lmargin2=16, spacing3=TRANSCRIPT_SPACING)
        self.textbox.tag_config("Agent B body", foreground=GUI_COLORS["text"], lmargin1=16, lmargin2=16, spacing3=TRANSCRIPT_SPACING)
        self.textbox.tag_config("speech_label", foreground=GUI_COLORS["muted_text"], lmargin1=16, lmargin2=16, font=(GUI_FONT_FAMILY, GUI_FONT_SMALL, "bold"))
        self.textbox.tag_config("spoken_text", foreground=GUI_COLORS["subtle_text"], lmargin1=16, lmargin2=16)
        self.textbox.tag_config("understood_text", foreground=GUI_COLORS["text"], lmargin1=16, lmargin2=16, font=(GUI_FONT_FAMILY, TRANSCRIPT_FONT_SIZE, "bold"), spacing3=TRANSCRIPT_SPACING)

    def build_minimal_metric_menu(self):
        """Create lazy metric windows ordered by the dialog-system pipeline."""
        menubar = tk.Menu(self.root)
        metric_menu = tk.Menu(menubar, tearoff=False)
        metric_menu.add_command(
            label="Conversation Metrics",
            command=lambda: self.toggle_metric_window("conversation_metrics", "Conversation Metrics", self.build_conversation_metrics_card),
        )
        metric_menu.add_command(
            label="Route Outcome Metrics",
            command=lambda: self.toggle_outcome_metric_window(),
        )
        metric_menu.add_separator()
        for index, family in enumerate(METRIC_FAMILY_SPECS, start=1):
            metric_menu.add_command(
                label=f"{index}. {family['title']}",
                command=lambda selected=family: self.toggle_pipeline_metric_window(selected),
            )
        menubar.add_cascade(label="Metrics", menu=metric_menu)
        self.root.configure(menu=menubar)

    def toggle_metric_window(self, key, title, builder):
        """Open or close a lazy metric window."""
        existing = self.metric_windows.get(key)
        if existing is not None and existing.winfo_exists():
            existing.destroy()
            self.metric_windows.pop(key, None)
            return

        window = tk.Toplevel(self.root)
        window.title(title)
        window.configure(bg=GUI_COLORS["app_bg"])
        window.geometry("520x360")
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(0, weight=1)
        window.protocol("WM_DELETE_WINDOW", lambda window_key=key: self.close_metric_window(window_key))
        builder(window, row=0)
        self.metric_windows[key] = window
        self.update_live_dialog_metrics()

    def toggle_outcome_metric_window(self):
        self.toggle_metric_window("route_outcome_metrics", "Route Outcome Metrics", self.build_outcome_metrics_card)
        if self.latest_metrics_text:
            self.apply_metrics(self.latest_metrics_text)

    def toggle_pipeline_metric_window(self, family):
        key = f"pipeline_{self.section_slug(family['title'])}"
        existing = self.metric_windows.get(key)
        if existing is not None and existing.winfo_exists():
            existing.destroy()
            self.metric_windows.pop(key, None)
            return

        window = tk.Toplevel(self.root)
        window.title(family["title"])
        window.configure(bg=GUI_COLORS["panel_bg"])
        window.geometry("420x320")
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(0, weight=1)
        window.protocol("WM_DELETE_WINDOW", lambda window_key=key: self.close_metric_window(window_key))
        self.build_stage_metric_family_group(window, family, row=0, column=0)
        self.metric_windows[key] = window
        self.update_live_dialog_metrics()

    def close_metric_window(self, key):
        window = self.metric_windows.pop(key, None)
        if window is not None and window.winfo_exists():
            window.destroy()

    def build_route_candidates_card(self, parent, row=0):
        """Build the route comparison card."""
        self.route_frame = self.make_section(
            parent,
            "Route Candidate Comparison",
            row=row,
            sticky="nsew",
            pady=0,
            key="route",
        )
        self.route_frame.grid_columnconfigure(0, weight=1)
        self.route_frame.grid_rowconfigure(0, weight=1)
        self.route_frame.grid_rowconfigure(1, weight=1)

        self.candidate_table = ttk.Treeview(
            self.route_frame,
            columns=("turn", "duration", "delta", "gap", "decision", "route", "lines"),
            show="headings",
            height=4,
            style="Data.Treeview",
        )
        candidate_columns = [
            ("turn", "Turn", 42, "center", False),
            ("duration", "Time", 54, "center", False),
            ("delta", "Delta", 54, "center", False),
            ("gap", "Target Gap", 78, "center", False),
            ("decision", "Result", 72, "center", False),
            ("route", "Candidate route", 190, "w", True),
            ("lines", "Lines", 128, "w", True),
        ]
        for col, label, width, anchor, stretch in candidate_columns:
            self.candidate_table.heading(col, text=label)
            self.candidate_table.column(col, width=width, minwidth=width, anchor=anchor, stretch=stretch)
        self.candidate_table.grid(row=0, column=0, sticky="nsew", pady=(0, 2))

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

        self.route_table.grid(row=1, column=0, sticky="nsew")

    def build_metric_data_tab(self, parent):
        """Build metric data as a left transcript and right metric workspace."""
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        split = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        split.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        conversation_shell = tk.Frame(split, bg=GUI_COLORS["app_bg"], bd=0, highlightthickness=0)
        conversation_shell.grid_rowconfigure(0, weight=1)
        conversation_shell.grid_columnconfigure(0, weight=1)
        conversation_page = make_scrollable_frame(conversation_shell, GUI_COLORS["app_bg"], padx=0, pady=0)
        conversation_page.grid_columnconfigure(0, weight=1)
        self.build_conversation_card(conversation_page, row=0, text_height=32)
        split.add(conversation_shell, weight=1)

        metric_shell = tk.Frame(split, bg=GUI_COLORS["app_bg"], bd=0, highlightthickness=0)
        metric_shell.grid_rowconfigure(0, weight=1)
        metric_shell.grid_columnconfigure(0, weight=1)
        split.add(metric_shell, weight=1)

        metric_split = ttk.PanedWindow(metric_shell, orient=tk.VERTICAL)
        metric_split.grid(row=0, column=0, sticky="nsew")

        summary_zone = tk.Frame(metric_split, bg=GUI_COLORS["app_bg"], bd=0, highlightthickness=0)
        summary_zone.grid_columnconfigure(0, weight=1, uniform="metric_summary")
        summary_zone.grid_columnconfigure(1, weight=1, uniform="metric_summary")
        summary_zone.grid_rowconfigure(0, weight=1, uniform="metric_summary")
        summary_zone.grid_rowconfigure(1, weight=1, uniform="metric_summary")
        self.build_snapshot_card(summary_zone, row=0, column=0)
        self.build_run_context_card(summary_zone, row=0, column=1)
        self.build_outcome_metrics_card(summary_zone, row=1, column=0)
        self.build_conversation_metrics_card(summary_zone, row=1, column=1)
        metric_split.add(summary_zone, weight=1)

        phase_zone = tk.Frame(metric_split, bg=GUI_COLORS["app_bg"], bd=0, highlightthickness=0)
        phase_zone.grid_columnconfigure(0, weight=1)
        phase_zone.grid_rowconfigure(0, weight=1)
        self.build_metric_stack_card(phase_zone, row=0, columns=4)
        metric_split.add(phase_zone, weight=2)

    def build_metric_stack_card(self, parent, row=0, column=0, columns=3):
        """Build one card that shows every speech-dialog metric family together."""
        self.metric_stack_frame = self.make_section(
            parent,
            "Metric Phases",
            row=row,
            column=column,
            sticky="nsew",
            key="metric_stack",
        )
        for column in range(columns):
            self.metric_stack_frame.grid_columnconfigure(column, weight=1, uniform="metric_stack")

        for index, family in enumerate(METRIC_FAMILY_SPECS):
            group_row = index // columns
            group_column = index % columns
            self.build_stage_metric_family_group(
                self.metric_stack_frame,
                family,
                row=group_row,
                column=group_column,
            )

    def build_outcome_metrics_card(self, parent, row=0, column=0):
        """Build the outcome metrics card."""
        self.metrics_frame = self.make_section(
            parent,
            "Outcome Metrics",
            row=row,
            column=column,
            sticky="nsew",
            key="evaluation",
        )
        self.metrics_frame.grid_columnconfigure(1, weight=1)
        self.metrics_frame.grid_columnconfigure(3, weight=1)

        metric_rows = [
            (("route", "Route"), ("duration", "Duration")),
            (("breakdown", "Breakdown"), ("reference_duration", "Reference")),
            (("line_sequence", "Line Seq"), ("line_changes", "Line Chg")),
            (("reference_line_sequence", "Ref Seq"), ("reference_line_changes", "Ref Chg")),
            (("reference_route", "Reference Route"), ("best_turn", "Best Turn")),
            (("constraint_route", "Target Route"), ("constraint_duration", "Target Time")),
            (("constraint_line_sequence", "Target Seq"), ("constraint_line_changes", "Target Chg")),
            (("constraint_gap", "Target Gap"), ("constraint_fullness", "Target Full")),
            (("candidate_routes", "Candidates"), ("route_revisions", "Revisions")),
            (("valid", "Valid"), ("goal", "Goal")),
            (("correct", "Correct"), ("runtime", "Runtime")),
        ]
        self.build_paired_metric_rows(
            self.metrics_frame,
            metric_rows,
            self.metric_values,
            label_font_size=BODY_FONT_SIZE,
            value_font_size=BODY_FONT_SIZE,
            wraplength=METRIC_WRAP,
        )

    def build_conversation_metrics_card(self, parent, row=0, column=0):
        """Build the conversation metrics card."""
        self.live_metrics_frame = self.make_section(
            parent,
            "Conversation Metrics",
            row=row,
            column=column,
            sticky="nsew",
            key="dialog_metrics",
        )
        self.live_metrics_frame.grid_columnconfigure(1, weight=1)
        self.live_metrics_frame.grid_columnconfigure(3, weight=1)

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
        self.build_paired_metric_rows(
            self.live_metrics_frame,
            live_metric_rows,
            self.live_metric_values,
            label_font_size=BODY_FONT_SIZE,
            value_font_size=BODY_FONT_SIZE,
        )

    def build_stage_metric_family_group(self, parent, family, row=0, column=0):
        """Build an unframed family group inside the complete metric stack card."""
        group = tk.Frame(parent, bg=GUI_COLORS["panel_bg"], bd=0, highlightthickness=0)
        group.grid(row=row, column=column, sticky="nsew", padx=(0, 8), pady=(0, 6))
        group.grid_columnconfigure(1, weight=1)

        tk.Label(
            group,
            text=family["title"],
            anchor="w",
            font=(GUI_FONT_FAMILY, GUI_FONT_NORMAL, "bold"),
            bg=GUI_COLORS["panel_bg"],
            fg=GUI_COLORS["text"],
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        for metric_row, (key, label) in enumerate(family["metrics"], start=1):
            tk.Label(
                group,
                text=label,
                anchor="w",
                font=(GUI_FONT_FAMILY, GUI_FONT_SMALL),
                bg=GUI_COLORS["panel_bg"],
                fg=GUI_COLORS["muted_text"],
            ).grid(row=metric_row, column=0, sticky="w", padx=(0, 6), pady=(0, 2))
            value = tk.Label(
                group,
                text="n/a",
                anchor="e",
                font=(GUI_MONO_FONT_FAMILY, GUI_FONT_SMALL, "bold"),
                bg=GUI_COLORS["panel_bg"],
                fg=GUI_COLORS["text"],
            )
            value.grid(row=metric_row, column=1, sticky="ew", padx=(0, 8), pady=(0, 2))
            self.stage_metric_values[key] = value

    def build_network_data_tab(self, parent):
        """Build the network-data page."""
        page = self.build_page_scroller(parent)
        self.build_network_overview_card(page, row=0)
        self.build_route_candidates_card(page, row=1)
        self.build_network_map_card(page, row=2)
        self.build_reference_browser_card(page, row=3)

    def build_network_overview_card(self, parent, row=0):
        """Build complete line and station data tables in one network card."""
        overview = build_network_overview(self.scenario["start_time_min"])
        frame = self.make_section(
            parent,
            f"Network Data: {overview.line_count} lines, {overview.station_count} stations, {overview.segment_count} segments",
            row=row,
            sticky="nsew",
            key="network_overview",
        )
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        line_panel = tk.Frame(frame, bg=GUI_COLORS["panel_bg"], bd=0, highlightthickness=0)
        line_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=0)
        line_panel.grid_columnconfigure(0, weight=1)
        self.network_line_table = self.build_scroll_table(
            line_panel,
            GUI_NETWORK_LINE_COLUMNS,
            height=GUI_NETWORK_TABLE_HEIGHT,
        )
        for item in overview.lines:
            self.network_line_table.insert(
                "",
                tk.END,
                values=(
                    item.name,
                    item.kind,
                    f"{item.headway_min}m",
                    f"{item.fullness_percent}%",
                    item.stop_count,
                    item.route,
                    item.segments,
                ),
            )

        station_panel = tk.Frame(frame, bg=GUI_COLORS["panel_bg"], bd=0, highlightthickness=0)
        station_panel.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=0)
        station_panel.grid_columnconfigure(0, weight=1)
        self.network_station_table = self.build_scroll_table(
            station_panel,
            GUI_NETWORK_STATION_COLUMNS,
            height=GUI_NETWORK_TABLE_HEIGHT,
        )
        for item in overview.stations:
            self.network_station_table.insert(
                "",
                tk.END,
                values=(
                    item.name,
                    f"{item.fullness_percent}%",
                    item.lines,
                    item.neighbors,
                    item.coordinates,
                ),
            )

    def build_network_map_card(self, parent, row=0):
        """Build the network map card."""
        self.map_frame = self.make_section(
            parent,
            "Network Map",
            row=row,
            sticky="nsew",
            pady=0,
            key="map",
        )
        self.map_frame.grid_rowconfigure(0, weight=1)
        self.map_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self.map_frame,
            bg=GUI_COLORS["map_bg"],
            highlightthickness=1,
            highlightbackground=GUI_COLORS["map_border"],
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

    def build_scroll_table(self, parent, columns, height):
        table = ttk.Treeview(
            parent,
            columns=tuple(column[0] for column in columns),
            show="headings",
            height=height,
            style="Data.Treeview",
        )
        for col, label, width, anchor, stretch in columns:
            table.heading(col, text=label)
            table.column(col, width=width, minwidth=width, anchor=anchor, stretch=stretch)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=table.yview)
        table.configure(yscrollcommand=scrollbar.set)
        table.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        return table

    def build_reference_browser_card(self, parent, row=0):
        """Build the transit reference card."""
        self.reference_frame = self.make_section(
            parent,
            "Transit Reference",
            row=row,
            sticky="nsew",
            key="reference",
        )
        self.build_reference_browser(self.reference_frame)

    def build_snapshot_card(self, parent, row=0, column=0):
        """Build the live snapshot card."""
        self.snapshot_frame = self.make_section(
            parent,
            "Live Snapshot",
            row=row,
            column=column,
            sticky="nsew",
            key="snapshot",
        )
        self.build_snapshot_grid(self.snapshot_frame)

    def build_run_context_card(self, parent, row=0, column=0):
        """Build the run context card."""
        self.summary_frame = self.make_section(
            parent,
            "Run Context",
            row=row,
            column=column,
            sticky="nsew",
            key="run",
        )
        self.summary_frame.grid_columnconfigure(1, weight=1)
        self.summary_frame.grid_columnconfigure(3, weight=1)

        summary_rows = [
            (("test_case", "Test"), ("persona", "Persona")),
            (("scenario", "Scenario"), ("speech", "Speech")),
            (("model", "Model"), ("device", "Device")),
            (("agent_a", "Agent A"), ("settings", "Settings")),
            (("run_state", "State"), ("elapsed", "Elapsed")),
            (("warnings", "Warnings"), ("", "")),
        ]
        self.build_paired_metric_rows(
            self.summary_frame,
            summary_rows,
            self.summary_values,
            label_font_size=GUI_FONT_SMALL + 1,
            value_font_size=BODY_FONT_SIZE,
            wraplength=SUMMARY_WRAP,
        )

    def build_paired_metric_rows(self, parent, rows, target_values, label_font_size=BODY_FONT_SIZE, value_font_size=BODY_FONT_SIZE, wraplength=None):
        """Build a compact two-column label/value grid."""
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_columnconfigure(3, weight=1)
        for row, pair in enumerate(rows):
            for column_offset, (key, label) in enumerate(pair):
                if not key:
                    continue
                label_column = column_offset * 2
                value_column = label_column + 1
                tk.Label(
                    parent,
                    text=label,
                    anchor="w",
                    font=(GUI_FONT_FAMILY, label_font_size),
                    bg=GUI_COLORS["panel_bg"],
                    fg=GUI_COLORS["muted_text"],
                ).grid(row=row, column=label_column, sticky="w", padx=(0, 6), pady=(0, 2))
                value = tk.Label(
                    parent,
                    text="-",
                    anchor="w",
                    font=(GUI_MONO_FONT_FAMILY, value_font_size, "bold"),
                    bg=GUI_COLORS["panel_bg"],
                    fg=GUI_COLORS["text"],
                    justify="left",
                )
                if wraplength is not None:
                    value.configure(wraplength=wraplength)
                value.grid(row=row, column=value_column, sticky="ew", padx=(0, 12), pady=(0, 2))
                target_values[key] = value

    @staticmethod
    def section_slug(title):
        """Convert a card title into a stable toggle key."""
        return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")

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
        outer = tk.Frame(
            parent,
            bg=GUI_COLORS["panel_bg"],
            bd=GUI_SECTION_BORDER_WIDTH,
            relief="solid",
            highlightthickness=0,
        )
        outer.grid(row=row, column=column, columnspan=columnspan, sticky=sticky, padx=padx, pady=pady)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(1, weight=1)

        header = tk.Frame(outer, bg=GUI_COLORS["panel_bg"], bd=0, highlightthickness=0)
        header.grid(row=0, column=0, sticky="ew", padx=SECTION_HEADER_PAD_X_LOCAL, pady=SECTION_HEADER_PAD_Y_LOCAL)
        header.grid_columnconfigure(0, weight=1)

        title_label = tk.Label(
            header,
            text=title,
            anchor="w",
            font=(GUI_FONT_FAMILY, SECTION_TITLE_SIZE, "bold"),
            bg=GUI_COLORS["panel_bg"],
            fg=GUI_COLORS["text"],
        )
        title_label.grid(row=0, column=0, sticky="w")

        content = tk.Frame(outer, bg=GUI_COLORS["panel_bg"], bd=0, highlightthickness=0)
        content.grid(row=1, column=0, sticky="nsew", padx=SECTION_PAD_X, pady=SECTION_CONTENT_PAD_Y_LOCAL)
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(0, weight=1)
        if key:
            self.section_widgets[key] = content
            self.section_visibility[key] = tk.BooleanVar(value=True)
            toggle = tk.Button(
                header,
                text="Hide",
                font=(GUI_FONT_FAMILY, GUI_FONT_SMALL, "bold"),
                command=lambda section_key=key: self.toggle_section(section_key),
                bg=GUI_COLORS["tab_button_bg"],
                fg=GUI_COLORS["text"],
                activebackground=GUI_COLORS["tab_unselected_hover"],
                activeforeground=GUI_COLORS["text"],
                bd=1,
                relief="solid",
            )
            toggle.grid(row=0, column=1, sticky="e")
            self.section_toggle_buttons[key] = toggle
        return content

    def toggle_section(self, section_key):
        """Hide or show a collapsible section."""
        content = self.section_widgets.get(section_key)
        if content is None:
            return
        visible = bool(self.section_visibility[section_key].get())
        if visible:
            content.grid_remove()
            self.section_visibility[section_key].set(False)
            self.section_toggle_buttons[section_key].configure(text="Show")
        else:
            content.grid()
            self.section_visibility[section_key].set(True)
            self.section_toggle_buttons[section_key].configure(text="Hide")

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
            tile = tk.Frame(
                parent,
                bg=GUI_COLORS["tab_bg"],
                bd=1,
                relief="solid",
                highlightthickness=0,
            )
            tile.grid(row=row, column=column, sticky="nsew", padx=(0, 2), pady=(0, 2))
            tile.grid_columnconfigure(0, weight=1)

            tk.Label(
                tile,
                text=label,
                anchor="w",
                font=(GUI_FONT_FAMILY, GUI_FONT_SMALL + 1),
                bg=GUI_COLORS["tab_bg"],
                fg=GUI_COLORS["muted_text"],
            ).grid(row=0, column=0, sticky="ew", padx=6, pady=(3, 0))

            value = tk.Label(
                tile,
                text="-",
                anchor="w",
                font=(GUI_MONO_FONT_FAMILY, BODY_FONT_SIZE, "bold"),
                bg=GUI_COLORS["tab_bg"],
                fg=GUI_COLORS["text"],
            )
            value.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 5))
            self.snapshot_values[key] = value

    def build_reference_browser(self, parent):
        """Build the station and line browser used for reference lookup."""
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=2)

        top = tk.Frame(parent, bg=GUI_COLORS["app_bg"], bd=0, highlightthickness=0)
        top.grid(row=0, column=0, sticky="nsew", pady=(0, SECTION_PAD_Y))
        top.grid_columnconfigure(0, weight=1)
        top.grid_columnconfigure(1, weight=1)
        top.grid_rowconfigure(0, weight=1)

        self.build_reference_list_panel(
            top,
            column=0,
            title="Lines",
            entries=[(line_name, line_name) for line_name in LINES],
            command=self.show_line,
            key_prefix="line",
        )
        self.build_reference_list_panel(
            top,
            column=1,
            title="Stations",
            entries=[(self.station_name(station), station) for station in STATION_POS],
            command=self.show_station,
            key_prefix="station",
        )

        bottom = tk.Frame(parent, bg=GUI_COLORS["app_bg"], bd=0, highlightthickness=0)
        bottom.grid(row=1, column=0, sticky="nsew")
        bottom.grid_columnconfigure(0, weight=1)
        bottom.grid_columnconfigure(1, weight=1)
        bottom.grid_rowconfigure(0, weight=1)

        line_detail_content = self.make_section(
            bottom,
            "Line details",
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, SECTION_PAD_X),
            pady=0,
            key="reference_line_details",
        )
        line_detail_content.grid_columnconfigure(0, weight=1)
        line_detail_content.grid_rowconfigure(0, weight=1)
        self.line_content = make_scrollable_frame(line_detail_content, GUI_COLORS["panel_bg"], padx=0, pady=0)
        self.line_content.grid_columnconfigure(0, weight=1)

        station_detail_content = self.make_section(
            bottom,
            "Station details",
            row=0,
            column=1,
            sticky="nsew",
            padx=(SECTION_PAD_X, 0),
            pady=0,
            key="reference_station_details",
        )
        station_detail_content.grid_columnconfigure(0, weight=1)
        station_detail_content.grid_rowconfigure(0, weight=1)
        self.station_content = make_scrollable_frame(station_detail_content, GUI_COLORS["panel_bg"], padx=0, pady=0)
        self.station_content.grid_columnconfigure(0, weight=1)

        first_line = next(iter(LINES))
        self.show_line(first_line)
        self.show_station(self.scenario["start_station"])

    def build_reference_list_panel(self, parent, column, title, entries, command, key_prefix):
        """Build a selectable list panel for lines or stations."""
        panel = self.make_section(
            parent,
            title,
            row=0,
            column=column,
            sticky="nsew",
            padx=(0, SECTION_PAD_X) if column == 0 else (SECTION_PAD_X, 0),
            key=f"reference_{key_prefix}_list",
        )
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(0, weight=1)

        scroller = make_scrollable_frame(panel, GUI_COLORS["panel_bg"], padx=0, pady=0)
        scroller.grid_columnconfigure(0, weight=1)

        for row, (label, value) in enumerate(entries):
            button = tk.Button(
                scroller,
                text=label,
                anchor="w",
                font=(GUI_FONT_FAMILY, REFERENCE_LIST_FONT, "bold"),
                command=lambda selected=value: command(selected),
                bg=GUI_COLORS["tab_button_bg"],
                fg=GUI_COLORS["text"],
                activebackground=GUI_COLORS["tab_unselected_hover"],
                activeforeground=GUI_COLORS["text"],
                bd=1,
                relief="solid",
            )
            button.grid(row=row, column=0, sticky="ew", padx=0, pady=(0, 4))

        return panel

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
        tabs = NotebookTabs(parent)
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        tabs.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
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
        line_fullness = line_fullness_percent(line_name, self.scenario["start_time_min"])
        tk.Label(
            parent,
            text=f"{self.short_line_label(line_name)}  {direction_text}  every {data['headway']}m  {line_fullness}% full",
            anchor="w",
            font=(GUI_FONT_FAMILY, GUI_FONT_NORMAL + 1, "bold"),
            bg=GUI_COLORS["panel_bg"],
            fg=GUI_COLORS["text"],
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
                    self.station_name(station),
                    f"{station_fullness_percent(station, self.scenario['start_time_min'])}%",
                    self.station_name(previous_station) if previous_station else "-",
                    self.station_name(next_station) if next_station else "-",
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
                travel_parts.append(f"{self.station_name(previous_station)}-{self.station_name(station)}:{segment_travel(previous_station, station)}m")
            if next_station:
                travel_parts.append(f"{self.station_name(station)}-{self.station_name(next_station)}:{segment_travel(station, next_station)}m")
            neighbors = (
                f"{self.station_name(previous_station) if previous_station else '-'}"
                f"  |  {self.station_name(next_station) if next_station else '-'}"
            )
            station_line_table.insert(
                "",
                tk.END,
                values=(
                    line_name,
                    f"{station_fullness_percent(station, self.scenario['start_time_min'])}%",
                    self.line_direction_text(line_name),
                    self.station_name(station),
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
        if LINES[line_name].get("kind") == "Ring":
            return f"{self.short_line_label(line_name)} loop"

        sequence = self.line_display_sequence(line_name)
        return f"{self.station_name(sequence[0])} <-> {self.station_name(sequence[-1])}"


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
        previous_label = self.station_name(previous_station) if previous_station else "start"
        next_label = self.station_name(next_station) if next_station else "end"
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
    def station_name(station):
        """Return the full station name for table and selector display."""
        return station if station else "-"

    @staticmethod
    def station_code(station):
        """Return a short station code for dense map and compact route labels."""
        if not station:
            return "-"
        letters = re.findall(r"[A-Za-z]", station.upper())
        return "".join(letters[:3]) if letters else station[:3].upper()

    def route_label(self, route):
        """Route label method for this module's MVC responsibility.

        Args:
            route: Input value used by `route_label`; see the function signature and caller context for the expected type.

        Returns:
            The computed value or side effect documented by the implementation.
        """
        return " -> ".join(self.station_name(station) for station in route)

    def compact_route_label(self, route):
        """Return a compact route label for narrow tables."""
        return " -> ".join(self.station_code(station) for station in route)

    def compact_line_sequence_label(self, route):
        """Return a compact line-sequence label for narrow tables."""
        estimate = estimate_route_time(
            route,
            self.scenario["start_time_min"],
            self.scenario["transfer_time_min"],
        )
        if not estimate:
            return "-"

        _, steps = estimate
        lines = route_line_sequence(steps)
        return " -> ".join(lines) if lines else "-"

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
        return " -> ".join(self.station_code(station) for station in stops)

    def compact_line_segment_text(self, line_name):
        """Compact line segment text method for this module's MVC responsibility.

        Args:
            line_name: Input value used by `compact_line_segment_text`; see the function signature and caller context for the expected type.

        Returns:
            The computed value or side effect documented by the implementation.
        """
        return "; ".join(
            f"{self.station_code(a)}-{self.station_code(b)}:{self.segment_minutes(a, b)}m"
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
        width = max(self.canvas.winfo_width(), MAP_MIN_WIDTH)
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
            label_text = self.station_name(station)
            label_font = max(MAP_STATION_MIN_FONT, int(MAP_STATION_FONT_SCALE * scale * 0.78))
            label_offset = radius + 8
            if x > width * 0.65:
                label_x = x - label_offset
                label_anchor = "e"
            else:
                label_x = x + label_offset
                label_anchor = "w"
            label_y = y
            text_id = self.canvas.create_text(
                label_x,
                label_y,
                text=label_text,
                anchor=label_anchor,
                font=(GUI_FONT_FAMILY, label_font, "bold"),
                fill=GUI_COLORS["map_outline"],
            )
            bbox = self.canvas.bbox(text_id)
            if bbox:
                pad = 3
                rect_id = self.canvas.create_rectangle(
                    bbox[0] - pad,
                    bbox[1] - pad,
                    bbox[2] + pad,
                    bbox[3] + pad,
                    fill=GUI_COLORS["map_bg"],
                    outline=GUI_COLORS["map_border"],
                )
                self.canvas.tag_lower(rect_id, text_id)
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
        speech_trace = self.pending_speech_traces.pop(speaker, None)
        self.textbox.insert(tk.END, f"{speaker}\n", speaker)
        if self.should_show_speech_trace(speech_trace):
            spoken = speech_trace.get("outgoing_text", message)
            understood = speech_trace.get("incoming_transcript", message)
            tts_engine = speech_trace.get("tts_engine", "tts")
            asr_engine = speech_trace.get("asr_engine", "asr")
            self.textbox.insert(tk.END, f"TTS spoken ({tts_engine}): ", "speech_label")
            self.textbox.insert(tk.END, f"{spoken}\n", "spoken_text")
            self.textbox.insert(tk.END, f"ASR understood ({asr_engine}): ", "speech_label")
            self.textbox.insert(tk.END, f"{understood}\n\n", "understood_text")
        else:
            self.textbox.insert(tk.END, f"{message}\n\n", f"{speaker} body")
        self.textbox.see(tk.END)

    @staticmethod
    def should_show_speech_trace(speech_trace):
        if not speech_trace:
            return False
        return bool(
            speech_trace.get("outgoing_enabled")
            or speech_trace.get("incoming_enabled")
            or speech_trace.get("outgoing_text") != speech_trace.get("incoming_transcript")
        )

    def add_system(self, message):
        """Add system method for this module's MVC responsibility.

        Args:
            message: Input value used by `add_system`; see the function signature and caller context for the expected type.

        Returns:
            The computed value or side effect documented by the implementation.
        """
        key, value = self.parse_system_message(message)
        if key and not self.minimal:
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
        generated_text = payload.get("generated_text", payload.get("source_text", ""))
        source_text = payload.get("outgoing_text", payload.get("source_text", ""))
        transcript = payload.get("incoming_transcript", payload.get("transcript", ""))
        self.pending_speech_traces[payload.get("speaker")] = payload

        gen_words = self.dialog_words(generated_text)
        out_words = self.dialog_words(source_text)
        tts_subs, tts_dels, tts_ins = self.edit_counts(gen_words, out_words)
        self.live_stats["tts_utterances"] += 1
        self.live_stats["tts_ref_words"] += len(gen_words)
        self.live_stats["tts_word_substitutions"] += tts_subs
        self.live_stats["tts_word_deletions"] += tts_dels
        self.live_stats["tts_word_insertions"] += tts_ins
        if payload.get("incoming_enabled"):
            self.live_stats["speech_incoming_enabled_count"] += 1
        if payload.get("outgoing_enabled"):
            self.live_stats["speech_outgoing_enabled_count"] += 1

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
        generation = float(payload.get("generation_sec", 0.0) or 0.0)
        speech = float(payload.get("speech_sec", 0.0) or 0.0)
        latency = float(payload.get("turn_latency_sec", 0.0) or 0.0)
        if speaker == "Agent B":
            if self.live_stats["agent_b_latency_count"] == 0:
                self.live_stats["first_agent_b_generation_sec"] = generation
                self.live_stats["first_agent_b_audio_sec"] = latency
            self.live_stats["agent_b_latency_total"] += latency
            self.live_stats["agent_b_latency_count"] += 1
        elif speaker == "Agent A":
            if self.live_stats["agent_a_latency_count"] == 0:
                self.live_stats["first_agent_a_generation_sec"] = generation
                self.live_stats["first_agent_a_audio_sec"] = generation + speech
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
        self.latest_metrics_text = metrics
        if not self.metric_values:
            return
        parsed = self.parse_metrics(metrics)
        mapping = {
            "Displayed route": "route",
            "Displayed duration": "duration",
            "Duration breakdown": "breakdown",
            "Displayed line sequence": "line_sequence",
            "Displayed line changes": "line_changes",
            "Reference route": "reference_route",
            "Reference duration": "reference_duration",
            "Reference line sequence": "reference_line_sequence",
            "Reference line changes": "reference_line_changes",
            "Constraint route": "constraint_route",
            "Constraint line sequence": "constraint_line_sequence",
            "Constraint line changes": "constraint_line_changes",
            "Constraint duration": "constraint_duration",
            "Constraint crowding": "constraint_fullness",
            "Constraint gap": "constraint_gap",
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
        if "Displayed line sequence" in parsed and parsed["Displayed line sequence"] != "None":
            line_sequence = [segment.strip() for segment in parsed["Displayed line sequence"].split("->")]
            self.metric_values["line_sequence"].configure(text=" -> ".join(line_sequence))
        if "Constraint route" in parsed and parsed["Constraint route"] != "None":
            route = [station.strip() for station in parsed["Constraint route"].split("->")]
            self.metric_values["constraint_route"].configure(text=self.route_label(route))
        if "Constraint line sequence" in parsed and parsed["Constraint line sequence"] != "None":
            line_sequence = [segment.strip() for segment in parsed["Constraint line sequence"].split("->")]
            self.metric_values["constraint_line_sequence"].configure(text=" -> ".join(line_sequence))

    def update_route_metric(self):
        """Update route metric method for this module's MVC responsibility.

        Returns:
            The computed value or side effect documented by the implementation.
        """
        route_text = self.route_label(self.current_route) if self.current_route else "No inferred route"
        self.metric_values["route"].configure(text=route_text)
        if "line_sequence" in self.metric_values or "line_changes" in self.metric_values:
            if self.current_route:
                estimate = estimate_route_time(
                    self.current_route,
                    self.scenario["start_time_min"],
                    self.scenario["transfer_time_min"],
                )
                if estimate:
                    _, steps = estimate
                    lines = route_line_sequence(steps)
                    line_sequence_text = " -> ".join(lines) if lines else "None"
                    line_change_text = str(max(len(lines) - 1, 0)) if lines else "0"
                else:
                    line_sequence_text = "No inferred route"
                    line_change_text = "0"
            else:
                line_sequence_text = "No inferred route"
                line_change_text = "0"
            if "line_sequence" in self.metric_values:
                self.metric_values["line_sequence"].configure(text=line_sequence_text)
            if "line_changes" in self.metric_values:
                self.metric_values["line_changes"].configure(text=line_change_text)
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
        asr_entity_wer = self.safe_ratio(
            self.live_stats["asr_keyword_fp"] + self.live_stats["asr_keyword_fn"],
            self.live_stats["asr_keyword_tp"] + self.live_stats["asr_keyword_fn"],
        )
        tts_change_rate = self.safe_ratio(
            self.live_stats["tts_word_substitutions"] + self.live_stats["tts_word_deletions"] + self.live_stats["tts_word_insertions"],
            self.live_stats["tts_ref_words"],
        )
        slot_accuracy = self.safe_ratio(self.live_stats["slot_hits"], self.live_stats["slot_total"])
        frame_accuracy = self.safe_ratio(self.live_stats["semantic_frame_hits"], self.live_stats["semantic_attempts"])
        semantic_error_rate = self.safe_ratio(self.live_stats["semantic_failures"], self.live_stats["semantic_attempts"])
        state_update_accuracy = self.safe_ratio(self.live_stats["valid_state_updates"], self.live_stats["state_updates"])
        nlu_route_valid_rate = state_update_accuracy
        nlu_goal_reached_rate = frame_accuracy
        nlu_station_mention_rate = slot_accuracy
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
        route_repair_rate = self.safe_ratio(self.live_stats["route_revisions"], self.live_stats["messages"])
        response_latency = self.safe_ratio(
            self.live_stats["agent_a_latency_total"] + self.live_stats["agent_b_latency_total"],
            self.live_stats["agent_a_latency_count"] + self.live_stats["agent_b_latency_count"],
        )
        cost_per_success = (
            f"{self.live_stats['messages']} turns"
            if route_success
            else "n/a"
        )

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
            "audio_snr_db": "n/a",
            "audio_si_snr_db": "n/a",
            "audio_clipping_rate": "n/a",
            "audio_packet_loss_rate": "n/a",
            "audio_sample_rate_mismatch": "n/a",
            "audio_loudness_lufs": "n/a",
            "audio_noise_estimate": "n/a",
            "audio_pesq": "n/a",
            "audio_dnsmos": "n/a",
            "vad_false_alarm_rate": "n/a",
            "vad_miss_rate": "n/a",
            "vad_detection_error_rate": "n/a",
            "vad_speech_non_speech_f1": "n/a",
            "vad_endpointing_latency_sec": "n/a",
            "diarization_der": "n/a",
            "diarization_missed_speech_rate": "n/a",
            "diarization_false_alarm_rate": "n/a",
            "diarization_speaker_confusion_rate": "n/a",
            "diarization_overlap_detection_f1": "n/a",
            "asr_word_error_rate": self.format_ratio(asr_wer),
            "asr_character_error_rate": self.format_ratio(asr_cer),
            "asr_token_error_rate": self.format_ratio(asr_wer),
            "asr_deletion_rate": self.format_ratio(self.safe_ratio(self.live_stats["asr_word_deletions"], self.live_stats["asr_ref_words"])),
            "asr_substitution_rate": self.format_ratio(self.safe_ratio(self.live_stats["asr_word_substitutions"], self.live_stats["asr_ref_words"])),
            "asr_insertion_rate": self.format_ratio(self.safe_ratio(self.live_stats["asr_word_insertions"], self.live_stats["asr_ref_words"])),
            "asr_entity_wer": self.format_ratio(asr_entity_wer),
            "asr_keyword_recall": self.format_ratio(asr_keyword_recall),
            "asr_confidence_calibration": "n/a",
            "slu_intent_accuracy": self.format_ratio(nlu_route_valid_rate),
            "slu_intent_error_rate": self.format_ratio(max(0.0, 1.0 - nlu_route_valid_rate)),
            "slu_slot_f1": self.format_ratio(slot_accuracy),
            "slu_slot_error_rate": self.format_ratio(max(0.0, 1.0 - slot_accuracy)),
            "slu_concept_error_rate": self.format_ratio(semantic_error_rate),
            "slu_sentence_semantic_accuracy": self.format_ratio(frame_accuracy),
            "slu_semantic_frame_accuracy": self.format_ratio(frame_accuracy),
            "dst_joint_goal_accuracy": self.format_ratio(route_success),
            "dst_average_goal_accuracy": self.format_ratio(frame_accuracy),
            "dst_requested_slot_f1": self.format_ratio(slot_accuracy),
            "dst_active_intent_accuracy": self.format_ratio(nlu_route_valid_rate),
            "dst_state_update_accuracy": self.format_ratio(state_update_accuracy),
            "dst_belief_state_calibration": "n/a",
            "dst_l2": "n/a",
            "dst_mrr": "n/a",
            "dst_roc": "n/a",
            "policy_dialog_act_accuracy": "n/a",
            "policy_dialog_act_f1": "n/a",
            "policy_next_action_accuracy": self.format_ratio(route_success),
            "policy_tool_call_exact_match": self.format_ratio(route_success),
            "policy_parameter_exact_match": self.format_ratio(route_success),
            "policy_invalid_action_rate": self.format_ratio(semantic_error_rate),
            "policy_fallback_rate": self.format_ratio(warning_pressure),
            "policy_repair_rate": self.format_ratio(route_repair_rate),
            "policy_confirmation_rate": self.format_ratio(question_rate),
            "tool_entity_match_rate": self.format_ratio(slot_accuracy),
            "tool_api_success_rate": self.format_ratio(route_success),
            "tool_tool_call_validity": self.format_ratio(route_success),
            "tool_result_relevance": self.format_ratio(route_success),
            "tool_hit_at_k": self.format_ratio(route_success),
            "tool_mrr": self.format_ratio(route_success),
            "tool_grounding_accuracy": self.format_ratio(route_success),
            "tool_hallucinated_field_rate": "n/a",
            "nlg_bleu": "n/a",
            "nlg_rouge": "n/a",
            "nlg_meteor": "n/a",
            "nlg_bert_score": "n/a",
            "nlg_slot_realization_accuracy": self.format_ratio(slot_accuracy),
            "nlg_delexicalized_bleu": "n/a",
            "nlg_distinct_1": self.format_ratio(distinct_1),
            "nlg_distinct_2": self.format_ratio(distinct_2),
            "nlg_repetition_rate": self.format_ratio(repetition_rate),
            "nlg_constraint_satisfaction_rate": self.format_ratio(route_success),
            "tts_predicted_mos": "n/a",
            "tts_intelligibility_wer": self.format_ratio(asr_wer),
            "tts_stoi": "n/a",
            "tts_mcd": self.format_ratio(tts_change_rate),
            "tts_pesq": "n/a",
            "tts_speechbert_score": "n/a",
            "tts_speaker_similarity": "n/a",
            "tts_f0_correlation": "n/a",
            "runtime_end_of_turn_detection_accuracy": "n/a",
            "runtime_endpointing_latency_sec": "n/a",
            "runtime_barge_in_true_positive_rate": "n/a",
            "runtime_barge_in_false_positive_rate": "n/a",
            "runtime_barge_in_suppression_latency_sec": "n/a",
            "runtime_response_latency_sec": self.format_seconds(response_latency),
            "runtime_time_to_first_token_sec": self.format_seconds(self.live_stats["first_agent_b_generation_sec"]) if self.live_stats["first_agent_b_generation_sec"] is not None else "n/a",
            "runtime_time_to_first_audio_sec": self.format_seconds(self.live_stats["first_agent_b_audio_sec"]) if self.live_stats["first_agent_b_audio_sec"] is not None else "n/a",
            "runtime_interruption_recovery_rate": self.format_ratio(recovery_rate),
            "e2e_task_success": self.format_ratio(route_success),
            "e2e_inform_rate": self.format_ratio(route_success),
            "e2e_request_success": self.format_ratio(route_success),
            "e2e_completion_rate": self.format_ratio(route_success),
            "e2e_abandonment_rate": self.format_ratio(1.0 - route_success),
            "e2e_escalation_rate": self.format_ratio(warning_pressure),
            "e2e_average_reward": self.format_ratio(predicted_satisfaction),
            "e2e_turns_to_success": str(self.live_stats["messages"]),
            "e2e_dialog_duration_sec": self.format_seconds(elapsed),
            "e2e_reprompt_count": str(self.live_stats["warnings"]),
            "e2e_confirmation_count": str(questions),
            "posthoc_predicted_user_satisfaction": self.format_ratio(predicted_satisfaction),
            "posthoc_per_domain_failure_rate": self.format_ratio(1.0 - route_success),
            "posthoc_cohort_fairness_gaps": "n/a",
            "posthoc_robustness_by_noise_gap": self.format_ratio(1.0 - robustness_noise),
            "posthoc_robustness_by_accent_gap": "n/a",
            "posthoc_robustness_by_device_gap": "n/a",
            "posthoc_robustness_by_environment_gap": speech_profile,
            "posthoc_cost_per_success": cost_per_success,
            "posthoc_safety_refusal_precision": "n/a",
            "posthoc_safety_refusal_recall": "n/a",
            "posthoc_privacy_redaction_accuracy": "n/a",
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
        target_gap = self.candidate_target_gap_label(candidate)

        if candidate["decision"] == "repeat":
            self.candidate_table.insert(
                "",
                tk.END,
                values=(
                    candidate["turn"],
                    f"{duration}m",
                    delta,
                    target_gap,
                    "repeat",
                    self.compact_route_label(candidate["route"]),
                    self.compact_line_sequence_label(candidate["route"]),
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
                target_gap,
                candidate["decision"],
                self.compact_route_label(candidate["route"]),
                self.compact_line_sequence_label(candidate["route"]),
            ),
        )
        self.update_live_dialog_metrics()

    @staticmethod
    def candidate_target_gap_label(candidate):
        """Return a compact candidate gap from the constraint-aware startup baseline."""
        duration_gap = candidate.get("duration_gap_min")
        change_gap = candidate.get("line_change_gap")
        fullness_gap = candidate.get("fullness_gap")
        if duration_gap is None and change_gap is None and fullness_gap is None:
            return "-"

        parts = []
        if duration_gap is not None:
            parts.append(f"{duration_gap:+d}m")
        if change_gap is not None:
            parts.append(f"{change_gap:+d}ch")
        if fullness_gap is not None:
            parts.append(f"{fullness_gap:+.1f}%")
        return " ".join(parts)

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
                    self.station_name(step["from"]),
                    self.station_name(step["to"]),
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
