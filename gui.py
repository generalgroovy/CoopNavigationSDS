import queue
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from config import (
    GUI_WIDTH,
    GUI_HEIGHT,
    GUI_DIALOG_MIN_WIDTH,
    GUI_MAP_MIN_WIDTH,
)
from metro_data import LINES, STATION_POS, line_segment_text
from route_planner import fmt_time


class DialogWindow:
    def __init__(self, event_queue, scenario):
        self.event_queue = event_queue
        self.scenario = scenario
        self.current_route = []
        self.summary_values = {}
        self.metric_values = {}

        self.root = tk.Tk()
        self.root.title("Speech Dialog Navigation Evaluation")
        self.root.geometry(f"{GUI_WIDTH}x{GUI_HEIGHT}")
        self.root.minsize(1040, 640)

        self.configure_style()
        self.build_layout()
        self.populate_line_table()
        self.draw_network()

        self.canvas.bind("<Configure>", lambda _: self.draw_network())
        self.root.after(100, self.process_events)

    def configure_style(self):
        self.root.configure(bg="#f5f6f8")
        style = ttk.Style()
        style.configure("Panel.TLabelframe", padding=8)
        style.configure("Panel.TLabelframe.Label", font=("Segoe UI", 9, "bold"))
        style.configure("Metric.TLabel", font=("Segoe UI", 9))
        style.configure("MetricValue.TLabel", font=("Consolas", 9, "bold"))

    def build_layout(self):
        self.main = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            sashwidth=5,
            bg="#c9ced6",
        )
        self.main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.left = ttk.Frame(self.main)
        self.right = ttk.Frame(self.main)
        self.main.add(self.left, minsize=GUI_DIALOG_MIN_WIDTH)
        self.main.add(self.right, minsize=GUI_MAP_MIN_WIDTH)

        self.build_left_panel()
        self.build_right_panel()

    def build_left_panel(self):
        self.left.grid_columnconfigure(0, weight=1)
        self.left.grid_rowconfigure(2, weight=1)

        self.summary_frame = ttk.LabelFrame(self.left, text="Run", style="Panel.TLabelframe")
        self.summary_frame.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=(0, 6))
        self.summary_frame.grid_columnconfigure(1, weight=1)

        for row, key in enumerate(("test_case", "persona", "scenario", "speech")):
            ttk.Label(self.summary_frame, text=self.label_for(key), style="Metric.TLabel").grid(
                row=row,
                column=0,
                sticky="w",
                pady=1,
            )
            value = ttk.Label(self.summary_frame, text="-", style="MetricValue.TLabel")
            value.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=1)
            self.summary_values[key] = value

        self.metrics_frame = ttk.LabelFrame(self.left, text="Evaluation", style="Panel.TLabelframe")
        self.metrics_frame.grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=(0, 6))
        self.metrics_frame.grid_columnconfigure(1, weight=1)

        metric_keys = (
            "route",
            "duration",
            "breakdown",
            "valid",
            "goal",
            "correct",
            "runtime",
        )
        for row, key in enumerate(metric_keys):
            ttk.Label(self.metrics_frame, text=self.label_for(key), style="Metric.TLabel").grid(
                row=row,
                column=0,
                sticky="nw",
                pady=1,
            )
            value = ttk.Label(
                self.metrics_frame,
                text="-",
                style="MetricValue.TLabel",
                wraplength=390,
                justify="left",
            )
            value.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=1)
            self.metric_values[key] = value

        self.transcript_frame = ttk.LabelFrame(self.left, text="Transcript", style="Panel.TLabelframe")
        self.transcript_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 6))
        self.transcript_frame.grid_rowconfigure(0, weight=1)
        self.transcript_frame.grid_columnconfigure(0, weight=1)

        self.textbox = ScrolledText(
            self.transcript_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 9),
            relief=tk.FLAT,
            height=18,
        )
        self.textbox.grid(row=0, column=0, sticky="nsew")
        self.textbox.tag_config("Agent A", foreground="#174ea6", font=("Segoe UI", 9, "bold"))
        self.textbox.tag_config("Agent B", foreground="#137333", font=("Segoe UI", 9, "bold"))
        self.textbox.tag_config("system", foreground="#6b7280")
        self.textbox.tag_config("warning", foreground="#b3261e", font=("Segoe UI", 9, "bold"))

    def build_right_panel(self):
        self.right.grid_columnconfigure(0, weight=1)
        self.right.grid_rowconfigure(0, weight=1)
        self.right.grid_rowconfigure(1, weight=0)

        self.map_frame = ttk.LabelFrame(self.right, text="Network", style="Panel.TLabelframe")
        self.map_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        self.map_frame.grid_rowconfigure(0, weight=1)
        self.map_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self.map_frame,
            bg="#fbfaf7",
            highlightthickness=1,
            highlightbackground="#d0d5dd",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.line_frame = ttk.LabelFrame(self.right, text="Lines", style="Panel.TLabelframe")
        self.line_frame.grid(row=1, column=0, sticky="ew")
        self.line_frame.grid_columnconfigure(0, weight=1)

        self.line_table = ttk.Treeview(
            self.line_frame,
            columns=("line", "interval", "stations", "times"),
            show="headings",
            height=min(len(LINES), 5),
        )

        columns = [
            ("line", "Line", 110, "w"),
            ("interval", "Int.", 55, "center"),
            ("stations", "Stations", 300, "w"),
            ("times", "Times", 360, "w"),
        ]
        for col, label, width, anchor in columns:
            self.line_table.heading(col, text=label)
            self.line_table.column(col, width=width, anchor=anchor, stretch=True)

        self.line_table.grid(row=0, column=0, sticky="ew")

    def populate_line_table(self):
        for line_name, data in LINES.items():
            self.line_table.insert(
                "",
                tk.END,
                values=(
                    line_name,
                    f"{data['headway']}m",
                    " -> ".join(data["stops"]),
                    line_segment_text(line_name),
                ),
            )

    def canvas_transform(self):
        width = max(self.canvas.winfo_width(), 420)
        height = max(self.canvas.winfo_height(), 300)

        xs = [x for x, _ in STATION_POS.values()]
        ys = [y for _, y in STATION_POS.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        padding_x = 58
        padding_y = 44
        source_w = max(max_x - min_x, 1)
        source_h = max(max_y - min_y, 1)
        scale = max(
            0.3,
            min(
                (width - 2 * padding_x) / source_w,
                (height - 2 * padding_y) / source_h,
            ),
        )

        used_w = source_w * scale
        used_h = source_h * scale
        offset_x = (width - used_w) / 2
        offset_y = (height - used_h) / 2

        def transform(point):
            x, y = point
            return (
                offset_x + (x - min_x) * scale,
                offset_y + (y - min_y) * scale,
            )

        return transform, scale

    def draw_network(self):
        if not hasattr(self, "canvas"):
            return

        self.canvas.delete("all")
        transform, scale = self.canvas_transform()

        route_edges = {
            tuple(sorted((a, b)))
            for a, b in zip(self.current_route, self.current_route[1:])
        }

        for _, data in LINES.items():
            for a, b in zip(data["stops"], data["stops"][1:]):
                x1, y1 = transform(STATION_POS[a])
                x2, y2 = transform(STATION_POS[b])
                self.canvas.create_line(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill="#ffffff",
                    width=max(8, int(11 * scale)),
                    capstyle=tk.ROUND,
                )

        for _, data in LINES.items():
            for a, b in zip(data["stops"], data["stops"][1:]):
                x1, y1 = transform(STATION_POS[a])
                x2, y2 = transform(STATION_POS[b])
                self.canvas.create_line(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=data["color"],
                    width=max(4, int(6 * scale)),
                    capstyle=tk.ROUND,
                )

        for a, b in route_edges:
            x1, y1 = transform(STATION_POS[a])
            x2, y2 = transform(STATION_POS[b])
            self.canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill="#d93025",
                width=max(8, int(10 * scale)),
                capstyle=tk.ROUND,
            )

        start = self.scenario["start_station"]
        destination = self.scenario["destination_station"]

        for station, pos in STATION_POS.items():
            x, y = transform(pos)

            if station == start:
                fill = "#cfe8ff"
                radius = 10
            elif station == destination:
                fill = "#ffe2a8"
                radius = 10
            elif station in self.current_route:
                fill = "#ffd7d2"
                radius = 9
            else:
                fill = "#ffffff"
                radius = 7

            radius = max(5, int(radius * scale))
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=fill,
                outline="#202124",
                width=1,
            )
            self.canvas.create_text(
                x,
                y - radius - 10,
                text=station,
                font=("Segoe UI", max(7, int(8 * scale))),
                fill="#202124",
            )

        self.update_route_metric()

    def process_events(self):
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
                elif kind == "metrics":
                    _, metrics = event
                    self.apply_metrics(metrics)
                elif kind == "done":
                    self.add_system("Finished")

        except queue.Empty:
            pass

        self.root.after(100, self.process_events)

    def add_message(self, speaker, message):
        self.textbox.insert(tk.END, f"{speaker}: ", speaker)
        self.textbox.insert(tk.END, f"{message}\n\n")
        self.textbox.see(tk.END)

    def add_system(self, message):
        key, value = self.parse_system_message(message)
        if key:
            self.set_summary(key, value)
        self.textbox.insert(tk.END, f"[{message}]\n", "system")
        self.textbox.see(tk.END)

    def add_warning(self, message):
        self.textbox.insert(tk.END, f"[warning] {message}\n\n", "warning")
        self.textbox.see(tk.END)

    def apply_metrics(self, metrics):
        parsed = self.parse_metrics(metrics)
        mapping = {
            "Displayed route": "route",
            "Displayed duration": "duration",
            "Duration breakdown": "breakdown",
            "Route valid": "valid",
            "Route reaches goal": "goal",
            "Route correct": "correct",
            "Runtime": "runtime",
        }

        for source_key, metric_key in mapping.items():
            if source_key in parsed:
                self.metric_values[metric_key].configure(text=parsed[source_key])

        if "Displayed route" in parsed and parsed["Displayed route"] != "None":
            self.metric_values["route"].configure(text=parsed["Displayed route"])

    def update_route_metric(self):
        route_text = " -> ".join(self.current_route) if self.current_route else "No inferred route"
        self.metric_values["route"].configure(text=route_text)

    def parse_metrics(self, metrics):
        parsed = {}
        for line in metrics.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
        return parsed

    def parse_system_message(self, message):
        mapping = {
            "Test case": "test_case",
            "Persona": "persona",
            "Scenario": "scenario",
            "Speech transport": "speech",
        }
        if ":" not in message:
            return None, None

        key, value = message.split(":", 1)
        return mapping.get(key.strip()), value.strip()

    def set_summary(self, key, value):
        if key in self.summary_values:
            self.summary_values[key].configure(text=value)

    def label_for(self, key):
        labels = {
            "test_case": "Test",
            "persona": "Persona",
            "scenario": "Scenario",
            "speech": "Speech",
            "route": "Route",
            "duration": "Duration",
            "breakdown": "Breakdown",
            "valid": "Valid",
            "goal": "Goal",
            "correct": "Correct",
            "runtime": "Runtime",
        }
        return labels.get(key, key)

    def run(self):
        self.root.mainloop()
