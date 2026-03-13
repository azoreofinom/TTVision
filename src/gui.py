import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sv_ttk
from PIL import Image, ImageTk,ImageDraw
import os
import analyze_video
import threading
import queue
import stats
import edit_video
import sys
import pywinstyles

def apply_theme_to_titlebar(root):
    version = sys.getwindowsversion()

    if version.major == 10 and version.build >= 22000:
        # Windows 11 – change only titlebar color
        pywinstyles.change_header_color(
            root,
            "#1c1c1c" if sv_ttk.get_theme() == "dark" else "#fafafa"
        )

    elif version.major == 10:
        # Windows 10 – enable dark titlebar without touching widget styling
        pywinstyles.change_header_color(root, "#1c1c1c")

        # force refresh of titlebar
        root.wm_attributes("-alpha", 0.99)
        root.wm_attributes("-alpha", 1)

class StatsGUI:
    def __init__(self, root, default_image_path="example_image.jpg"):
        self.video_path = None
        self.root = root
        self.root.title("TTVision")
        self.root.geometry("1200x800")

        self.stop_event = None
        self.worker_thread = None
        self.metadata_queue = queue.Queue()
        self.warning_queue = queue.Queue()
        self.game_metadata = None
        self.game_summary = None
        self.stats_object = None

        self.default_image_path = default_image_path

        # Main layout: 2 columns (left image, right stats)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        # ==== LEFT SIDE ====
        left_frame = ttk.Frame(self.root, padding=10)
        left_frame.grid(row=0, column=0, sticky="nsew")


        self.progress = ttk.Progressbar(left_frame, length=300, mode='determinate', maximum=100)
        self.progress.pack(pady=10)
       

        # Frame to hold buttons side by side
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(pady=10)

        # Load button
        self.open_btn = ttk.Button(btn_frame, text="Load Video", command=self.open_video)
        self.open_btn.pack(side=tk.LEFT, padx=5)

        # Analyze button
        self.analyze_btn = ttk.Button(btn_frame, text="Analyze", command=self.analyze_video)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)

        # Edit button
        self.edit_btn = ttk.Button(btn_frame, text="Edit", command=self.edit_video)
        self.edit_btn.pack(side=tk.LEFT, padx=5)


        # Cancel button
        self.cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self.cancel_analyze)
        self.cancel_btn.pack(side=tk.LEFT, padx=5)
        self.cancel_btn.config(state='disabled')


        #preset selector
        preset_frame = ttk.Frame(left_frame)
        preset_frame.pack(pady=5)

        ttk.Label(preset_frame, text="Video Editing Preset:").pack(side=tk.LEFT, padx=5)

        self.ffmpeg_preset = tk.StringVar(value="fast")
        preset_options = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"]
        self.preset_dropdown = ttk.Combobox(preset_frame, textvariable=self.ffmpeg_preset, values=preset_options, state="readonly", width=12)
        self.preset_dropdown.pack(side=tk.LEFT, padx=5)





        # Image display area
        self.image_label = ttk.Label(left_frame, text="Loading image...", anchor="center")
        self.image_label.pack(expand=True, fill="both")

        

        # ==== RIGHT SIDE ====
        right_frame = ttk.Frame(self.root, padding=10)
        right_frame.grid(row=0, column=1, sticky="nsew")

        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=0)
        right_frame.columnconfigure(0, weight=1)

        # ---- Stats Columns ----
        stats_frame = ttk.Frame(right_frame)

        stats_frame.grid(row=0, column=0, sticky="nsew", pady=10)
        # stats_frame.grid(row=0, column=0, sticky="", pady=10)

        stats_frame.columnconfigure(0, weight=1)  # left player
        stats_frame.columnconfigure(1, weight=0)  # metrics (centered)
        stats_frame.columnconfigure(2, weight=1)  # right player

        # Columns
        left_col = ttk.Frame(stats_frame)
        left_col.grid(row=0, column=0, sticky="n", padx=(0, 10))

        mid_col = ttk.Frame(stats_frame)
        mid_col.grid(row=0, column=1, sticky="n")

        right_col = ttk.Frame(stats_frame)
        right_col.grid(row=0, column=2, sticky="n", padx=(10, 0))

        # Column headers
        ttk.Label(left_col, text="Left Player", font=("Arial", 12, "bold")).pack(pady=15)
        ttk.Label(mid_col, text="Metric", font=("Arial", 12, "bold")).pack(pady=15)
        ttk.Label(right_col, text="Right Player", font=("Arial", 12, "bold")).pack(pady=15)

        # Metrics
        self.metrics = ["Points Won", "Points won on own serve","Points won on opponent serve","Serve Win %", "Receive Win %", "Most consecutive won", "Avg Rally Length (Win)"]
        self.left_stats = {}
        self.right_stats = {}

        for metric in self.metrics:
            ttk.Label(mid_col, text=metric).pack(pady=12)
            
            l_label = ttk.Label(left_col, text="--")
            l_label.pack(pady=12)
            self.left_stats[metric] = l_label

            r_label = ttk.Label(right_col, text="--")
            r_label.pack(pady=12)
            self.right_stats[metric] = r_label

       
       
        # --- Filters Frame inside right_frame ---
        self.filters_frame = ttk.LabelFrame(right_frame, text="Filters", padding=10)
        self.filters_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        self.create_filters(self.filters_frame)

        # --- Apply button at bottom of right_frame ---
        ttk.Button(right_frame, text="Apply Filters", command=self.apply_filters).grid(
            row=2, column=0, pady=10, sticky="ew"
        )


        # Load default image if available
        if os.path.exists(self.default_image_path):
            self.load_image(self.default_image_path)
        else:
            self.image_label.config(text="Default image not found")

        self.root.after(200, self.check_warning_queue)


    def create_filters(self, parent):
        """Create filters inside a dedicated frame within right_frame."""
        self.filters = {
            "Winner": {"Left": tk.BooleanVar(), "Right": tk.BooleanVar()},
            "Server": {"Left": tk.BooleanVar(), "Right": tk.BooleanVar()},
            "Serve Type": {"Short": tk.BooleanVar(), "Half Long": tk.BooleanVar(), "Long": tk.BooleanVar()},
            "Rally Length": {"Direct Serve": tk.BooleanVar(), "Short": tk.BooleanVar(), "Medium": tk.BooleanVar(), "Long": tk.BooleanVar()},
        }

        current_row = 0
        for name, options in self.filters.items():
            group = ttk.LabelFrame(parent, text=name, padding=5)
            group.grid(row=current_row, column=0, sticky="ew", pady=5)
            group.columnconfigure(tuple(range(len(options))), weight=1)

            for j, (opt, var) in enumerate(options.items()):
                ttk.Checkbutton(group, text=opt, variable=var).grid(row=0, column=j, padx=5, sticky="w")

            current_row += 1
 
    
    def apply_filters(self):
        """Collect and print selected filters."""
        selected = {
            name: [opt for opt, var in opts.items() if var.get()]
            for name, opts in self.filters.items()
        }
        print("Selected Filters:", selected)
        if self.stats_object is not None:
            bounces = self.stats_object.filter_stats(selected)
            image_copy = self.pil_image.copy()
            draw = ImageDraw.Draw(image_copy)
            for bounce in bounces:
                draw.circle(bounce,radius=2,fill=(255,0,0))
            
            self.tk_image = ImageTk.PhotoImage(image_copy)
            self.image_label.config(image=self.tk_image, text="")
       

    def open_video(self):
        """Open file dialog to select a new video."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.mkv *.webm *.avi *.mov *.wmv *.flv *.3gp")]
        )
        self.video_path = file_path
        print(f"path:{self.video_path}")
        # if file_path:
        #     self.load_image(file_path)


    def load_image(self, path):
        """Load and display an image from a path."""
        self.pil_image = Image.open(path)
        
        self.tk_image = ImageTk.PhotoImage(self.pil_image)
        self.image_label.config(image=self.tk_image, text="")

    def analyze_video(self):
        """Perform analysis on the currently loaded image."""
        print("Analyzing video...")  # replace with your actual logic
        if self.video_path:
            self.stop_event = threading.Event()
            self.worker_thread = threading.Thread(target=analyze_video.main,args=(self.video_path,self.stop_event,self.metadata_queue,self.warning_queue,self.progress_callback),daemon=True)
            self.worker_thread.start()
            self.edit_btn.config(state='disabled')
            self.analyze_btn.config(state='disabled')
            self.cancel_btn.config(state='normal')
            self.update_progress(0)
            # serve_detection.main(self.video_path)
        else:
            print("Load a video to analyze")


    def edit_video(self):
        print("Editing video...")
        if self.video_path:
            self.stop_event = threading.Event()
            self.worker_thread = threading.Thread(
                target=edit_video.remove_low_overlap_segments,
                args=(self.video_path, self.stop_event, self.progress_callback, self.ffmpeg_preset.get(), self.warning_queue),
                daemon=True
            )
            self.worker_thread.start()
            self.edit_btn.config(state='disabled')
            self.analyze_btn.config(state='disabled')
            self.cancel_btn.config(state='normal')
            self.update_progress(0)
        else:
            print("Load a video to edit")


    def cancel_analyze(self):
        """Cancel current operation or close the program."""
        print("Canceling operation...")
        if self.stop_event:
            self.stop_event.set()
        self.cancel_btn.config(state='disabled')
        self.analyze_btn.config(state='normal')
        self.edit_btn.config(state='normal')
        self.update_progress(0)
        # self.root.destroy()  # closes the window

    def progress_callback(self,step,total):
        percent = int((step / total) * 100)
        print(f"percent:{percent}")
        self.root.after(0, lambda: self.update_progress(percent))
        # self.root.after(0, lambda: self.update_status(f"Working... {percent}%"))
        if percent >= 100:
            # self.root.after(0, lambda: self.update_status("Done!"))
            self.root.after(0, lambda: self.analyze_btn.config(state='normal'))
            self.root.after(0, lambda: self.edit_btn.config(state='normal'))
            self.root.after(0, lambda: self.cancel_btn.config(state='disabled'))
            if not self.metadata_queue.empty():
                self.stats_object = stats.Stats(self.metadata_queue.get())
                self.game_summary = self.stats_object.get_summary_statistics()

                # self.game_summary = stats.get_stats(self.metadata_queue.get())
                print(f"summary:{self.game_summary}")

                self.root.after(0, lambda: self.update_stats(self.game_summary["Left"],self.game_summary["Right"]))
                


    def update_progress(self, value):
        self.progress.config(value=value)

    # ==== Stat Updater ====
    def update_stats(self, left_values, right_values):
        """Update stats for left and right players."""
        for metric, value in left_values.items():
            if metric in self.left_stats:
                self.left_stats[metric].config(text=round(value,2))
        for metric, value in right_values.items():
            if metric in self.right_stats:
                self.right_stats[metric].config(text=round(value,2))
    
  
    def check_warning_queue(self):
        try:
            while True:
                msg = self.warning_queue.get_nowait()
                messagebox.showwarning("Warning", msg)
                self.cancel_btn.config(state='disabled')
                self.analyze_btn.config(state='normal')
                self.edit_btn.config(state='normal')

        except queue.Empty:
            pass

        self.root.after(200, self.check_warning_queue)
   

if __name__ == "__main__":
    root = tk.Tk()
    sv_ttk.set_theme("dark")

    basedir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(basedir, ".."))

    # app = StatsGUI(root, default_image_path=os.path.join(basedir, "images/output_table_horizontal.png")) 
    app = StatsGUI(root, default_image_path=os.path.join(project_root, "images", "output_table_horizontal.png"))
    if sys.platform.startswith("win"):
        apply_theme_to_titlebar(root)
    
    root.mainloop()
