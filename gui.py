import os
import queue
import threading
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
from image_processing import load_image, generate_depth_map
from model_loader import load_depth_anything_v2_model, load_default_model, reload_midas_repository

MAX_LOG_LINES = 100
default_depth = os.path.join(os.getcwd(), "assets", "images", "default", "default.jpg")
tiling_options = [[2, 2], [4, 4], [6, 6], [8, 8], [10, 10], [12, 12], [14, 14], [16, 16], [18, 18], [20, 20], [22, 22],
                  [24, 24], [26, 26], [28, 28], [30, 30], [32, 32], [34, 34], [36, 36], [38, 38], [40, 40], [42, 42],
                  [44, 44], [46, 46], [48, 48], [50, 50], [52, 52], [54, 54], [56, 56], [58, 58], [60, 60], [62, 62],
                  [64, 64]]


def count_image_files(folder_path):
    image_count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_count += 1
    return image_count


class DepthMapApp(ctk.CTk):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sidebar_reset = None
        self.sidebar_clear_logs = None
        self.sidebar_reload_midas = None
        self.video_path = None
        self.sidebar_load_video = None
        self.sidebar_load_images = None
        self.logo_label = None
        self.sidebar_frame = None
        self.title("Depth Map Generator - Random-Code-Guy")

        self.cpu_label = None
        self.reload_midas_button = None
        self.display_image = None
        self.image_tensor = None
        self.image_path = None
        self.ram_label = None
        self.status_text = None
        self.reset_button = None
        self.clear_log_button = None
        self.generate_button = None
        self.control_frame = None
        self.load_button = None
        self.display_mode_selector = None
        self.image_frame = None
        self.display_mode_var = None
        self.map_frame = None
        self.load_model_button = None
        self.model_selector = None
        self.model_var = None
        self.model_frame = None
        self.smoothing_history_var = None

        self.model = None
        self.model_name = None
        self.device = None
        self.input_folder = None
        self.output_folder = None
        self.project_name = None
        self.image_total = 0
        self.image_count = 0

        self.stop_event = threading.Event()

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="GDepth V2.1", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_load_images = ctk.CTkButton(self.sidebar_frame, text="Load Images", command=self.select_folder)
        self.sidebar_load_images.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_load_video = ctk.CTkButton(self.sidebar_frame, text="Load Video", command=self.load_video)
        self.sidebar_load_video.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_reload_midas = ctk.CTkButton(self.sidebar_frame, text="Reload MiDaS",
                                                  command=self.reload_repository)
        self.sidebar_reload_midas.grid(row=3, column=0, padx=20, pady=10)

        self.sidebar_clear_logs = ctk.CTkButton(self.sidebar_frame, text="Clear Logs", command=self.clear_log)
        self.sidebar_clear_logs.grid(row=4, column=0, padx=20, pady=10)

        self.sidebar_reset = ctk.CTkButton(self.sidebar_frame, text="Reset Settings", command=self.reset)
        self.sidebar_reset.grid(row=5, column=0, padx=20, pady=10)

        # create log textbox
        self.status_text = ctk.CTkTextbox(self, width=250, wrap='word')
        self.status_text.grid(row=1, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # create model view
        self.model_frame = ctk.CTkScrollableFrame(self, label_text="Model Options")
        self.model_frame.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.model_frame.grid_columnconfigure(0, weight=0)

        self.model_var = ctk.StringVar(value="Select Model")
        self.model_selector = ctk.CTkComboBox(self.model_frame, variable=self.model_var,
                                              values=["ZoeDepth N", "ZoeDepth K", "ZoeDepth NK", "DepthAnythingV2 vits",
                                                      "DepthAnythingV2 vitb", "DepthAnythingV2 vitl",
                                                      "DepthAnythingV2 vitg"])
        self.model_selector.pack(side=ctk.LEFT, padx=5, pady=5)
        self.model_selector.grid(row=1, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

        self.display_mode_var = ctk.StringVar(value="Output Mode")
        self.display_mode_selector = ctk.CTkComboBox(self.model_frame, variable=self.display_mode_var,
                                                     values=["Normal", "Inverted"])
        self.display_mode_selector.grid(row=2, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

        self.color_map_var = ctk.StringVar(value="Output Style")
        self.color_map_selector = ctk.CTkComboBox(self.model_frame, variable=self.color_map_var, values=["gray",
                                                                                                         "viridis",
                                                                                                         "plasma",
                                                                                                         "inferno",
                                                                                                         "magma",
                                                                                                         "cividis"])

        self.color_map_selector.grid(row=3, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

        self.smoothing_switch = ctk.CTkSwitch(master=self.model_frame, text="Temporal Smoothing")
        self.smoothing_switch.grid(row=4, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

        self.smoothing_history_var = ctk.StringVar(value="Smoothing History")
        self.smoothing_history_selector = ctk.CTkComboBox(self.model_frame, variable=self.smoothing_history_var,
                                                          values=["5", "10", "15", "20", "25", "30", "35", "40", "45",
                                                                  "50"])
        self.smoothing_history_selector.grid(row=5, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

        self.outline_switch = ctk.CTkSwitch(master=self.model_frame, text="Outline Enhance")
        self.outline_switch.grid(row=6, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

        self.super_resolution_switch = ctk.CTkSwitch(master=self.model_frame, text="Resolution Enhance")
        self.super_resolution_switch.grid(row=7, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

        self.model_button = ctk.CTkButton(self.model_frame, text="Load Model", command=self.load_model)
        self.model_button.grid(row=8, column=0, padx=(10, 0), pady=(10, 0))

        # create video frame
        self.video_frame = ctk.CTkScrollableFrame(self, label_text="Video Options")
        self.video_frame.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.video_frame.grid_columnconfigure(0, weight=0)

        self.video_checkbox_1 = ctk.CTkCheckBox(master=self.video_frame, text="Save Audio (In)")
        self.video_checkbox_1.grid(row=1, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

        self.video_checkbox_2 = ctk.CTkCheckBox(master=self.video_frame, text="Zerolatency (Out)")
        self.video_checkbox_2.grid(row=2, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

        self.video_checkbox_3 = ctk.CTkCheckBox(master=self.video_frame, text="HW Acceleration (In/Out)")
        self.video_checkbox_3.grid(row=3, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

        self.output_codec_var = ctk.StringVar(value="Codec (Out)")
        self.output_codec_selector = ctk.CTkComboBox(self.video_frame, variable=self.output_codec_var,
                                                     values=["H264", "H265", "Nvenc"])
        self.output_codec_selector.grid(row=4, column=0, padx=(10, 0), pady=(15, 0), sticky="w")

        self.output_container_var = ctk.StringVar(value="Container (Out)")
        self.output_container_selector = ctk.CTkComboBox(self.video_frame, variable=self.output_container_var,
                                                         values=["mp4", "mkv"])
        self.output_container_selector.grid(row=5, column=0, padx=(10, 0), pady=(15, 0), sticky="w")

        # create preview and progressbar frame
        self.slider_progressbar_frame = ctk.CTkFrame(self)
        self.slider_progressbar_frame.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.slider_progressbar_frame.grid_columnconfigure(0, weight=1)
        self.slider_progressbar_frame.grid_rowconfigure(4, weight=1)

        depth_image = ctk.CTkImage(light_image=Image.open(default_depth), dark_image=Image.open(default_depth),
                                   size=(360, 260))

        self.depth_label = ctk.CTkLabel(self.slider_progressbar_frame, text="", image=depth_image)
        self.depth_label.grid(row=0, column=0, padx=(0, 0), pady=(15, 15), sticky="nsew")

        self.count_label = ctk.CTkLabel(self.slider_progressbar_frame, text="0/0",
                                        font=ctk.CTkFont(size=10, weight="bold"))
        self.count_label.grid(row=1, column=0, padx=(0, 0), pady=(0, 0))

        self.progressbar_1 = ctk.CTkProgressBar(self.slider_progressbar_frame)
        self.progressbar_1.grid(row=2, column=0, padx=(15, 15), pady=(0, 0), sticky="nsew")
        self.progressbar_1.configure(mode="indeterminnate")

        # Create tiling frame
        self.tiling_frame = ctk.CTkScrollableFrame(self, label_text="Tiling Options")
        self.tiling_frame.grid(row=1, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.tiling_frame.grid_columnconfigure(0, weight=1)
        self.tiling_frame_switches = []

        for i, option in enumerate(tiling_options):
            tiling_text = f"{option[0]}x{option[1]} Tiling"
            switch = ctk.CTkSwitch(master=self.tiling_frame, text=tiling_text)
            switch.grid(row=i, column=0, padx=0, pady=(0, 20))
            self.tiling_frame_switches.append((switch, option))

        # create Starting frame
        self.start_frame = ctk.CTkFrame(self)
        self.start_frame.grid(row=1, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")

        self.start_video_button = ctk.CTkButton(master=self.start_frame, fg_color="#3b8ed0", text="Start Video Process",
                                                border_width=0, text_color=("#ffffff", "#3b8ed0"))
        self.start_video_button.grid(row=0, column=0, padx=(40, 0), pady=(20, 10), sticky="nsew")

        self.start_depth_button = ctk.CTkButton(master=self.start_frame, fg_color="#3b8ed0", text="Start Depth Process",
                                                border_width=0, text_color=("#ffffff", "#3b8ed0"),
                                                command=self.toggle_process)

        self.start_depth_button.grid(row=1, column=0, padx=(40, 0), pady=(10, 0), sticky="nsew")

        self.status_text.insert("0.0", ">> Process Log Ouput\n\n")

        self.log_queue = queue.Queue()
        self.process_log_queue()

    def update_image(self, image_path):
        try:
            if not os.path.exists(image_path):
                return

            image = Image.open(image_path)

            if image.mode == 'I;16':
                # Convert to numpy array
                image_np = np.array(image, dtype=np.uint16)
                # Normalize the image to 0-255 range
                image_np = (image_np / 256).astype(np.uint8)
                # Convert to PIL Image
                image = Image.fromarray(image_np, mode='L')
                image = image.convert('RGB')

            elif image.mode != 'RGB':
                image = image.convert('RGB')

            new_image = ctk.CTkImage(light_image=image, dark_image=image, size=(360, 260))
            self.depth_label.configure(image=new_image)
            self.depth_label.image = new_image

        except Exception as e:
            print(f"Error in update_image: {e}")

    def reload_repository(self):
        reload_midas_repository(self)

    def load_model(self):
        if not self.project_name:
            messagebox.showwarning("Warning", "Load Images First.")
            return
        model_name = self.model_var.get()
        if model_name.startswith("DepthAnythingV2"):
            encoder = model_name.split(" ")[1]
            model_path = os.path.join(os.getcwd(), "models", model_name + ".pth")
            if not model_path:
                return
            load_depth_anything_v2_model(self, model_path, encoder)
        else:
            load_default_model(self, model_name)

    def load_image(self):
        if not self.model:
            messagebox.showwarning("Warning", "Please load the model first.")
            return
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not self.image_path:
            self.log_queue.put(f"Image not loaded.\n")
            return
        self.image_tensor, self.display_image = load_image(self.image_path, self.device)
        self.log_queue.put(f">> Images loaded.\n")
        self.update_image(self.image_path)

    def select_folder(self):
        self.input_folder = filedialog.askdirectory(title="Select Images Folder")

        if not self.input_folder:
            self.log_queue.put("Folder not selected.\n")
            return

        self.project_name = os.path.basename(os.path.normpath(self.input_folder))
        self.log_queue.put(f">> Project name set : {self.project_name}\n")

        self.image_total = count_image_files(self.input_folder)
        self.log_queue.put(f">> Loaded {self.image_total} image files.\n")

        self.update_image_count()
        self.create_folder_structure()

    def create_folder_structure(self):

        self.log_queue.put(f">> Creating project folder structure.\n")

        self.output_folder = os.path.join(os.getcwd(), "assets", "images", self.project_name)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        maps_folder = os.path.join(self.output_folder, "maps")
        if not os.path.exists(maps_folder):
            os.makedirs(maps_folder)

        tiles_folder = os.path.join(self.output_folder, "tiles")
        if not os.path.exists(tiles_folder):
            os.makedirs(tiles_folder)

        masks_folder = os.path.join(self.output_folder, "masks")
        if not os.path.exists(masks_folder):
            os.makedirs(masks_folder)

        self.log_queue.put(f">> Folder structure created.\n")

    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.mkv;*.avi")])
        if not self.video_path:
            self.log_queue.put(f">> Video not loaded.\n")
            return
        self.log_queue.put(f"Video loaded.\n")

    def get_selected_tiling_options(self):
        selected_tiling = []
        for switch, option in self.tiling_frame_switches:
            if switch.get():
                selected_tiling.append(option)
        return selected_tiling

    def update_image_count(self):
        new_text = str(self.image_count) + " / " + str(self.image_total)
        self.count_label.configure(text=new_text)

    def toggle_process(self):
        if self.start_depth_button.cget("text") == "Start Depth Process":
            self.start_depth_button.configure(text="Stop Depth Process")
            self.process_images()
        else:
            self.stop_event.set()
            self.start_depth_button.configure(text="Start Depth Process")

    def process_images(self):
        def task():
            if not self.input_folder:
                self.log_queue.put("Select input folder first.\n")
                return

            if not self.model:
                self.log_queue.put("Select depth model first.\n")
                return

            self.log_queue.put(">> Starting depth map generation...\n")

            display_mode = self.display_mode_var.get()
            color_map = self.color_map_var.get()
            smoothing_history = self.smoothing_history_var.get()

            if self.smoothing_switch.get():
                temporal_smoothing = True
            else:
                temporal_smoothing = False

            if self.outline_switch.get():
                out_lining = True
            else:
                out_lining = False

            if self.super_resolution_switch.get():
                use_super_resolution = True
            else:
                use_super_resolution = False

            tile_sizes = self.get_selected_tiling_options()
            self.log_queue.put(f">> Using tiling options {tile_sizes}.\n")

            for filename in os.listdir(self.input_folder):
                if self.stop_event.is_set():
                    self.log_queue.put("Depth map generation stopped.\n")
                    break

                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_count = self.image_count + 1
                    input_file = os.path.join(self.input_folder, filename)

                    # Load image
                    image = Image.open(input_file).convert('RGB')

                    # Process image
                    self.log_queue.put(f">> Processing {filename}\n")
                    self.update_image_count()
                    self.progressbar_1.start()
                    generate_depth_map(self, image, self.model, self.model_name, self.log_queue, self.output_folder,
                                       filename, display_mode, color_map, tile_sizes,
                                       use_temporal_smoothing=temporal_smoothing, history_size=smoothing_history,
                                       use_amf=use_super_resolution,use_rgf=use_super_resolution)

                    self.log_queue.put(f'>> {filename} Processed.\n')

            self.log_queue.put(f'>> {self.project_name} Completed!.\n')
            self.start_depth_button.configure(text="Start Depth Process")

        # Clear the stop event before starting the task
        self.stop_event.clear()
        threading.Thread(target=task).start()

    def clear_log(self):
        self.status_text.delete(1.0, ctk.END)

    def reset(self):
        self.progressbar_1.stop()
        self.update_image(default_depth)
        self.model_var.set("Select Model")
        self.display_mode_var.set("Output Mode")
        self.color_map_var.set("Output Style")
        self.smoothing_history_var.set("Smoothing History")
        self.status_text.delete(1.0, ctk.END)
        self.image_count = 0
        self.image_total = 0
        self.input_folder = None
        self.output_folder = None
        self.model = None
        self.model_name = None
        self.device = None
        self.image_path = None
        self.image_tensor = None
        self.display_image = None
        self.log_queue.queue.clear()
        self.log_queue.put("Application reset to default settings.\n")

    def process_log_queue(self):
        try:
            while True:
                log_message = self.log_queue.get_nowait()
                self.status_text.insert(ctk.END, log_message)
                self.status_text.see(ctk.END)
                line_count = int(self.status_text.index('end-1c').split('.')[0])
                if line_count > MAX_LOG_LINES:
                    self.status_text.delete(1.0, 2.0)
        except queue.Empty:
            pass
        self.after(100, self.process_log_queue)
