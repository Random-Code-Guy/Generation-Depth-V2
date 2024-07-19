import os
import queue
import threading
from tkinter import filedialog, messagebox, DoubleVar, IntVar, StringVar, Checkbutton, HORIZONTAL, WORD, END

import ttkbootstrap as ttk
from ttkbootstrap.constants import *

from image_processing import load_image, generate_high_quality_depth_map
from model_loader import load_depth_anything_v2_model, load_default_model, reload_midas_repository
from utils import process_log_queue

MAX_LOG_LINES = 100


class DepthMapApp:
    def __init__(self, root):
        self.sigma_gaussian_var = None
        self.difference_multiplier_var = None
        self.sigma_difference_var = None
        self.sigma_blur_var = None
        self.filter_strength_var = None
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
        self.root = root
        self.root.title("Depth Map Generator - Random-Code-Guy")

        self.model = None
        self.model_name = None
        self.device = None

        self.create_widgets()
        self.log_queue = queue.Queue()
        process_log_queue(self)

    def create_widgets(self):
        # Model Settings
        self.model_frame = ttk.Labelframe(self.root, text="MODELS")
        self.model_frame.pack(side=TOP, fill=X, padx=5, pady=5)

        self.model_var = ttk.StringVar(value="Select Model")
        self.model_selector = ttk.Combobox(self.model_frame, textvariable=self.model_var,
                                           values=["ZoeDepth N", "ZoeDepth K", "ZoeDepth NK", "DepthAnythingV2 vits",
                                                   "DepthAnythingV2 vitb", "DepthAnythingV2 vitl",
                                                   "DepthAnythingV2 vitg"], width=30)
        self.model_selector.pack(side=LEFT, padx=5, pady=5)
        self.model_selector.configure(state="readonly")

        self.load_model_button = ttk.Button(self.model_frame, text="Load Model", command=self.load_model)
        self.load_model_button.pack(side=LEFT, padx=5, pady=5)

        self.reload_midas_button = ttk.Button(self.model_frame, text="Reload MiDaS Repository",
                                              command=self.reload_repository)
        self.reload_midas_button.pack(side=LEFT, padx=5, pady=5)

        # Map Output Settings
        self.map_frame = ttk.Labelframe(self.root, text="MAPS")
        self.map_frame.pack(side=TOP, fill=X, padx=5, pady=5)

        self.display_mode_var = ttk.StringVar(value="Grey Normal")
        self.display_mode_selector = ttk.Combobox(self.map_frame, textvariable=self.display_mode_var,
                                                  values=["Grey Normal", "Grey Inverted", "Colored Normal",
                                                          "Colored Inverted"])
        self.display_mode_selector.pack(side=LEFT, padx=5, pady=5)
        self.display_mode_selector.configure(state="readonly")

        # Image Options
        self.image_frame = ttk.Labelframe(self.root, text="OPTIONS")
        self.image_frame.pack(side=TOP, fill=X, padx=5, pady=5)

        # Parameter Sliders
        self.filter_strength_var = DoubleVar(value=0.998)
        self.sigma_blur_var = IntVar(value=20)
        self.sigma_difference_var = IntVar(value=40)
        self.difference_multiplier_var = DoubleVar(value=5)
        self.sigma_gaussian_var = IntVar(value=2)

        self.create_slider(self.image_frame, "Tiling Filter Strength", self.filter_strength_var, 0.9, 1.0,
                           format_string="{:.2f}")
        self.create_slider(self.image_frame, "Tiling Blur", self.sigma_blur_var, 1, 50,
                           format_string="{:.0f}")
        self.create_slider(self.image_frame, "Tiling Difference", self.sigma_difference_var, 1, 100,
                           format_string="{:.0f}")
        self.create_slider(self.image_frame, "Difference Multiplier", self.difference_multiplier_var, 1,
                           10, format_string="{:.2f}")
        self.create_slider(self.image_frame, "Final Filter Strength", self.sigma_gaussian_var, 1, 10,
                           format_string="{:.2f}")

        # Tiling Options
        self.tiling_options_frame = ttk.Labelframe(self.image_frame, text="Tiling Options")
        self.tiling_options_frame.pack(side=BOTTOM, fill=X, padx=5, pady=5)

        self.tiling_options = ["4,4", "8,8", "16,16", "32,32", "64,64"]
        self.tiling_variables = []
        for option in self.tiling_options:
            var = StringVar(value="")
            chk = Checkbutton(self.tiling_options_frame, text=option, variable=var, onvalue=option, offvalue="")
            chk.pack(anchor='w')
            self.tiling_variables.append(var)

        # Controls
        self.control_frame = ttk.Labelframe(self.root, text="CONTROLS")
        self.control_frame.pack(side=TOP, fill=X, padx=5, pady=5)

        self.load_button = ttk.Button(self.control_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=LEFT, padx=5, pady=5)

        self.generate_button = ttk.Button(self.control_frame, text="Generate Depth Map",
                                          command=self.generate_depth_map)
        self.generate_button.pack(side=LEFT, padx=5, pady=5)

        self.clear_log_button = ttk.Button(self.control_frame, text="Clear Log", command=self.clear_log)
        self.clear_log_button.pack(side=LEFT, padx=5, pady=5)

        self.reset_button = ttk.Button(self.control_frame, text="Reset", command=self.reset)
        self.reset_button.pack(side=LEFT, padx=5, pady=5)

        # Status Text
        self.status_text = ttk.Text(self.root, height=10, wrap=WORD)
        self.status_text.pack(fill=BOTH, expand=True, padx=5, pady=5)

        # RAM Usage Label
        self.ram_label = ttk.Label(self.root, text="RAM Usage: 0.00 MB")
        self.ram_label.pack(side=BOTTOM, fill=X, padx=5, pady=5)

        self.cpu_label = ttk.Label(self.root, text="CPU usage: 0%")
        self.cpu_label.pack(side=BOTTOM, fill=X, padx=5, pady=5)

    @staticmethod
    def create_slider(frame, label, variable, from_, to, format_string="{:.2f}"):
        slider_frame = ttk.Frame(frame)
        slider_frame.pack(side=LEFT, fill=X, padx=5, pady=5)

        ttk.Label(slider_frame, text=label).pack(side=TOP, padx=5, pady=5)
        slider = ttk.Scale(slider_frame, variable=variable, from_=from_, to=to, orient=HORIZONTAL)
        slider.pack(side=TOP, fill=X, padx=5, pady=5)

        value_label = ttk.Label(slider_frame, text=format_string.format(variable.get()))
        value_label.pack(side=TOP, padx=5, pady=5)

        variable.trace("w", lambda *args: value_label.config(text=format_string.format(variable.get())))

    def reload_repository(self):
        reload_midas_repository(self)

    def load_model(self):
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
            self.log_queue.put(f"Image not loaded from :  {self.image_path}.\n")
            return
        self.image_tensor, self.display_image = load_image(self.image_path, self.device)
        self.log_queue.put(f"Image loaded from :  {self.image_path}.\n")

    def generate_depth_map(self):
        def task():
            if not self.image_path:
                messagebox.showwarning("Warning", "Please load an image first.")
                return
            self.log_queue.put("Starting depth map generation...\n")
            display_mode = self.display_mode_var.get()
            input_filename = os.path.basename(self.image_path)
            output_filename = os.path.splitext(input_filename)[0] + '_depth.png'
            assets_folder = 'assets'
            if not os.path.exists(assets_folder):
                os.makedirs(assets_folder)
            output_path = os.path.join('maps', output_filename)

            sigma_gaussian = self.sigma_gaussian_var.get()
            filter_strength = self.filter_strength_var.get()
            sigma_blur = self.sigma_blur_var.get()
            sigma_difference = self.sigma_difference_var.get()
            difference_multiplier = self.difference_multiplier_var.get()

            selected_tiling_options = [var.get() for var in self.tiling_variables if var.get()]
            tile_sizes = [tuple(map(int, option.split(','))) for option in selected_tiling_options]

            generate_high_quality_depth_map(
                self.display_image, self.model, self.model_name, self.log_queue,
                assets_folder, display_mode, output_path,
                tile_sizes=tile_sizes, filter_strength=filter_strength, sigma_blur=sigma_blur,
                sigma_difference=sigma_difference, difference_multiplier=difference_multiplier,
                gaussian_sigma=sigma_gaussian)
            self.log_queue.put(f"Depth map saved successfully to maps folder as {output_filename}.\n")

        threading.Thread(target=task).start()

    def clear_log(self):
        self.status_text.delete(1.0, END)

    def reset(self):
        self.model_var.set("Select Model")
        self.display_mode_var.set("Grey Normal")
        self.status_text.delete(1.0, END)
        self.model = None
        self.model_name = None
        self.device = None
        self.image_path = None
        self.image_tensor = None
        self.display_image = None
        self.filter_strength_var.set(0.998)
        self.sigma_blur_var.set(20)
        self.sigma_difference_var.set(40)
        self.difference_multiplier_var.set(5)
        self.sigma_gaussian_var.set(2)
        for var in self.tiling_variables:
            var.set("")
        self.log_queue.queue.clear()
        self.log_queue.put("Application reset to default settings.\n")
