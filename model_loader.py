import torch
import threading

from utils import Logger
import sys
import os

# Append the cloned repository to the system path
sys.path.append(os.path.join(os.getcwd(), 'Depth-Anything-V2'))

# Import the Depth Anything V2 model
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError as e:
    print(f"Error importing DepthAnythingV2: {e}")

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def reload_midas_repository(app):
    def task():
        try:
            app.log_queue.put(f">> Reloading MiDaS Repository.\n")
            logger = Logger(app.log_queue)
            sys.stdout = logger
            sys.stderr = logger
            torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
            app.log_queue.put(f">> MiDaS Repository successfully Reloaded.\n")
        except Exception as reload_e:
            app.log_queue.put(f"Error Reloading MiDaS Repository: {str(reload_e)}\n")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    threading.Thread(target=task).start()


def load_default_model(app, model_name):
    def task():
        try:
            logger = Logger(app.log_queue)
            sys.stdout = logger
            sys.stderr = logger
            torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
            if "Zoe" in model_name:
                model_code = model_name.replace(" ", "_").replace("ZoeDepth", "ZoeD")
                app.log_queue.put(f"{model_code}\n")
                app.model = torch.hub.load("isl-org/ZoeDepth", model_code, pretrained=True, trust_repo=True)
                app.model_name = "ZoeDepth"

            app.model.to(DEVICE)
            app.model.eval()

            app.log_queue.put(f">> {model_name} model loaded successfully.\n")
        except Exception as default_e:
            app.log_queue.put(f"Error loading model: {str(default_e)}\n")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    threading.Thread(target=task).start()


def load_depth_anything_v2_model(app, model_path, encoder):
    def task():
        try:
            logger = Logger(app.log_queue)
            sys.stdout = logger
            sys.stderr = logger

            app.model = DepthAnythingV2(**model_configs[encoder])
            app.model.load_state_dict(torch.load(model_path.replace(" ", "_"), map_location='cpu'))
            app.model = app.model.to(DEVICE).eval()
            app.model_name = f"DepthAnythingV2_{encoder}"

            app.log_queue.put(f">> DepthAnythingV2 {encoder} model loaded successfully.\n")
        except Exception as version2_e:
            app.log_queue.put(f"Error loading model: {str(version2_e)}\n")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    threading.Thread(target=task).start()
