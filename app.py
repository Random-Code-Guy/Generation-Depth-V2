import ttkbootstrap as ttk
import os
from gui import DepthMapApp
from utils import start_update_thread

if __name__ == "__main__":
    if not os.path.exists('maps'):
        os.makedirs('maps')
    if not os.path.exists('assets'):
        os.makedirs('assets')
        
    root = ttk.Window(themename="superhero")
    app = DepthMapApp(root)
    start_update_thread(app)
    root.mainloop()
