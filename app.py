import customtkinter as ctk
from gui import DepthMapApp


if __name__ == "__main__":
    ctk.set_appearance_mode("System")  # Options: "dark", "light"
    ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

    app = DepthMapApp()
    app.geometry("1100x700")
    app.resizable(False, False)  # Make the window non-resizable
    app.mainloop()
