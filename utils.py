import psutil
import queue
import io
import os
import threading

MAX_LOG_LINES = 100


class Logger(io.StringIO):
    def __init__(self, log_queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_queue = log_queue

    def write(self, message):
        for line in message.splitlines():
            if line.strip():
                self.log_queue.put(f">> {line}\n")

    def flush(self):
        pass


def process_log_queue(app):
    try:
        while True:
            message = app.log_queue.get_nowait()
            app.status_text.insert("end", message)
            app.status_text.see("end")

            while int(app.status_text.index('end-1c').split('.')[0]) > MAX_LOG_LINES:
                app.status_text.delete('1.0', '2.0')
    except queue.Empty:
        pass
    app.root.after(100, process_log_queue, app)


def update_usage(app):
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        app.cpu_label.config(text=f"CPU usage: {cpu_usage}%")

        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        app.ram_label.config(text=f"RAM Usage: {memory_usage:.2f} MB")

        stop_update_thread(app)
        app.root.after(1000, start_update_thread, app)


def start_update_thread(app):
    app.update_thread = threading.Thread(target=update_usage, args=(app,))
    app.update_thread.daemon = True
    app.update_thread.start()


def stop_update_thread(app):
    if app.update_thread is not None and app.update_thread.is_alive():
        app.update_thread.join()
        app.update_thread = None
