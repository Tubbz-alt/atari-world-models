import logging
import queue
import threading
from tkinter import Canvas, Tk

from PIL import Image, ImageTk
from torchvision import transforms
from xvfbwrapper import Xvfb

from .play import PlayGame
from .utils import Step

logger = logging.getLogger(__name__)


class GUI:
    def __init__(self, demo):
        self.demo = demo
        self.value_q = queue.Queue()
        self.event_q = queue.Queue()

    def gui(self):
        self.root = Tk()
        self.root.after("idle", self.update)

        self.root.bind("<Key>", self.print_key)

        self.screen = Canvas(self.root, width=600, height=400)
        self.screen.pack()

        self.vae_screen = Canvas(self.root, width=600, height=400)
        self.vae_screen.pack()

        self.screen_obj = ImageTk.PhotoImage(
            image=Image.new("RGBA", (600, 400), color=(0, 0, 0))
        )
        self.screen_obj_actual = self.screen.create_image(
            0, 0, anchor="nw", image=self.screen_obj
        )

        self.vae_screen_obj = ImageTk.PhotoImage(
            image=Image.new("RGBA", (600, 400), color=(0, 0, 0))
        )
        self.vae_screen_obj_actual = self.vae_screen.create_image(
            0, 0, anchor="nw", image=self.vae_screen_obj
        )

        player = Player(self.demo, self.value_q, self.event_q)
        player_thread = player.start()

        self.root.mainloop()
        player_thread.join()

    def print_key(self, event):
        self.event_q.put(event.keysym)
        if event.keysym == "q":
            self.root.destroy()

    def update(self):
        real_screen, small_screen, z, vae_screen, h = self.value_q.get()

        self.image = ImageTk.PhotoImage(transforms.ToPILImage(mode="RGB")(real_screen))
        self.screen.itemconfig(self.screen_obj_actual, image=self.image)

        self.image_vae = ImageTk.PhotoImage(
            transforms.ToPILImage(mode="RGB")(vae_screen[0]).resize((600, 400))
        )
        self.vae_screen.itemconfig(self.vae_screen_obj_actual, image=self.image_vae)

        self.root.after(5, self.update)


class Player:
    def __init__(self, demo, value_q, event_q):
        self.demo = demo
        self.value_q = value_q
        self.event_q = event_q

    def start(self):
        thread = threading.Thread(target=self.play)
        thread.start()
        return thread

    def play(self):
        vdisplay = Xvfb()
        vdisplay.start()
        player = PlayGame(
            self.demo.game,
            self.demo.observations_dir,
            self.demo.models_dir,
            self.demo.samples_dir,
        )

        quit = False

        while True:
            if quit:
                break

            playing = player.generator()

            for i in playing:
                self.value_q.put(i)
                try:
                    event = self.event_q.get_nowait()
                except queue.Empty:
                    pass
                else:
                    if event == "q":
                        logger.info("quit event received")
                        quit = True
                        break
                    elif event == "r":
                        logger.info("restart event received")
                        player = PlayGame(
                            self.demo.game,
                            self.demo.observations_dir,
                            self.demo.models_dir,
                            self.demo.samples_dir,
                        )
                        break
        logger.info("stopping xvfb")
        vdisplay.stop()


class Demo(Step):
    hyperparams_key = None

    def __call__(self):
        gui = GUI(self)
        gui.gui()
