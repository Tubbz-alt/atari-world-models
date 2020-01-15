import logging
import queue
import threading
from tkinter import Canvas, Label, Message, Tk, ttk

from PIL import Image, ImageTk
from torchvision import transforms
from xvfbwrapper import Xvfb

from .play import PlayGame
from .utils import Step

logger = logging.getLogger(__name__)


LEGEND = """
<space> ... pause/unpause
q       ... quit
r       ... restart"""


class GUI:
    def __init__(self, demo):
        self.demo = demo
        self.value_q = queue.Queue()
        self.event_q = queue.Queue()

    def gui(self):
        self.root = Tk()
        self.root.config(background="White")
        self.root.after("idle", self.update)

        self.root.bind("<Key>", self.handle_key_press)

        self.legend = Message(
            self.root, background="White", text=LEGEND, font=("profont", 29), width=600
        )
        self.legend.grid(row=1, column=0, padx=200)

        self.action_left = ttk.Progressbar(
            self.root, orient="horizontal", length=200, mode="determinate"
        )
        self.action_right = ttk.Progressbar(
            self.root, orient="horizontal", length=200, mode="determinate"
        )
        self.action_gas = ttk.Progressbar(
            self.root, orient="horizontal", length=200, mode="determinate"
        )
        self.action_break = ttk.Progressbar(
            self.root, orient="horizontal", length=200, mode="determinate"
        )
        self.action_left.grid(row=4, column=0)
        self.action_right.grid(row=6, column=0)
        self.action_gas.grid(row=8, column=0)
        self.action_break.grid(row=10, column=0)

        self.label_left = Label(
            self.root, background="White", text="Steering left", font=("profont", 29)
        )
        self.label_right = Label(
            self.root, background="White", text="Steering right", font=("profont", 29)
        )
        self.label_gas = Label(
            self.root, background="White", text="Gas", font=("profont", 29)
        )
        self.label_break = Label(
            self.root, background="White", text="Break", font=("profont", 29)
        )
        self.label_left.grid(row=3, column=0)
        self.label_right.grid(row=5, column=0)
        self.label_gas.grid(row=7, column=0)
        self.label_break.grid(row=9, column=0)

        self.label_screen = Label(
            self.root,
            background="White",
            text="Actual screen output",
            font=("profont", 29),
        )
        self.label_screen.grid(row=0, column=1, pady=30)

        self.screen = Canvas(self.root, width=600, height=400)
        self.screen.grid(row=1, column=1)

        self.label_vae = Label(
            self.root, background="White", text="View through VAE", font=("profont", 29)
        )
        self.label_vae.grid(row=2, column=1, pady=30)

        self.vae_screen = Canvas(self.root, width=600, height=400)
        self.vae_screen.grid(row=3, column=1, rowspan=8)

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

    def handle_key_press(self, event):
        self.event_q.put(event.keysym)
        if event.keysym == "q":
            self.root.destroy()

    def update(self):
        try:
            (
                real_screen,
                small_screen,
                z,
                vae_screen,
                mdn_rnn_screen,
                actual_action,
            ) = self.value_q.get(block=False, timeout=0.05)
        except queue.Empty:
            pass
        else:
            self.image = ImageTk.PhotoImage(
                transforms.ToPILImage(mode="RGB")(real_screen)
            )
            self.screen.itemconfig(self.screen_obj_actual, image=self.image)

            self.image_vae = ImageTk.PhotoImage(
                transforms.ToPILImage(mode="RGB")(vae_screen[0]).resize((600, 400))
            )
            self.vae_screen.itemconfig(self.vae_screen_obj_actual, image=self.image_vae)

            steering, gas, break_ = actual_action
            self.action_gas["value"] = gas * 100
            self.action_break["value"] = break_ * 100
            if steering < 0:
                self.action_left["value"] = abs(steering) * 100
                self.action_right["value"] = 0
            else:
                self.action_right["value"] = abs(steering) * 100
                self.action_left["value"] = 0

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
        pause_game = True

        while True:
            if quit:
                break

            player_generator = player.generator()
            while True:

                if not pause_game:
                    try:
                        i = next(player_generator)
                    except StopIteration:
                        break
                    else:
                        self.value_q.put(i)

                try:
                    event = self.event_q.get(block=False, timeout=0.05)
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
                    elif event == "space":
                        pause_game = not pause_game
        logger.info("stopping xvfb")
        vdisplay.stop()


class Demo(Step):
    hyperparams_key = None

    def __call__(self):
        gui = GUI(self)
        gui.gui()
