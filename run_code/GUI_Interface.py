import tkinter as tk
from tkinter import filedialog
from playsound import playsound
import matplotlib.pyplot as plt
import os,sys
import wave
import pyaudio
sys.path.append('../software/models/')
import utilFunctions as UF
import subprocess
import main
import verify as VF


class SoundPlayerApp:
    def __init__(self, master):
        self.master = master
        master.title("Graphical User Interface ")
        master.option_add("*Font", "Helvetica 12")  # Set font size for all text

        self.file_label = tk.Label(master, text="Select an input file:")
        self.file_label.grid(row=0, column=0)

        self.file_var = tk.StringVar()
        self.file_dropdown = tk.OptionMenu(master, self.file_var, "")
        self.file_dropdown.grid(row=1, column=1)

        self.browse_button = tk.Button(master, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=1, column=2)

        self.listen_button = tk.Button(master, text="Listen", command=self.start_sound)
        self.listen_button.grid(row=1, column=3)

        self.pause_button = tk.Button(master, text="Pause", command=self.pause_sound, state=tk.DISABLED)
        self.pause_button.grid(row=1, column=4)

        self.file_label = tk.Label(master, text="Run Main Code:")
        self.file_label.grid(row=2, column=0)

        self.run_main_button = tk.Button(master, text="Run", command=self.run_main)
        self.run_main_button.grid(row=2, column=1, columnspan=3)

        self.file_label_1 = tk.Label(master, text="Listen Output Sound ")
        self.file_label_1.grid(row=3, column=0)

        self.listen_button_2 = tk.Button(master, text="Listen", command=self.start_output_sound)
        self.listen_button_2.grid(row=3, column=1, columnspan = 3)

        self.file_label_2 = tk.Label(master, text="Verify Input Output ")
        self.file_label_2.grid(row=4, column=0)

        self.run_verify_button = tk.Button(master, text="Verify", command=self.run_verify)
        self.run_verify_button.grid(row=4, column=1, columnspan=3)

        self.playing = False
        self.paused = False

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Wave files", "*.wav")])
        if filename:
            self.file_var.set(os.path.basename(filename))
            self.filename = filename

    def start_sound(self):
        if hasattr(self, 'filename') and not self.playing:
            self.playing = True
            self.listen_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.play_sound()

    def start_output_sound(self):
        if hasattr(self, 'filename') and not self.playing:
            self.playing = True
            self.listen_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.play_sound_out()

    def play_sound(self):
        playsound(self.filename)

    def play_sound_out(self):
        outputFile = os.path.basename(self.filename)[:-4] + '_sineModel.wav'
        playsound(outputFile)


    def pause_sound(self):
        self.paused = True

    def run_main(self):
        if hasattr(self, 'filename'):
            main.main(self.filename)

    def run_verify(self):
        if hasattr(self, 'filename'):
            inputFile = self.filename
            outputFile = os.path.basename(inputFile)[:-4] + '_sineModel.wav'
            (fs, signal1) = UF.wavread(inputFile)
            (fs, signal2) = UF.wavread(outputFile)

            mse = VF.MSE(signal1, signal2)

            plt.figure(figsize=(6,6))

            sizes = [100 - mse, mse]
            labels = ['Similarity', 'Error']
            colors = ['blue', 'red']
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
            plt.title('MSE')

            plt.show()
root = tk.Tk()
root.geometry("600x400")
app = SoundPlayerApp(root)
root.mainloop()
## edited before running output file
