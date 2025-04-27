import tkinter as tk
from src.moodMusicCurator.components.facial_analysis import FacialAnalysisGUI
from src.moodMusicCurator.logging import MoodMusicLogger

def main():
    logger = MoodMusicLogger.get_logger()
    logger.info("Starting Mood Music Curator Facial Analysis GUI")

    root = tk.Tk()
    app = FacialAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()