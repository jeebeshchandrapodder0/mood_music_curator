import logging
import os

class MoodMusicLogger:
    _logger = None

    @staticmethod
    def get_logger():
        if MoodMusicLogger._logger is None:
            # Ensure the logs directory exists
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
            os.makedirs(log_dir, exist_ok=True)

            # Create and configure the logger
            MoodMusicLogger._logger = logging.getLogger('mood_music_curator')
            MoodMusicLogger._logger.setLevel(logging.INFO)

            # Create a file handler for app.log
            log_file = os.path.join(log_dir, 'app.log')
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)

            # Define the log format
            formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
            handler.setFormatter(formatter)

            # Add the handler to the logger
            MoodMusicLogger._logger.addHandler(handler)

        return MoodMusicLogger._logger