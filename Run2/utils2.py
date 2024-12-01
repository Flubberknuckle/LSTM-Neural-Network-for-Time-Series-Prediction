from datetime import datetime

class Timer:
    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = datetime.now()

    def stop(self):
        end_dt = datetime.now()
        print(f'Time taken: {end_dt - self.start_dt}')
