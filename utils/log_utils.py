import time


class ZYLog:
    def __init__(self, logging_file):
        self.logging_file = logging_file

    def write_log(self, text):
        with open(self.logging_file, 'a') as f:
            f.write("{}\t".format(time.strftime("%m-%d-%H-%M")) + str(text))
            f.write('\n')

    def write_config(self, config: dict):
        with open(self.logging_file, 'a') as f:
            f.write("\nConfig's Content\n")
            for key, value in config.items():
                f.write("{}: {}".format(key, value))
                f.write('\n')
            f.write('\n')
