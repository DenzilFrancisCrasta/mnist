''' Logging Module with Formatters '''

class Logger(object):
    def __init__(self, filename, formatter):
        self.filename = filename
        self.formatter = formatter
        self.f = open(filename, 'w')

    def log(self, msg_arg_list):
        self.f.write(self.formatter.format(msg_arg_list))

class NullLogger(object):
    ''' Mock object that does nothing '''
    def __init__(self, filename, formatter):
        pass
    def log(self, msg_arg_list):
        pass
        
class Formatter(object):
    def __init__(self, template):
        self.template = template

    def format(self, msg_arg_list):
        return self.template.format(*msg_arg_list)


if  __name__ == "__main__":
    error_formatter = Formatter("Epoch {}, Step {}, Loss: {}, lr: {}")
    error_logger = Logger('log_loss.txt', error_formatter)
    msg = [2, 3, 3.3, 0.5]
    error_logger.log(msg)
