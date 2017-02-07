''' MNIST Digit Classification Application '''
from mnist import MnistApplication

# Bootstrap the application
app = MnistApplication()
app.initialize_switches()

args = app.parser.parse_args()
print(args)
