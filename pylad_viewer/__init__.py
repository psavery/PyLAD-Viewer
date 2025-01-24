import signal

# Kill the program when ctrl-c is used
signal.signal(signal.SIGINT, signal.SIG_DFL)
