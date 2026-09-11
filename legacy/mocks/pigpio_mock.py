class PiMock:
    def __init__(self): self.connected = True
    def set_mode(self, *_args, **_kwargs): pass
    def write(self, *_args, **_kwargs): pass
    def hardware_PWM(self, *_args, **_kwargs): pass
    def stop(self): pass

def pi(): return PiMock()
