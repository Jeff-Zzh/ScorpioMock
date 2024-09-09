import os

print(os.getcwd())
print(os.path.dirname(os.getcwd()))
log_dir = os.path.join(os.path.dirname(os.getcwd()),'log')
print(log_dir)

class MyClass:
    def __init__(self):
        self.attr1 = 1

myclass = MyClass()
print(type(myclass))
print(type(myclass).__name__)