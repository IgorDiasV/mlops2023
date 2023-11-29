class Pipeline:
    def __init__(self):
        self.tasks = []

    def task(self, depends_on=None):
        idx = 0
        if depends_on:
            idx = self.tasks.index(depends_on) + 1

        def inner(f):
            self.tasks.insert(idx, f)
            return f
        return inner

    def run(self, input_=None):
        output = input_
        for task in self.tasks:
            if output:
                output = task(output)
            else:
                output = task()
        return output