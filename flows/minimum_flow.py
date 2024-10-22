"""
This flow is to check that everything is working correctly.

It include an artifact (i.e. `self.data`) to verify that it shows in the Metaflow UI.
"""
from metaflow import FlowSpec, step


class MinimumFlow(FlowSpec):
    @step
    def start(self):
        self.data = 123
        self.next(self.end)

    @step
    def end(self):
        print("Data from start step", self.data)
        print("Flow is done!")


if __name__ == "__main__":
    MinimumFlow()
