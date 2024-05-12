class ScheduledMovement:
    def __init__(self, time, tiploc, predicted_delay=None, actual_delay=None):
        self.tiploc = tiploc
        self.time = time
        self.predicted_delay = predicted_delay
        self.actual_delay = actual_delay

    def add_predicted_delay(self, delay):
        if self.predicted_delay is None:
            self.predicted_delay = delay
        else:
            raise OverflowError("Predicted delay already set")

    def add_actual_delay(self, delay):
        if self.actual_delay is None:
            self.actual_delay = delay
        else:
            raise OverflowError("Actual delay already set")

    def get_feature_set(self):
        # TODO: Get predicted weather at location at schedule time
        # TODO: Return formatted list of features for prediction/training
        pass
