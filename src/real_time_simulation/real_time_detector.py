import time
from collections import deque

class RealTimeDetector:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.buffer = deque(maxlen=config['buffer_size'])

    def process_sample(self, sample):
        """
        Process a single sample in real-time.
        
        :param sample: A single traffic sample
        :return: True if anomaly detected, False otherwise
        """
        # Add the sample to the buffer
        self.buffer.append(sample)
        
        # If the buffer is full, make a prediction
        if len(self.buffer) == self.config['buffer_size']:
            X = pd.DataFrame(list(self.buffer))
            prediction = self.model.predict(X)
            
            # Return True if the latest sample is predicted as an anomaly
            return prediction[-1] == 1
        
        return False

    def run_detection(self, traffic_stream):
        """
        Run real-time detection on a stream of traffic data.
        
        :param traffic_stream: Generator yielding traffic samples
        """
        for sample in traffic_stream:
            is_anomaly = self.process_sample(sample)
            
            if is_anomaly:
                print(f"Anomaly detected at {time.time()}!")
            
            # Additional processing or logging can be added here