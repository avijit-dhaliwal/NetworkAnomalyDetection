import pandas as pd
import numpy as np

class TrafficSimulator:
    def __init__(self, nsl_kdd_df, cicids2017_df, config):
        self.nsl_kdd_df = nsl_kdd_df
        self.cicids2017_df = cicids2017_df
        self.config = config

    def generate_traffic(self, num_samples):
        # Randomly sample from both datasets
        nsl_kdd_samples = self.nsl_kdd_df.sample(n=num_samples // 2, replace=True)
        cicids2017_samples = self.cicids2017_df.sample(n=num_samples // 2, replace=True)
        
        # Combine samples from both datasets
        combined_samples = pd.concat([nsl_kdd_samples, cicids2017_samples], ignore_index=True)
        
        # Shuffle the combined samples to mix the traffic
        combined_samples = combined_samples.sample(frac=1).reset_index(drop=True)
        
        return combined_samples

    def generate_stream(self, duration_seconds, samples_per_second):
        """
        Generate a stream of traffic data over a specified duration.
        
        :param duration_seconds: Duration of the stream in seconds
        :param samples_per_second: Number of samples to generate per second
        :return: Generator yielding traffic samples
        """
        total_samples = duration_seconds * samples_per_second
        
        # Generate all samples at once
        all_samples = self.generate_traffic(total_samples)
        
        # Yield samples over time
        for i in range(total_samples):
            yield all_samples.iloc[i]
            
            # Simulate delay between samples
            time.sleep(1 / samples_per_second)