import numpy as np

class CostEstimator:
    def __init__(self, model, tfidf, borough_map, jobtype_map, cluster_map):
        self.model = model
        self.tfidf = tfidf
        self.borough_map = borough_map
        self.jobtype_map = jobtype_map
        self.cluster_map = cluster_map

    def predict(self, job_text, borough, job_type, cluster_label, year):
        tfidf_vector = self.tfidf.transform([job_text]).toarray()
        struct = np.array([[self.borough_map[borough],
                            self.jobtype_map[job_type],
                            self.cluster_map[cluster_label],
                            year]])
        combined = np.hstack((struct, tfidf_vector))
        return self.model.predict(combined)[0]
