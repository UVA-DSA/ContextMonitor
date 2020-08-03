Code for Real-Time Context-aware Detection of Unsafe Events in Robot-Assisted Surgery.

To install dependencies, use pip install -r requirements.txt. After that, install tensorflow==1.14.0 (version used in our experiments).

All models were trained and evaluated using the Leave One SuperTrialOut setup.

To train the gesture classification module, run "python3 experimental_setup.py 0"

To train the anomaly detection, run losorelabelledSuboptimals.py
