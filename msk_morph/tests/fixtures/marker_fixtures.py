"""
marker_fixtures.py
==================
Golden landmark coordinates for test_pipeline_e2e.py (participant "test").
EXPECTED_MARKERS maps each marker name to its [x, y, z] location (metres).

Regenerate after an intentional change:  pytest -m real --update-golden
"""

EXPECTED_MARKERS = {
    'rASIS': [-0.096047, -0.05536, 0.125012],
    'lASIS': [0.099439, -0.058304, 0.124141],
    'midASIS': [0.001696, -0.056832, 0.124577],
    'rPSIS': [-0.052406, 0.084418, 0.158693],
    'lPSIS': [0.051152, 0.081181, 0.158165],
    'midPSIS': [-0.000627, 0.082799, 0.158429],
    'mid_midASISmidPSIS': [0.000535, 0.012983, 0.141503],
    'rHJC': [-0.088343, 0.010063, 0.063683],
    'rFLE': [-0.096271, -0.028897, -0.341615],
    'rFME': [-0.017424, -0.005508, -0.334455],
    'midrFE': [-0.056847, -0.017203, -0.338035],
    'lHJC': [0.088409, 0.008059, 0.062137],
    'lFLE': [0.118887, -0.01642, -0.343891],
    'lFME': [0.037318, -0.012494, -0.337824],
    'midlFE': [0.078102, -0.014457, -0.340858],
    'midHJC': [3.3e-05, 0.009061, 0.06291],
}
