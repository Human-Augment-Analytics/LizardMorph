
# Average Landmark Positions for Toe and Finger
# Used to determine target standard orientation

# Consolidated Finger Data (Standard: Pointing LEFT)
# LM 1 (Base: 4861) -> LM 9 (Tip: 4556) => Decreasing X => Left
FINGER_AVERAGES = [
    {"x": 4861.21951, "y": 2358.06829, "name": "1"}, 
    {"x": 4860.76098, "y": 2294.48293, "name": "2"},
    {"x": 4767.73659, "y": 2318.25366, "name": "3"},
    {"x": 4773.98537, "y": 2288.21951, "name": "4"},
    {"x": 4663.99512, "y": 2312.80000, "name": "5"},
    {"x": 4676.52195, "y": 2254.64390, "name": "6"},
    {"x": 4630.79512, "y": 2284.95122, "name": "7"},
    {"x": 4634.75122, "y": 2264.60976, "name": "8"},
    {"x": 4556.61951, "y": 2259.46341, "name": "9"},
]

# Consolidated Toe Data (Standard: Pointing UP)
# LM 1 (Base: 1965 Y) -> LM 9 (Tip: 1515 Y) => Decreasing Y (Top of Image) => Up
TOE_AVERAGES = [
    {"x": 6745.44949, "y": 1965.93434, "name": "1"},
    {"x": 6806.57576, "y": 2001.46465, "name": "2"},
    {"x": 6705.78788, "y": 1736.08586, "name": "3"},
    {"x": 6733.06061, "y": 1726.51010, "name": "4"},
    {"x": 6649.31313, "y": 1640.45455, "name": "5"},
    {"x": 6708.77778, "y": 1619.31313, "name": "6"},
    {"x": 6655.16162, "y": 1591.15152, "name": "7"},
    {"x": 6672.93939, "y": 1585.62626, "name": "8"},
    {"x": 6641.67172, "y": 1515.94444, "name": "9"},
]
