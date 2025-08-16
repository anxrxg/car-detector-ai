# src/color_ranges.py

COLOR_RANGES = {
    "red": {
        "lower1": (0, 100, 100), "upper1": (10, 255, 255),
        "lower2": (170, 100, 100), "upper2": (180, 255, 255)
    },
    "blue": {
        "lower": (100, 150, 0), "upper": (140, 255, 255)
    },
    "green": {
        "lower": (40, 40, 40), "upper": (80, 255, 255)
    },
    "white": {
        "lower": (0, 0, 200), "upper": (180, 25, 255) # Adjust V for brightness
    },
    "black": {
        "lower": (0, 0, 0), "upper": (180, 255, 30) # Adjust V for darkness
    },
    "silver": { # Often a range of grays
        "lower": (0, 0, 150), "upper": (180, 15, 220)
    },
    "yellow": {
        "lower": (20, 100, 100), "upper": (30, 255, 255)
    },
    "orange": {
        "lower": (10, 100, 100), "upper": (20, 255, 255)
    }
}
