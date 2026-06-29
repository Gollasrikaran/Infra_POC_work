# """
# shoelace_calculator.py
# ======================
# Deterministic civil-engineering area calculator using the Shoelace
# (Gauss) Formula.

# Why this exists:
#   The LLM is excellent at *reading* coordinates from a cross-section
#   drawing, but prone to arithmetic errors when it also tries to
#   *calculate* the enclosed area (especially if it misreads one vertex).

#   Solution from shoelace.txt:
#     "Let the LLM only extract the coordinates as text, and then pass
#      those coordinates to a real Python math script."

# Formula:
#     Area = ½ × |Σ(xᵢ × yᵢ₊₁) − Σ(yᵢ × xᵢ₊₁)|

# Reference verification (from shoelace.txt):
#     Vertices: [(-53.87, 656.00), (-40.00, 652.73), (-35.00, 656.00)]
#     Expected area: 30.85 sq ft  ✓
# """


# def calculate_cross_section_area(coordinates: list[tuple]) -> float:
#     """
#     Calculate the area of a polygon using the Shoelace / Gauss formula.

#     Parameters
#     ----------
#     coordinates : list of (x, y) tuples
#         Ordered vertices tracing the perimeter of the cross-section
#         region (clockwise or counter-clockwise — result is the same).
#         Units must be consistent (e.g. feet).

#     Returns
#     -------
#     float
#         Area in the same square-unit as the input coordinates,
#         rounded to 2 decimal places.
#         Returns 0.0 if fewer than 3 vertices are supplied.

#     Example
#     -------
#     >>> coords = [(-53.87, 656.00), (-40.00, 652.73), (-35.00, 656.00)]
#     >>> calculate_cross_section_area(coords)
#     30.85
#     """
#     n = len(coordinates)
#     if n < 3:
#         return 0.0

#     sum1 = 0.0  # Σ xᵢ × yᵢ₊₁  (down-right diagonal)
#     sum2 = 0.0  # Σ yᵢ × xᵢ₊₁  (down-left diagonal)

#     for i in range(n):
#         x1, y1 = coordinates[i]
#         x2, y2 = coordinates[(i + 1) % n]   # wraps back to index 0 at end
#         sum1 += x1 * y2
#         sum2 += y1 * x2

#     area = 0.5 * abs(sum1 - sum2)
#     return round(area, 2)


# # ---------------------------------------------------------------------------
# # Quick self-test (runs only when executed directly, not when imported)
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Example from shoelace.txt — expected 30.85 sq ft
#     test_coords = [(-53.87, 656.00), (-40.00, 652.73), (-35.00, 656.00)]
#     result = calculate_cross_section_area(test_coords)
#     print(f"Test area: {result} sq ft  (expected 30.85)")
#     assert result == 30.85, f"FAIL: got {result}"
#     print("OK - Shoelace formula verified.")


def calculate_cross_section_area(coordinates: list) -> float:
    # 1. Sanitize input: Ensure all coordinates are float tuples, even if LLM sent strings or lists
    try:
        cleaned_coords = [(float(pt[0]), float(pt[1])) for pt in coordinates]
    except (ValueError, TypeError, IndexError):
        return 0.0  # Return 0 if the LLM payload is completely malformed

    n = len(cleaned_coords)
    if n < 3:
        return 0.0

    sum1 = 0.0  
    sum2 = 0.0  

    for i in range(n):
        x1, y1 = cleaned_coords[i]
        x2, y2 = cleaned_coords[(i + 1) % n]   
        sum1 += x1 * y2
        sum2 += y1 * x2

    area = 0.5 * abs(sum1 - sum2)
    return round(area, 2)


'''
1. Input Safety FilterWhat it does: The code checks if len(coordinates) < 3.The Explanation: A valid geometric cross-section (polygon) must have at least three points to enclose an area (a triangle). If the data payload is corrupted or contains fewer than three points, the script safely returns 0.0 instead of crashing the application pipeline.2. The Shoelace Matrix LoopWhat it does: The script initializes sum1 and sum2, then loops through the coordinates using the modulo operator (i + 1) % n.The Explanation: This is the core of Gauss's Area Formula. The loop performs cross-multiplication across the coordinate arrays. Using the modulo operator % n is a key optimization—when the loop reaches the very last vertex, it automatically wraps around and multiplies it back by the first vertex. This closes the boundary loop without requiring the LLM to provide a duplicate closing coordinate.3. Absolute Scaling and FormattingWhat it does: It takes the difference, halves it using 0.5 * abs(), and rounds it to two decimal places.The Explanation: Civil engineering drawings can be traced clockwise or counter-clockwise. Clockwise tracing yields a negative result in matrix math. The abs() function strips the sign, ensuring the area is always a true, positive scalar quantity. Finally, rounding to two decimal places standardizes the output for construction estimation sheets.
'''