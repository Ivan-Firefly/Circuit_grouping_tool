import numpy as np
import pandas as pd
import ezdxf
from scipy.spatial import distance_matrix
from itertools import permutations
import io


def read_csv(filename):
    df = pd.read_csv(filename)
    grouped_coords = df.groupby(["tag", "Circuits"])[['x', 'y']].apply(lambda g: g.values.tolist()).to_dict()
    return grouped_coords


def solve_tsp(coords):
    n = len(coords)
    dist_matrix = distance_matrix(coords, coords)

    best_path = None
    best_length = float('inf')

    for perm in permutations(range(n)):
        length = sum(dist_matrix[perm[i], perm[i + 1]] for i in range(n - 1))
        if length < best_length:
            best_length = length
            best_path = perm

    return [coords[i] for i in best_path]


def split_path(path, segments):
    n = len(path)
    if n < 2:
        raise ValueError("At least two points are required to form a path.")

    total_length = sum(np.linalg.norm(np.array(path[i + 1]) - np.array(path[i])) for i in range(n - 1))
    segment_length = total_length / segments

    new_points = [path[0]]
    current_length = 0
    last_point = np.array(path[0])

    for i in range(1, n):
        next_point = np.array(path[i])
        while current_length + np.linalg.norm(next_point - last_point) >= segment_length * len(new_points):
            ratio = (segment_length * len(new_points) - current_length) / np.linalg.norm(next_point - last_point)
            new_point = last_point + ratio * (next_point - last_point)
            new_points.append(new_point.tolist())
        current_length += np.linalg.norm(next_point - last_point)
        last_point = next_point

    return new_points[:segments]


def create_dxf(segment_starts):
    doc = ezdxf.new()
    msp = doc.modelspace()

    block = doc.blocks.new(name="SegmentMarker")
    block.add_circle(center=(0, 0), radius=5)

    # Add attributes to the block
    block.add_attdef("Line_ID", dxfattribs={"insert": (0, 1), "height": 1})
    block.add_attdef("Circuit", dxfattribs={"insert": (0, -1), "height": 1})

    for (tag, circuits), points in segment_starts.items():
        for idx, point in enumerate(points):
            x, y = point
            block_ref = msp.add_blockref("SegmentMarker", insert=(x, y))
            block_ref.add_attrib("Line_ID", f"{tag}")
            block_ref.add_attrib("Circuit", f"{idx + 1}")

    doc.saveas("output.dxf")

    text_buffer = io.StringIO()
    doc.write(text_buffer)
    text_data = text_buffer.getvalue().encode("utf-8")
    dxf_buffer = io.BytesIO(text_data)
    return dxf_buffer


def start(uploaded_file):
    grouped_coords = read_csv(uploaded_file)
    segment_starts = {}

    for (tag, circuits), coords in grouped_coords.items():
        ordered_path = solve_tsp(coords)
        try:
            segment_starts[(tag, circuits)] = split_path(ordered_path, segments=circuits)
        except ValueError as e:
            print(f"Skipping {tag}: {e}")

    return create_dxf(segment_starts)  # Return binary DXF content

# start("1.csv")
