import pandas as pd
import numpy as np
import ezdxf
import math

def manhattan_distance(point1, point2):
    """Calculate Manhattan distance between two points"""
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1]) + abs(point1[2] - point2[2])


def radius_search(center_point, remaining_circuits, control_method, available_current, current_radius, step,
                  max_iterations):
    """Search for the next circuit within increasing radius using vectorized operations"""
    search_radius = current_radius

    # Filter circuits with the same control method and current <= available_current
    valid_circuits = remaining_circuits[
        (remaining_circuits['Control method'] == control_method) &
        (remaining_circuits['Circuit CB Current'] <= available_current)
        ].copy()  # Use explicit .copy() to avoid SettingWithCopyWarning

    if valid_circuits.empty:
        return None

    # Extract coordinates for vectorized distance calculation
    x0, y0, z0 = center_point

    for _ in range(max_iterations):
        # Vectorized distance calculation - using .loc to avoid SettingWithCopyWarning
        valid_circuits.loc[:, 'distance'] = valid_circuits.apply(
            lambda row: manhattan_distance(
                (x0, y0, z0),
                (row['North_X'], row['East_Y'], row['Elevation_Z'])
            ), axis=1
        )

        # Filter circuits within radius
        candidates = valid_circuits[valid_circuits['distance'] <= search_radius]

        if not candidates.empty:
            # Find circuit with smallest current
            return candidates.sort_values(by='Circuit CB Current').iloc[0]

        # If no suitable circuits found, increase radius
        search_radius += step

    # If nothing found after max_iterations, return None
    return None


def find_equidistant_point(circuits_group):
    """Calculate equidistant point for a group of circuits"""
    return np.mean(circuits_group[['North_X', 'East_Y', 'Elevation_Z']].values, axis=0)


def calculate_distances(circuits_group, chain_type):
    """Calculate distances based on chain type"""
    equidistant_point = find_equidistant_point(circuits_group)

    if chain_type == 'MB':
        # Distance from each circuit to equidistant point - vectorized
        # Using a temporary column to store distances
        circuits_group = circuits_group.copy()  # Create a copy to avoid warnings
        circuits_group.loc[:, 'Distance'] = circuits_group.apply(
            lambda row: manhattan_distance(
                (row['North_X'], row['East_Y'], row['Elevation_Z']),
                equidistant_point
            ), axis=1
        )

        return circuits_group['Distance'].tolist(), equidistant_point

    elif chain_type == 'daisy':
        if len(circuits_group) == 1:
            return [0], equidistant_point

        # Calculate sum of distances between adjacent circuits
        coords = circuits_group[['North_X', 'East_Y', 'Elevation_Z']].values
        total_distance = sum(manhattan_distance(coords[i], coords[i + 1]) for i in range(len(coords) - 1))

        # Each circuit gets the same total distance
        return [total_distance] * len(circuits_group), equidistant_point


def group_circuits(input_file, chain_type, init_radius, radius_step,
                   max_iterations, max_current_per_group, max_circuit_per_group):
    """Main function to group circuits based on proximity and constraints"""
    # Read input file
    df = pd.read_excel(input_file)

    # Create working copy of the dataframe
    working_df = df.copy()
    working_df['assigned'] = False

    # Sort by Circuit CB Current in descending order
    working_df = working_df.sort_values(by='Circuit CB Current', ascending=False)

    # Add columns for grouping results
    result_df = df.copy()
    result_df['Group'] = None
    result_df['Sequence_In_Group'] = None
    result_df['Distance'] = None
    result_df['Equidistant_X'] = None
    result_df['Equidistant_Y'] = None
    result_df['Equidistant_Z'] = None

    # Start grouping
    group_number = 1

    while not working_df[working_df['assigned'] == False].empty:
        # Find unassigned circuit with max current
        unassigned_df = working_df[working_df['assigned'] == False]

        if len(unassigned_df) == 0:
            break

        # Start a new group with the circuit having highest current
        start_circuit = unassigned_df.iloc[0]
        start_idx = start_circuit.name
        control_method = start_circuit['Control method']

        group_circuits = [start_idx]
        working_df.loc[start_idx, 'assigned'] = True  # Use .loc to avoid warnings

        # Initialize available current for this group
        available_current = max_current_per_group - start_circuit['Circuit CB Current']

        # If chain_type is MB, use first circuit as reference point
        # If chain_type is daisy, reference point will be updated with each addition
        reference_point = (start_circuit['North_X'], start_circuit['East_Y'], start_circuit['Elevation_Z'])

        # Keep adding circuits until constraints are met
        while len(group_circuits) < max_circuit_per_group and available_current > 0:
            # Find remaining circuits that haven't been assigned yet
            remaining_circuits = working_df[working_df['assigned'] == False]

            if remaining_circuits.empty:
                break

            # Find next suitable circuit using vectorized radius search
            next_circuit = radius_search(
                reference_point,
                remaining_circuits,
                control_method,
                available_current,
                init_radius,
                radius_step,
                max_iterations
            )

            if next_circuit is None:
                break

            next_idx = next_circuit.name

            # Add circuit to group
            group_circuits.append(next_idx)
            working_df.loc[next_idx, 'assigned'] = True  # Use .loc to avoid warnings

            # Update available current
            available_current -= next_circuit['Circuit CB Current']

            # If daisy chain, update reference point to the last added circuit
            if chain_type == 'daisy':
                reference_point = (next_circuit['North_X'], next_circuit['East_Y'], next_circuit['Elevation_Z'])

            # If MB type, keep the first circuit as reference
            # (reference_point already set above)

        # Calculate equidistant point and distances for this group
        group_df = working_df.loc[group_circuits].copy()  # Create a copy to avoid warnings
        distances, equidistant_point = calculate_distances(group_df, chain_type)

        # Update result dataframe with group information
        for i, idx in enumerate(group_circuits):
            result_df.loc[idx, 'Group'] = group_number
            result_df.loc[idx, 'Sequence_In_Group'] = i + 1  # Add sequence in group
            result_df.loc[idx, 'Distance'] = distances[i]
            result_df.loc[idx, 'Equidistant_X'] = equidistant_point[0]
            result_df.loc[idx, 'Equidistant_Y'] = equidistant_point[1]
            result_df.loc[idx, 'Equidistant_Z'] = equidistant_point[2]

        # Increment group number
        group_number += 1

    # Save intermediate result to intermediate file
    # result_df.to_excel(intermediate_file, index=False)

    return result_df


# ====================== REMOTE INDEX PROCESSING BASED ON CLOSEST DISTANCES ====================== #

def find_closest_distances(equidistant_points):
    """
    Find the closest distance to another equidistant point for each point
    Returns a dictionary mapping point index to (closest_point_index, distance)
    """
    n = len(equidistant_points)
    closest_distances = {}

    for i in range(n):
        min_dist = float('inf')
        closest_idx = -1

        for j in range(n):
            if i != j:  # Don't compare a point with itself
                dist = manhattan_distance(equidistant_points[i], equidistant_points[j])
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = j

        closest_distances[i] = (closest_idx, min_dist)

    return closest_distances


def assign_remote_indices_by_distance(closest_distances):
    """
    Assign remote indices based on distance to the closest group
    Group with largest distance gets index 0, second largest gets 1, etc.
    """
    # Extract just the distances
    distances = [(idx, info[1]) for idx, info in closest_distances.items()]

    # Sort by distance in descending order
    sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)

    # Assign remote indices
    remote_indices = {}
    for remote_idx, (idx, _) in enumerate(sorted_distances):
        remote_indices[idx] = remote_idx

    return remote_indices


def process_groups_with_distances(result_df):
    """Process groups and assign remote indices based on distance to the closest group"""
    # Group the DataFrame by Group
    groups = result_df['Group'].unique()

    # Extract equidistant points for each group
    equidistant_points = []
    idx_to_group = {}  # Maps index in equidistant_points list to group number
    group_to_idx = {}  # Maps group number to index in equidistant_points list

    for i, group_num in enumerate(groups):
        group_data = result_df[result_df['Group'] == group_num].iloc[0]
        equidistant_points.append((
            group_data['Equidistant_X'],
            group_data['Equidistant_Y'],
            group_data['Equidistant_Z']
        ))
        idx_to_group[i] = group_num
        group_to_idx[group_num] = i

    # Calculate closest distances
    closest_distances = find_closest_distances(equidistant_points)

    # Assign remote indices based on distances
    remote_indices_by_idx = assign_remote_indices_by_distance(closest_distances)

    # Map from group numbers to remote indices and closest distances
    remote_index_mapping = {}
    closest_distance_mapping = {}
    closest_group_mapping = {}

    for group_num in groups:
        idx = group_to_idx[group_num]
        remote_index_mapping[group_num] = remote_indices_by_idx[idx]

        closest_idx, distance = closest_distances[idx]
        closest_group = idx_to_group[closest_idx]

        closest_distance_mapping[group_num] = distance
        closest_group_mapping[group_num] = closest_group

    # Add columns to result DataFrame
    result_df['Remote_Index'] = result_df['Group'].map(remote_index_mapping)
    # result_df['Closest_Distance'] = result_df['Group'].map(closest_distance_mapping)
    # result_df['Closest_Group'] = result_df['Group'].map(closest_group_mapping)
    #
    # Save updated result to intermediate file
    # result_df.to_excel('index.xlsx', index=False)

    return result_df, remote_index_mapping, group_to_idx, equidistant_points

# ====================== PANEL ASSIGNMENT PROCESSING ====================== #

def find_equidistant_between_points(points):
    """Calculate equidistant point between multiple points"""
    if not points:
        return None

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    z_coords = [p[2] for p in points]

    return (np.mean(x_coords), np.mean(y_coords), np.mean(z_coords))


def check_group_distance_to_panel(group_eq_point, panel_point, group_max_distance, max_distance_limit):
    """Check if the distance between group and panel is within limits"""
    dist = manhattan_distance(group_eq_point, panel_point)
    total_dist = dist + group_max_distance
    return total_dist <= max_distance_limit, total_dist


def assign_groups_to_panels(result_df, output_file, sections_per_panel, max_outgoings_per_section,
                            max_distance_limit, panel_type):
    """Assign groups to panels with multiple sections based on rules"""
    # Working copy of the dataframe
    df = result_df.copy()

    # Add panel-related columns
    df['Panel'] = None
    df['Section'] = None
    df['Outgoing'] = None
    df['Panel_X'] = None
    df['Panel_Y'] = None
    df['Panel_Z'] = None
    df['Distance_To_Panel'] = None

    # Get unique groups and their information
    groups = df['Group'].unique()
    group_info = {}
    for group_num in groups:
        group_data = df[df['Group'] == group_num].iloc[0]
        group_info[group_num] = {
            'remote_index': group_data['Remote_Index'],
            'control_method': group_data['Control method'],
            'eq_point': (group_data['Equidistant_X'], group_data['Equidistant_Y'], group_data['Equidistant_Z']),
            'max_distance': df[df['Group'] == group_num]['Distance'].max(),
            'assigned': False
        }

    # Sort groups by remote index
    sorted_groups = sorted(group_info.keys(), key=lambda g: group_info[g]['remote_index'])

    panel_number = 1

    while True:
        # Find the first unassigned group with lowest remote index
        unassigned_groups = [g for g in sorted_groups if not group_info[g]['assigned']]
        if not unassigned_groups:
            break

        # Start a new panel with this group
        start_group = unassigned_groups[0]
        start_control_method = group_info[start_group]['control_method']
        panel_point = group_info[start_group]['eq_point']

        # Initialize panel tracking
        panel_groups = []
        section_number = 1
        outgoing_number = 1
        current_section_control_method = start_control_method
        section_groups = []

        # Mark first group as assigned
        group_info[start_group]['assigned'] = True
        panel_groups.append((start_group, section_number, outgoing_number))
        section_groups.append(start_group)

        # Continuously try to assign more groups to this panel
        while section_number <= sections_per_panel:
            # Find closest unassigned group with compatible control method
            compatible_groups = []

            for group in unassigned_groups:
                if group_info[group]['assigned']:
                    continue
                outgoing_number_candidate=outgoing_number+1
                # Check control method compatibility
                group_control_method = group_info[group]['control_method']

                # For locked panel type, control method must match first section
                if panel_type == 'locked' and group_control_method != start_control_method:
                    continue

                # For free panel type, control method must match current section
                if group_control_method != current_section_control_method:
                    # Only move to different control method if starting a new section
                    if outgoing_number_candidate >= 1:
                        continue

                # Check distance limitation
                group_eq_point = group_info[group]['eq_point']
                group_max_distance = group_info[group]['max_distance']

                is_within_limit, total_dist = check_group_distance_to_panel(
                    group_eq_point, panel_point, group_max_distance, max_distance_limit
                )

                if is_within_limit:
                    compatible_groups.append((group, total_dist))

            if not compatible_groups:
                # No compatible groups found for this section
                if outgoing_number == 1:
                    # If no groups were added to this section, move to next section
                    section_number += 1
                    outgoing_number = 1

                    # Reset section groups
                    section_groups = []

                    # For free panel type, we can change control method in the new section
                    if panel_type == 'free' and section_number <= sections_per_panel:
                        # Try to find unassigned group with different control method
                        alternative_methods = set(group_info[g]['control_method'] for g in unassigned_groups
                                                  if not group_info[g]['assigned'])

                        # Exclude current section control method if possible
                        if current_section_control_method in alternative_methods and len(alternative_methods) > 1:
                            alternative_methods.remove(current_section_control_method)

                        if alternative_methods:
                            current_section_control_method = list(alternative_methods)[0]
                    continue
                else:
                    # If some groups were added but no more can be found, move to next section
                    section_number += 1
                    outgoing_number = 1

                    # Reset section groups
                    section_groups = []

                    # For free panel type, we can change control method in the new section
                    if panel_type == 'free' and section_number <= sections_per_panel:
                        # Try to find unassigned group with different control method
                        alternative_methods = set(group_info[g]['control_method'] for g in unassigned_groups
                                                  if not group_info[g]['assigned'])

                        # Exclude current section control method if possible
                        if current_section_control_method in alternative_methods and len(alternative_methods) > 1:
                            alternative_methods.remove(current_section_control_method)

                        if alternative_methods:
                            current_section_control_method = list(alternative_methods)[0]
                    continue

            # Sort compatible groups by distance
            compatible_groups.sort(key=lambda x: x[1])

            # Select closest group
            next_group, _ = compatible_groups[0]

            # Assign this group to the panel
            group_info[next_group]['assigned'] = True
            outgoing_number += 1
            panel_groups.append((next_group, section_number, outgoing_number))
            section_groups.append(next_group)

            # Increment outgoing number
            # outgoing_number += 1

            # If reached max outgoings per section, move to next section
            if outgoing_number >= max_outgoings_per_section:
                section_number += 1
                outgoing_number = 0

                # Reset section groups
                section_groups = []

                # For free panel type, we can change control method in the new section
                if panel_type == 'free' and section_number <= sections_per_panel:
                    # Try to find unassigned group with different control method
                    alternative_methods = set(group_info[g]['control_method'] for g in unassigned_groups
                                              if not group_info[g]['assigned'])

                    # Exclude current section control method if possible
                    if current_section_control_method in alternative_methods and len(alternative_methods) > 1:
                        alternative_methods.remove(current_section_control_method)

                    if alternative_methods:
                        current_section_control_method = list(alternative_methods)[0]

            # Recalculate panel equidistant point based on all assigned groups
            assigned_eq_points = [group_info[g]['eq_point'] for g, _, _ in panel_groups]
            panel_point = find_equidistant_between_points(assigned_eq_points)

            # Check if all groups are still within distance limits after relocating panel
            all_within_limits = True
            for group, _, _ in panel_groups:
                is_within_limit, _ = check_group_distance_to_panel(
                    group_info[group]['eq_point'], panel_point, group_info[group]['max_distance'], max_distance_limit
                )
                if not is_within_limit:
                    all_within_limits = False
                    break

            if not all_within_limits:
                # If relocating panel puts some groups out of range, undo the last assignment
                group_info[next_group]['assigned'] = False
                panel_groups.pop()  # Remove the last group

                try:
                    section_groups.pop()  # Remove from section groups
                except:
                    print("Empty section")
                # Recalculate panel point without the last group
                assigned_eq_points = [group_info[g]['eq_point'] for g, _, _ in panel_groups]
                panel_point = find_equidistant_between_points(assigned_eq_points)

                # Move to next section
                if outgoing_number > 1:  # Only if we had assigned some groups to this section
                    section_number += 1
                    outgoing_number = 1

                    # Reset section groups
                    section_groups = []

                    # For free panel type, we can change control method in the new section
                    if panel_type == 'free' and section_number <= sections_per_panel:
                        # Try to find unassigned group with different control method
                        alternative_methods = set(group_info[g]['control_method'] for g in unassigned_groups
                                                  if not group_info[g]['assigned'])

                        # Exclude current section control method if possible
                        if current_section_control_method in alternative_methods and len(alternative_methods) > 1:
                            alternative_methods.remove(current_section_control_method)

                        if alternative_methods:
                            current_section_control_method = list(alternative_methods)[0]

        # Update dataframe with panel assignments
        for group, section, outgoing in panel_groups:
            mask = df['Group'] == group
            df.loc[mask, 'Panel'] = panel_number
            df.loc[mask, 'Section'] = section
            df.loc[mask, 'Outgoing'] = outgoing
            df.loc[mask, 'Panel_X'] = panel_point[0]
            df.loc[mask, 'Panel_Y'] = panel_point[1]
            df.loc[mask, 'Panel_Z'] = panel_point[2]

            # Calculate distance from each circuit to the panel
            df.loc[mask, 'Distance_To_Panel'] = df[mask].apply(
                lambda row: manhattan_distance(
                    (row['North_X'], row['East_Y'], row['Elevation_Z']),
                    panel_point
                ),
                axis=1
            )

        # Move to next panel
        panel_number += 1

    # Save final result to output file
    df.to_excel(output_file, index=False)

    return df


def generate_dxf_with_blocks(result_df, output_file, panel_side, square_size, circuit_radius, text_height, grouping):
    """Generate DXF file with panels and circuits using BLOCKS for cleaner structure.

    Args:
        result_df: DataFrame containing panel and circuit data
        output_file: Path to save the DXF file
        panel_side: Size of panel triangle side length
        square_size: Size of the group square
        circuit_radius: Radius of circuit circles
        text_height: Height of text labels

    Returns:
        bool: True if successful
    """
    # Import ezdxf here to keep it as a dependency only for this function

    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    block_defs = doc.blocks

    # Create panel block (equilateral triangle centered at origin)
    if 'PANEL_BLOCK' not in block_defs:
        height = panel_side * math.sqrt(3) / 2
        panel_points = [
            (0, -height / 3, 0),
            (-panel_side / 2, 2 * height / 3, 0),
            (panel_side / 2, 2 * height / 3, 0),
            (0, -height / 3, 0)
        ]
        panel_block = block_defs.new(name='PANEL_BLOCK')
        panel_block.add_polyline3d(panel_points)
        # Add PANEL_NAME attribute
        panel_block.add_attdef(
            tag='PANEL_NAME',
            insert=(0, 0, 0),
            height=text_height,
            dxfattribs={'layer': '0'}
        )

    # Create group block (square with attributes)
    if 'GROUP_BLOCK' not in block_defs:
        group_block = block_defs.new(name='GROUP_BLOCK')

        half_size = square_size / 2.0
        # Define 4 corners of the square
        square_points = [
            (-half_size, -half_size),
            (half_size, -half_size),
            (half_size, half_size),
            (-half_size, half_size),
            (-half_size, -half_size)  # Close the square
        ]

        # Add polyline (LWPOLYLINE) to represent the square
        group_block.add_lwpolyline(points=square_points, close=True)

        # Add attributes to the group block
        attributes = [
            ('Group', text_height),
            ('Panel', text_height * 0.7),
            ('Section', text_height * 0.7),
            ('Outgoing', text_height * 0.7),
            ('Equidistant_Z', text_height * 0.0007),
            ('Control method', text_height * 0.0007)
        ]

        for i, (tag, height) in enumerate(attributes):
            insert_y = -half_size * 0.7 * i  # Use negative if you want them stacking downward
            group_block.add_attdef(
                tag=tag,
                insert=(0, insert_y, 0),
                height=height,
                dxfattribs={
                    'layer': '0'
                }
            )

    # Create circuit block (circle and text with attribute)
    if 'CIRCUIT_BLOCK' not in block_defs:
        circuit_block = block_defs.new(name='CIRCUIT_BLOCK')
        circuit_block.add_circle(center=(0, 0, 0), radius=circuit_radius)

        # Common DXF attributes
        dxf_attrs = {'layer': '0'}

        attributes = [
            ('Circuit', text_height),
            ('Panel', text_height * 0.7),
            ('Section', text_height * 0.7),
            ('Outgoing', text_height * 0.7),
            ('Group', text_height * 0.0007),
            ('Elevation_Z', text_height * 0.0007),
            ('Control method', text_height * 0.0007)
        ]

        for i, (tag, height) in enumerate(attributes):
            insert_y = -circuit_radius * 0.7 * i  # Use negative if you want them stacking downward
            circuit_block.add_attdef(
                tag=tag,
                insert=(0, insert_y, 0),
                height=height,
                dxfattribs={
                    'layer': '0',
                }
            )

    # Assign unique color layers for panels
    panels = result_df['Panel'].unique()
    layer_colour = 1
    for panel in panels:
        if "Panel_" + str(panel) not in doc.layers:
            doc.layers.new(name="Panel_" + str(panel), dxfattribs={'color': layer_colour})
            layer_colour += 1
            if layer_colour > 255:  # DXF color limit
                layer_colour = 1

    # Process each panel
    for panel in panels:
        panel_data = result_df[result_df['Panel'] == panel]
        if len(panel_data) == 0:
            continue

        # Get panel coordinates from first circuit in panel
        panel_row = panel_data.iloc[0]
        panel_insert_point = (
            panel_row['Panel_X'],
            panel_row['Panel_Y'],
            panel_row['Panel_Z']
        )

        # Insert panel block
        triangle_ref = msp.add_blockref('PANEL_BLOCK', insert=panel_insert_point,
                                        dxfattribs={'layer': "Panel_" + str(panel)})
        triangle_ref.add_auto_attribs({
            'PANEL_NAME': f"Panel {panel}"
        })

        # Create group and entity list
        panel_group_name = f"Panel_{panel}"
        panel_group = doc.groups.new(panel_group_name)
        panel_entities = [triangle_ref]

        # Create connection lines from panel to each group's equidistant point
        # Group the circuits by group number
        for group_num, group_circuits in panel_data.groupby('Group'):
            first_circuit = group_circuits.iloc[0]

            # Get equidistant point for this group
            group_eq_point = (
                first_circuit['Equidistant_X'],
                first_circuit['Equidistant_Y'],
                first_circuit['Equidistant_Z']
            )

            # Draw line from panel to group equidistant point
            line = msp.add_line(
                start=panel_insert_point,
                end=group_eq_point,
                dxfattribs={'layer': "Panel_" + str(panel)}
            )
            panel_entities.append(line)

            # Insert GROUP_BLOCK at the equidistant point
            group_blockref = msp.add_blockref('GROUP_BLOCK', insert=group_eq_point,
                                              dxfattribs={'layer': "Panel_" + str(panel)})

            # Add attributes to the group block reference

            group_blockref.add_auto_attribs({
                key: str(first_circuit[key])
                for key in ['Group', 'Panel', 'Section', 'Outgoing', 'Equidistant_Z','Control method']
            })

            panel_entities.append(group_blockref)

            # Process each circuit in the group
            for _, circuit in group_circuits.iterrows():
                circuit_insert = (
                    circuit['North_X'],
                    circuit['East_Y'],
                    circuit['Elevation_Z']
                )

                # Circuit layer is panel number
                layer_name = "Panel_" + str(panel)

                # Insert circuit block reference and set attributes
                blockref = msp.add_blockref('CIRCUIT_BLOCK', insert=circuit_insert, dxfattribs={'layer': layer_name})
                blockref.add_auto_attribs({
                    key: str(circuit[key])
                    for key in ['Circuit', 'Panel', 'Section', 'Outgoing', 'Group', 'Elevation_Z','Control method']
                })

                panel_entities.append(blockref)

                # Draw line from circuit to its group's equidistant point
                circuit_line = msp.add_line(
                    start=circuit_insert,
                    end=group_eq_point,
                    dxfattribs={'layer': layer_name}
                )
                panel_entities.append(circuit_line)

        # Add all entities to the panel group
        if grouping:
            panel_group.extend(panel_entities)

    doc.saveas(output_file)
    return True
# Combined execution function
def execute_full_process(input_file, output_file,
                         chain_type='daisy', init_radius=5000, radius_step=5000,
                       max_iterations=5, max_current_per_group=20, max_circuit_per_group=5, panel_type='free', sections_per_panel=3,
                         max_outgoings_per_section=10, max_distance_limit=150000,dxf_outputfile='layout.dxf', panel_side=1000, square_size=300, circuit_radius=300, text_height=250,
                         grouping=True):
    """Execute the full process: grouping, MST, remote index, and panel assignment"""
    # Step 1: Group circuits
    result_df = group_circuits(
        input_file,
        chain_type,
        init_radius,
        radius_step,
        max_iterations,
        max_current_per_group,
        max_circuit_per_group
    )


    # Step 2: Process groups with MST and assign remote indices
    mst_result_df, remote_index_mapping, group_mapping, equidistant_points = process_groups_with_distances(
        result_df
    )

    # Step 3: Assign groups to panels
    final_df = assign_groups_to_panels(
        mst_result_df,
        output_file,
        sections_per_panel,
        max_outgoings_per_section,
        max_distance_limit,
        panel_type
    )


    dxf_result = generate_dxf_with_blocks(
        final_df,
        dxf_outputfile,
        panel_side,
        square_size,
        circuit_radius,
        text_height,
        grouping
    )



    num_groups = final_df['Group'].nunique()
    num_panels = final_df['Panel'].nunique()

    print(f"Process complete.")
    print(f"Created {num_groups} groups.")
    print(f"Created {num_panels} panels.")
    print(f"Results saved to {output_file}")

    return final_df


# Example usage
# if __name__ == "__main__":
#     # Change these parameters as needed

#     input_file = "input1.xlsx"
#     output_file = "output.xlsx"
#     dxf_outputfile = 'layout.dxf'

#     # Algorithm parameters
#     chain_type = "daisy"  # or "MB"
#     panel_type = "locked"  # or "locked"
#     sections_per_panel = 1
#     max_outgoings_per_section = 20
#     max_distance_limit = 100000
#     init_radius = 5000
#     radius_step = 5000
#     max_iterations = 3
#     max_current_per_group = 20
#     max_circuit_per_group = 4
#     panel_side = 1000
#     square_size = 300
#     circuit_radius = 300
#     text_height = 250




    # final_result = execute_full_process(
    #     input_file,
    #     output_file,
    #     chain_type=chain_type, init_radius=init_radius, radius_step=radius_step,
    #     max_iterations=max_iterations, max_current_per_group=max_current_per_group,
    #     max_circuit_per_group=max_circuit_per_group, panel_type=panel_type, sections_per_panel=sections_per_panel,
    #     max_outgoings_per_section=max_outgoings_per_section, max_distance_limit=max_distance_limit,
    #     dxf_outputfile=dxf_outputfile, panel_side=panel_side,square_size=square_size, circuit_radius=circuit_radius,
    #     text_height=text_height,grouping=True
    # )
