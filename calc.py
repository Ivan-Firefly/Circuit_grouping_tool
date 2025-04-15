import pandas as pd
import math
import ezdxf
from itertools import combinations
from settings import *

# converting to meters
MAX_DISTANCE_TO_PANEL = MAX_DISTANCE_TO_PANEL *1000
MAX_DISTANCE_BETWEEN_CIRCUITS = MAX_DISTANCE_BETWEEN_CIRCUITS *1000


def manhattan_distance(p1, p2):
    """Calculate Manhattan distance between two points"""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])


def calculate_centroid(points):
    """Calculate the centroid of multiple points"""
    if len(points) == 0:
        return None
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    z = sum(p[2] for p in points) / len(points)
    return (x, y, z)


def is_valid_reassignment(circuit, target_panel, panels_data, max_distance=MAX_DISTANCE_TO_PANEL):
    """Check if reassigning a circuit to another panel is valid"""
    panel_coords = (target_panel['North_X'], target_panel['East_Y'], target_panel['Elevation_Z'])
    circuit_coords = (circuit['North_X'], circuit['East_Y'], circuit['Elevation_Z'])
    dist = manhattan_distance(panel_coords, circuit_coords)

    if dist > max_distance:
        return False

    # Check if adding to the section with same control method would exceed outgoing limits
    target_section = None
    for section_num in range(1, 4):
        section_circuits = panels_data.loc[
            (panels_data['Panel'] == target_panel['Panel']) &
            (panels_data['Section'] == section_num)
            ]
        if len(section_circuits) == 0:
            target_section = section_num
            break

        first_circuit = section_circuits.iloc[0]
        if first_circuit['Control method'] == circuit['Control method']:
            if len(section_circuits) < 10:  # Max outgoing per section
                target_section = section_num
                break

    return target_section is not None


def group_circuits(df, max_distance=MAX_DISTANCE_TO_PANEL, max_outgoing_cb_current=MAX_CIRCUIT_CB_CURRENT, max_distance_between_circuits=MAX_DISTANCE_BETWEEN_CIRCUITS,
                   sections_per_panel=3, max_outgoing_per_section=OUTGOINGS_PER_SECTION-SPARE_OUTGOINGS_PER_SECTION):
    """Group circuits into panels based on proximity and constraints"""

    # Create a copy of the input dataframe
    circuits = df.copy()

    # Add columns for panel, section, and outgoing assignments
    circuits['Panel'] = None
    circuits['Panel_North_X'] = None
    circuits['Panel_East_Y'] = None
    circuits['Panel_Elevation_Z'] = None
    circuits['Section'] = None
    circuits['Outgoing'] = None
    circuits['Distance_to_Panel'] = None

    # Sort circuits by CB Current (descending) to start with highest current
    circuits = circuits.sort_values('Circuit CB Current', ascending=False).reset_index(drop=True)

    panel_count = 0

    while circuits[circuits['Panel'].isna()].shape[0] > 0:
        panel_count += 1
        panel_name = f"Panel_{panel_count}"

        # Get the first unassigned circuit with highest CB Current
        first_circuit_idx = circuits[circuits['Panel'].isna()].index[0]
        first_circuit = circuits.loc[first_circuit_idx]

        # Initialize panel location at the first circuit
        panel_location = (first_circuit['North_X'], first_circuit['East_Y'], first_circuit['Elevation_Z'])

        # Assign first circuit to the panel
        circuits.loc[first_circuit_idx, 'Panel'] = panel_name
        circuits.loc[first_circuit_idx, 'Panel_North_X'] = panel_location[0]
        circuits.loc[first_circuit_idx, 'Panel_East_Y'] = panel_location[1]
        circuits.loc[first_circuit_idx, 'Panel_Elevation_Z'] = panel_location[2]
        circuits.loc[first_circuit_idx, 'Section'] = 1
        circuits.loc[first_circuit_idx, 'Outgoing'] = 1
        circuits.loc[first_circuit_idx, 'Distance_to_Panel'] = 0

        # Variables to track sections and outgoings
        current_section = 1
        current_outgoing = 1
        outgoing_current = first_circuit['Circuit CB Current']
        section_control_method = first_circuit['Control method']

        # Assigned circuits coordinates for this panel
        assigned_coords = [(first_circuit['North_X'], first_circuit['East_Y'], first_circuit['Elevation_Z'])]

        # Continue assigning circuits until we reach section and panel limits
        while current_section <= sections_per_panel:
            # Find unassigned circuits within max_distance of the panel
            candidate_circuits = []

            for idx, circuit in circuits[circuits['Panel'].isna()].iterrows():
                circuit_coords = (circuit['North_X'], circuit['East_Y'], circuit['Elevation_Z'])
                dist_to_panel = manhattan_distance(panel_location, circuit_coords)

                # Check if the circuit is within range of the panel
                if dist_to_panel <= max_distance:
                    # Check if control method matches current section's control method
                    if circuit['Control method'] == section_control_method:
                        # Check if the circuit is within max_distance_between_circuits from any assigned circuit
                        is_close_to_assigned = False
                        for assigned_coord in assigned_coords:
                            if manhattan_distance(assigned_coord, circuit_coords) <= max_distance_between_circuits:
                                is_close_to_assigned = True
                                break

                        # Check if adding this circuit would exceed the outgoing current limit
                        if (outgoing_current + circuit[
                            'Circuit CB Current'] <= max_outgoing_cb_current) and is_close_to_assigned:
                            candidate_circuits.append({
                                'idx': idx,
                                'cb_current': circuit['Circuit CB Current'],
                                'distance': dist_to_panel
                            })

            # If no more candidates found for current section and control method
            if not candidate_circuits:
                # Try to start a new outgoing in the same section if we haven't reached the limit
                if current_outgoing < max_outgoing_per_section:
                    current_outgoing += 1
                    outgoing_current = 0

                    # Try again with the reset outgoing current but keep same control method
                    continue
                else:
                    # Try to start a new section if we haven't reached the limit
                    if current_section < sections_per_panel:
                        current_section += 1
                        current_outgoing = 1
                        outgoing_current = 0
                        section_control_method = None

                        # Try to find the first unassigned circuit for the new section
                        for idx, circuit in circuits[circuits['Panel'].isna()].iterrows():
                            circuit_coords = (circuit['North_X'], circuit['East_Y'], circuit['Elevation_Z'])
                            dist_to_panel = manhattan_distance(panel_location, circuit_coords)

                            if dist_to_panel <= max_distance:
                                if section_control_method is None or circuit[
                                    'Control method'] == section_control_method:
                                    section_control_method = circuit['Control method']

                                    # Assign circuit to the new section
                                    circuits.loc[idx, 'Panel'] = panel_name
                                    circuits.loc[idx, 'Panel_North_X'] = panel_location[0]
                                    circuits.loc[idx, 'Panel_East_Y'] = panel_location[1]
                                    circuits.loc[idx, 'Panel_Elevation_Z'] = panel_location[2]
                                    circuits.loc[idx, 'Section'] = current_section
                                    circuits.loc[idx, 'Outgoing'] = current_outgoing
                                    circuits.loc[idx, 'Distance_to_Panel'] = dist_to_panel

                                    outgoing_current = circuit['Circuit CB Current']
                                    assigned_coords.append(circuit_coords)

                                    # Recalculate panel location
                                    panel_location = calculate_centroid(assigned_coords)

                                    # Update panel coordinates for all assigned circuits
                                    for assigned_idx in circuits[circuits['Panel'] == panel_name].index:
                                        old_dist = circuits.loc[assigned_idx, 'Distance_to_Panel']
                                        circuit_coords = (circuits.loc[assigned_idx, 'North_X'],
                                                          circuits.loc[assigned_idx, 'East_Y'],
                                                          circuits.loc[assigned_idx, 'Elevation_Z'])
                                        new_dist = manhattan_distance(panel_location, circuit_coords)

                                        # Check if any circuit now exceeds max distance
                                        if new_dist > max_distance:
                                            # Remove the last assigned circuit
                                            circuits.loc[idx, 'Panel'] = None
                                            circuits.loc[idx, 'Panel_North_X'] = None
                                            circuits.loc[idx, 'Panel_East_Y'] = None
                                            circuits.loc[idx, 'Panel_Elevation_Z'] = None
                                            circuits.loc[idx, 'Section'] = None
                                            circuits.loc[idx, 'Outgoing'] = None
                                            circuits.loc[idx, 'Distance_to_Panel'] = None
                                            assigned_coords.pop()

                                            # Revert panel location
                                            panel_location = calculate_centroid(assigned_coords)
                                            section_control_method = None
                                            outgoing_current = 0
                                            break
                                        else:
                                            # Update panel coordinates for this circuit
                                            circuits.loc[assigned_idx, 'Panel_North_X'] = panel_location[0]
                                            circuits.loc[assigned_idx, 'Panel_East_Y'] = panel_location[1]
                                            circuits.loc[assigned_idx, 'Panel_Elevation_Z'] = panel_location[2]
                                            circuits.loc[assigned_idx, 'Distance_to_Panel'] = new_dist

                                    break  # Found a circuit for the new section

                        # If we found a circuit for the new section, continue assigning more
                        if section_control_method is not None:
                            continue

                    # If we've reached the max sections or couldn't find a circuit for a new section,
                    # move on to create a new panel
                    break

            # Select the closest circuit from candidates with highest CB Current
            if candidate_circuits:
                candidates_df = pd.DataFrame(candidate_circuits)
                # Sort by CB current (descending) then by distance (ascending)
                candidates_df = candidates_df.sort_values(['cb_current', 'distance'], ascending=[False, True])
                best_candidate_idx = candidates_df.iloc[0]['idx']
                best_candidate = circuits.loc[best_candidate_idx]

                # Assign the circuit to the panel
                circuits.loc[best_candidate_idx, 'Panel'] = panel_name
                circuits.loc[best_candidate_idx, 'Panel_North_X'] = panel_location[0]
                circuits.loc[best_candidate_idx, 'Panel_East_Y'] = panel_location[1]
                circuits.loc[best_candidate_idx, 'Panel_Elevation_Z'] = panel_location[2]
                circuits.loc[best_candidate_idx, 'Section'] = current_section
                circuits.loc[best_candidate_idx, 'Outgoing'] = current_outgoing

                circuit_coords = (best_candidate['North_X'], best_candidate['East_Y'], best_candidate['Elevation_Z'])
                dist_to_panel = manhattan_distance(panel_location, circuit_coords)
                circuits.loc[best_candidate_idx, 'Distance_to_Panel'] = dist_to_panel

                # Update outgoing current
                outgoing_current += best_candidate['Circuit CB Current']

                # Add coordinates to assigned list
                assigned_coords.append(circuit_coords)

                # Recalculate panel location as centroid of all assigned circuits
                panel_location = calculate_centroid(assigned_coords)

                # Update panel coordinates for all assigned circuits
                valid_relocation = True
                for assigned_idx in circuits[circuits['Panel'] == panel_name].index:
                    circuit_coords = (circuits.loc[assigned_idx, 'North_X'],
                                      circuits.loc[assigned_idx, 'East_Y'],
                                      circuits.loc[assigned_idx, 'Elevation_Z'])
                    new_dist = manhattan_distance(panel_location, circuit_coords)

                    # Check if any circuit now exceeds max distance
                    if new_dist > max_distance:
                        valid_relocation = False
                        break

                if valid_relocation:
                    # Update all circuits' distances with the new panel location
                    for assigned_idx in circuits[circuits['Panel'] == panel_name].index:
                        circuit_coords = (circuits.loc[assigned_idx, 'North_X'],
                                          circuits.loc[assigned_idx, 'East_Y'],
                                          circuits.loc[assigned_idx, 'Elevation_Z'])
                        new_dist = manhattan_distance(panel_location, circuit_coords)
                        circuits.loc[assigned_idx, 'Panel_North_X'] = panel_location[0]
                        circuits.loc[assigned_idx, 'Panel_East_Y'] = panel_location[1]
                        circuits.loc[assigned_idx, 'Panel_Elevation_Z'] = panel_location[2]
                        circuits.loc[assigned_idx, 'Distance_to_Panel'] = new_dist
                else:
                    # Revert the assignment of the last circuit
                    circuits.loc[best_candidate_idx, 'Panel'] = None
                    circuits.loc[best_candidate_idx, 'Panel_North_X'] = None
                    circuits.loc[best_candidate_idx, 'Panel_East_Y'] = None
                    circuits.loc[best_candidate_idx, 'Panel_Elevation_Z'] = None
                    circuits.loc[best_candidate_idx, 'Section'] = None
                    circuits.loc[best_candidate_idx, 'Outgoing'] = None
                    circuits.loc[best_candidate_idx, 'Distance_to_Panel'] = None

                    # Remove from assigned coordinates
                    assigned_coords.pop()

                    # Recalculate panel location
                    panel_location = calculate_centroid(assigned_coords)

                    # Check if we need to move to a new outgoing or section
                    if outgoing_current >= max_outgoing_cb_current:
                        current_outgoing += 1
                        outgoing_current = 0

                        if current_outgoing > max_outgoing_per_section:
                            current_section += 1
                            current_outgoing = 1
                            section_control_method = None

                            if current_section > sections_per_panel:
                                break
            else:
                # No more valid circuits for this section, move to next section
                current_section += 1
                current_outgoing = 1
                outgoing_current = 0
                section_control_method = None

                if current_section > sections_per_panel:
                    break

    return circuits


def optimize_panels(result_df, max_distance=MAX_DISTANCE_TO_PANEL, sections_per_panel=SECTIONS_PER_PANEL, max_outgoing_per_section=OUTGOINGS_PER_SECTION-SPARE_OUTGOINGS_PER_SECTION):
    """Optimize panel utilization based on the specified constraints"""
    optimized = result_df.copy()

    # Get all panels
    panels = optimized['Panel'].unique()

    # First phase: Check panels with insufficient sections
    panels_to_optimize = []
    for panel in panels:
        panel_circuits = optimized[optimized['Panel'] == panel]
        used_sections = panel_circuits['Section'].nunique()

        # If panel uses fewer than required sections, add to optimization list
        if used_sections < sections_per_panel:
            panels_to_optimize.append(panel)

    # Second phase: Check outgoing utilization per section
    for panel in panels:
        panel_circuits = optimized[optimized['Panel'] == panel]
        for section in range(1, sections_per_panel + 1):
            section_circuits = panel_circuits[panel_circuits['Section'] == section]
            if 0 < len(section_circuits) < max_outgoing_per_section / 2:  # Less than 50% utilization
                if panel not in panels_to_optimize:
                    panels_to_optimize.append(panel)

    # Try to merge panels or redistribute circuits
    if len(panels_to_optimize) > 1:
        for panel_combo in combinations(panels_to_optimize, 2):
            panel1, panel2 = panel_combo
            panel1_circuits = optimized[optimized['Panel'] == panel1]
            panel2_circuits = optimized[optimized['Panel'] == panel2]

            if len(panel1_circuits) == 0 or len(panel2_circuits) == 0:
                continue

            # Check if we can merge these panels
            if (panel1_circuits['Section'].nunique() + panel2_circuits['Section'].nunique() <= sections_per_panel):
                can_merge = True

                # Check control method compatibility
                for section in panel2_circuits['Section'].unique():
                    control_methods_panel2 = panel2_circuits[panel2_circuits['Section'] == section][
                        'Control method'].unique()
                    for section1 in panel1_circuits['Section'].unique():
                        control_methods_panel1 = panel1_circuits[panel1_circuits['Section'] == section1][
                            'Control method'].unique()
                        if any(method in control_methods_panel1 for method in control_methods_panel2):
                            can_merge = False
                            break
                    if not can_merge:
                        break

                if can_merge:
                    # Merge panel2 into panel1
                    panel1_location = (panel1_circuits['Panel_North_X'].iloc[0],
                                       panel1_circuits['Panel_East_Y'].iloc[0],
                                       panel1_circuits['Panel_Elevation_Z'].iloc[0])

                    # Check if all panel2 circuits would be within range of panel1
                    for _, circuit in panel2_circuits.iterrows():
                        circuit_coords = (circuit['North_X'], circuit['East_Y'], circuit['Elevation_Z'])
                        if manhattan_distance(panel1_location, circuit_coords) > max_distance:
                            can_merge = False
                            break

                    if can_merge:
                        # Calculate optimal location for merged panel
                        all_coords = [(c['North_X'], c['East_Y'], c['Elevation_Z'])
                                      for _, c in pd.concat([panel1_circuits, panel2_circuits]).iterrows()]
                        new_panel_location = calculate_centroid(all_coords)

                        # Update panel assignment for panel2 circuits
                        for idx in panel2_circuits.index:
                            optimized.loc[idx, 'Panel'] = panel1
                            optimized.loc[idx, 'Panel_North_X'] = new_panel_location[0]
                            optimized.loc[idx, 'Panel_East_Y'] = new_panel_location[1]
                            optimized.loc[idx, 'Panel_Elevation_Z'] = new_panel_location[2]

                            # Update distance to panel
                            circuit_coords = (optimized.loc[idx, 'North_X'],
                                              optimized.loc[idx, 'East_Y'],
                                              optimized.loc[idx, 'Elevation_Z'])
                            optimized.loc[idx, 'Distance_to_Panel'] = manhattan_distance(new_panel_location,
                                                                                         circuit_coords)

                        # Update panel location for panel1 circuits
                        for idx in panel1_circuits.index:
                            optimized.loc[idx, 'Panel_North_X'] = new_panel_location[0]
                            optimized.loc[idx, 'Panel_East_Y'] = new_panel_location[1]
                            optimized.loc[idx, 'Panel_Elevation_Z'] = new_panel_location[2]

                            # Update distance to panel
                            circuit_coords = (optimized.loc[idx, 'North_X'],
                                              optimized.loc[idx, 'East_Y'],
                                              optimized.loc[idx, 'Elevation_Z'])
                            optimized.loc[idx, 'Distance_to_Panel'] = manhattan_distance(new_panel_location,
                                                                                         circuit_coords)

                        # Renumber sections if needed
                        section_mapping = {}
                        current_section = 1

                        # First map existing sections from panel1
                        for section in sorted(panel1_circuits['Section'].unique()):
                            section_mapping[f"panel1_{section}"] = current_section
                            current_section += 1

                        # Then map sections from panel2
                        for section in sorted(panel2_circuits['Section'].unique()):
                            section_mapping[f"panel2_{section}"] = current_section
                            current_section += 1

                        # Apply section mapping
                        for idx in panel1_circuits.index:
                            old_section = optimized.loc[idx, 'Section']
                            optimized.loc[idx, 'Section'] = section_mapping[f"panel1_{old_section}"]

                        for idx in panel2_circuits.index:
                            old_section = optimized.loc[idx, 'Section']
                            optimized.loc[idx, 'Section'] = section_mapping[f"panel2_{old_section}"]

    # Final phase: Handle circuits far from their panels (>1500000)
    circuits_to_reassign = optimized[optimized['Distance_to_Panel'] > 1500000]

    for idx, far_circuit in circuits_to_reassign.iterrows():
        current_panel = far_circuit['Panel']
        best_distance = far_circuit['Distance_to_Panel']
        best_panel = None

        # Find a better panel for this circuit
        for panel in optimized['Panel'].unique():
            if panel == current_panel:
                continue

            panel_data = optimized[optimized['Panel'] == panel]
            if len(panel_data) == 0:
                continue

            panel_info = {
                'Panel': panel,
                'North_X': panel_data['Panel_North_X'].iloc[0],
                'East_Y': panel_data['Panel_East_Y'].iloc[0],
                'Elevation_Z': panel_data['Panel_Elevation_Z'].iloc[0]
            }

            panel_coords = (panel_info['North_X'], panel_info['East_Y'], panel_info['Elevation_Z'])
            circuit_coords = (far_circuit['North_X'], far_circuit['East_Y'], far_circuit['Elevation_Z'])
            distance = manhattan_distance(panel_coords, circuit_coords)

            if distance < best_distance and is_valid_reassignment(far_circuit, panel_info, optimized):
                best_distance = distance
                best_panel = panel_info

        # If found a better panel, reassign the circuit
        if best_panel:
            # Find appropriate section with matching control method
            target_section = None
            target_outgoing = None

            for section_num in range(1, 4):
                section_circuits = optimized.loc[
                    (optimized['Panel'] == best_panel['Panel']) &
                    (optimized['Section'] == section_num)
                    ]

                if len(section_circuits) == 0:
                    target_section = section_num
                    target_outgoing = 1
                    break

                control_methods = section_circuits['Control method'].unique()
                if far_circuit['Control method'] in control_methods:
                    # Find an outgoing with capacity
                    for outgoing_num in range(1, max_outgoing_per_section + 1):
                        outgoing_circuits = section_circuits[section_circuits['Outgoing'] == outgoing_num]

                        if len(outgoing_circuits) == 0:
                            target_section = section_num
                            target_outgoing = outgoing_num
                            break

                        # Check if adding this circuit would exceed outgoing current limit
                        outgoing_current_sum = outgoing_circuits['Circuit CB Current'].sum()
                        if outgoing_current_sum + far_circuit['Circuit CB Current'] <= 20:  # Max outgoing current
                            target_section = section_num
                            target_outgoing = outgoing_num
                            break

                    if target_section is not None:
                        break

            # If found a suitable section/outgoing, reassign
            if target_section is not None:
                optimized.loc[idx, 'Panel'] = best_panel['Panel']
                optimized.loc[idx, 'Panel_North_X'] = best_panel['North_X']
                optimized.loc[idx, 'Panel_East_Y'] = best_panel['East_Y']
                optimized.loc[idx, 'Panel_Elevation_Z'] = best_panel['Elevation_Z']
                optimized.loc[idx, 'Section'] = target_section
                optimized.loc[idx, 'Outgoing'] = target_outgoing
                optimized.loc[idx, 'Distance_to_Panel'] = best_distance

    return optimized

def generate_dxf_with_blocks(result_df, output_file):
    """Generate DXF file with panels and circuits using BLOCKS for cleaner structure."""
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    block_defs = doc.blocks

    # Constants
    panel_side = PANEL_BLOCK_SIZE
    circuit_radius = CIRCUIT_BLOCK_SIZE
    text_height = BLOCK_TEXT_SIZE


    # Create panel block (equilateral triangle centered at origin)
    # Create panel block (triangle centered at origin with attribute)
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

    # Create circuit block (circle and text with attribute)
    if 'CIRCUIT_BLOCK' not in block_defs:
        circuit_block = block_defs.new(name='CIRCUIT_BLOCK')
        circuit_block.add_circle(center=(0, 0, 0), radius=circuit_radius)
        # Add CIRCUIT_NAME attribute
        circuit_block.add_attdef(
            tag='CIRCUIT_NAME',
            insert=(0, 0, 0),
            height=text_height,
            dxfattribs={'layer': '0'}
        )

    # Assign unique color layers for panels
    panels = result_df['Panel'].unique()
    layer_colour = 1
    for panel in panels:
        if panel not in doc.layers:
            doc.layers.new(name=panel, dxfattribs={'color': layer_colour})
            layer_colour += 1

    for panel in panels:
        panel_data = result_df[result_df['Panel'] == panel]
        if len(panel_data) == 0:
            continue

        panel_row = panel_data.iloc[0]
        panel_insert_point = (
            panel_row['Panel_North_X'],
            panel_row['Panel_East_Y'],
            panel_row['Panel_Elevation_Z']
        )

        # Insert panel block
        triangle_ref = msp.add_blockref('PANEL_BLOCK', insert=panel_insert_point, dxfattribs={'layer': panel})
        triangle_ref.add_auto_attribs({'PANEL_NAME': str(panel)})

        # Create group and entity list
        panel_group_name = f"Group_{panel}"
        panel_group = doc.groups.new(panel_group_name)
        panel_entities = [triangle_ref]

        # Insert circuits using block with attributes
        for _, circuit in panel_data.iterrows():
            circuit_insert = (
                circuit['North_X'],
                circuit['East_Y'],
                circuit['Elevation_Z']
            )
            # Insert circuit block reference and set CIRCUIT_NAME attribute
            blockref = msp.add_blockref('CIRCUIT_BLOCK', insert=circuit_insert, dxfattribs={'layer': panel})
            blockref.add_auto_attribs({'CIRCUIT_NAME': str(circuit['Circuit'])})

            panel_entities.append(blockref)

        panel_group.extend(panel_entities)

    doc.saveas(output_file)
    return True


def main(input_file, output_excel, output_dxf):
    """Main function to execute the circuit grouping workflow"""
    # Load data
    df = pd.read_excel(input_file)

    # Group circuits into panels
    result = group_circuits(df)

    # Optimize panels
    optimized_result = optimize_panels(result)

    # Generate DXF file
    generate_dxf_with_blocks(optimized_result, output_dxf)

    # Save results to Excel
    optimized_result.to_excel(output_excel, index=False)

    # Print summary
    panel_count = optimized_result['Panel'].nunique()
    print(f"Grouping completed successfully. Created {panel_count} panels.")

    return optimized_result



main(INPUT_FILE, OUTPUT_FILE_XLSX, OUTPUT_FILE_DXF)
