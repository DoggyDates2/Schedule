#!/usr/bin/env python3
"""
Dog Walking Route Optimization System - Version 2
With Force Assignment and Dynamic Cascading
"""

import os
import sys
import json
import time
import math
import logging
import statistics
from datetime import datetime
from collections import defaultdict, Counter

# Google Sheets imports
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except ImportError:
    print("Please install required packages: pip install gspread oauth2client")
    sys.exit(1)

# For Slack notifications (optional)
try:
    import requests
except ImportError:
    requests = None
    print("Warning: requests not installed, Slack notifications disabled")


class DogReassignmentSystem:
    def __init__(self):
        print("=" * 60)
        print("DOG WALKING ROUTE OPTIMIZATION SYSTEM")
        print("Version 2.0 - Force Assignment & Dynamic Cascading")
        print("=" * 60)
        
        # Configuration
        self.CLUSTER_THRESHOLD = 1  # minutes
        self.MIN_GROUP_SIZE = 4
        self.MIN_DOGS_FOR_DRIVER = 7
        self.GROUP_CONSOLIDATION_TIME_LIMIT = 10  # minutes
        self.OUTLIER_MULTIPLIER = 1.5
        self.OUTLIER_ABSOLUTE = 3  # minutes
        self.DENSE_ROUTE_THRESHOLD = 2  # minutes average
        
        # Initialize data structures
        self.dog_assignments = []
        self.distance_matrix = {}
        self.dog_coordinates = {}
        self.active_drivers = set()
        self.driver_assignment_counts = defaultdict(int)
        self.haversine_fallback_count = 0
        
        # Setup Google Sheets
        self.setup_google_sheets()
        
    def setup_google_sheets(self):
        """Initialize Google Sheets connection"""
        try:
            scope = ['https://spreadsheets.google.com/feeds',
                     'https://www.googleapis.com/auth/drive']
            
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                'credentials.json', scope)
            client = gspread.authorize(creds)
            
            # Open spreadsheets - Updated worksheet names
            self.map_sheet = client.open("DoggyDates Schedule").worksheet("Map")
            self.matrix_sheet = client.open("Matrix").worksheet("Matrix")
            
            print("✅ Connected to Google Sheets successfully")
            
        except Exception as e:
            print(f"❌ Error connecting to Google Sheets: {e}")
            sys.exit(1)
    
    def load_distance_matrix(self):
        """Load distance matrix from Google Sheets"""
        print("\n📊 Loading distance matrix...")
        
        try:
            # Get all values from matrix sheet
            matrix_data = self.matrix_sheet.get_all_values()
            
            if not matrix_data:
                print("❌ No data found in matrix sheet")
                return
            
            # First row contains dog IDs
            header_dog_ids = matrix_data[0][1:]  # Skip first cell
            
            # Process each row (skip header)
            for row_idx, row in enumerate(matrix_data[1:], 1):
                if not row:
                    continue
                    
                from_dog_id = row[0]
                if not from_dog_id:
                    continue
                
                if from_dog_id not in self.distance_matrix:
                    self.distance_matrix[from_dog_id] = {}
                
                # Process distances in this row
                for col_idx, distance_str in enumerate(row[1:], 0):
                    if col_idx < len(header_dog_ids):
                        to_dog_id = header_dog_ids[col_idx]
                        
                        if to_dog_id and distance_str:
                            try:
                                # Handle both time formats
                                if ':' in distance_str:
                                    parts = distance_str.split(':')
                                    if len(parts) == 2:
                                        minutes = int(parts[0])
                                        seconds = int(parts[1])
                                        total_minutes = minutes + seconds / 60.0
                                    else:
                                        continue
                                else:
                                    total_minutes = float(distance_str)
                                
                                self.distance_matrix[from_dog_id][to_dog_id] = total_minutes
                                
                            except (ValueError, IndexError):
                                continue
            
            total_entries = sum(len(distances) for distances in self.distance_matrix.values())
            print(f"✅ Loaded {total_entries} distance entries for {len(self.distance_matrix)} dogs")
            
        except Exception as e:
            print(f"❌ Error loading distance matrix: {e}")
            print("   Will fall back to haversine formula when needed")
    
    def load_dog_coordinates(self):
        """Load dog coordinates from map sheet for haversine fallback"""
        print("\n📍 Loading dog coordinates for fallback calculations...")
        
        try:
            # Get specific columns
            dog_ids = self.map_sheet.col_values(10)[1:]  # Column J (Dog ID)
            latitudes = self.map_sheet.col_values(4)[1:]  # Column D
            longitudes = self.map_sheet.col_values(5)[1:]  # Column E
            
            coords_loaded = 0
            for i in range(min(len(dog_ids), len(latitudes), len(longitudes))):
                if dog_ids[i] and latitudes[i] and longitudes[i]:
                    try:
                        lat = float(latitudes[i])
                        lon = float(longitudes[i])
                        self.dog_coordinates[dog_ids[i]] = (lat, lon)
                        coords_loaded += 1
                    except ValueError:
                        continue
            
            print(f"✅ Loaded coordinates for {coords_loaded} dogs")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load coordinates: {e}")
    
    def haversine_time(self, coord1, coord2):
        """Calculate time between two coordinates using haversine formula"""
        if coord1 == coord2:
            return 0.0
            
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Haversine formula
        R = 6371  # Earth's radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        distance_km = R * c
        distance_miles = distance_km * 0.621371
        
        # Rough estimate: 1 mile ≈ 2.5 minutes driving in suburban area
        time_minutes = distance_miles * 2.5
        
        return time_minutes
    
    def get_time_with_fallback(self, dog_id1, dog_id2):
        """Get time between two dogs, using matrix first, then fallback to haversine"""
        if dog_id1 == dog_id2:
            return 0.0
        
        # Try distance matrix first
        if dog_id1 in self.distance_matrix and dog_id2 in self.distance_matrix[dog_id1]:
            return self.distance_matrix[dog_id1][dog_id2]
        
        # Try reverse direction
        if dog_id2 in self.distance_matrix and dog_id1 in self.distance_matrix[dog_id2]:
            return self.distance_matrix[dog_id2][dog_id1]
        
        # Fallback to haversine if coordinates available
        if dog_id1 in self.dog_coordinates and dog_id2 in self.dog_coordinates:
            self.haversine_fallback_count += 1
            return self.haversine_time(
                self.dog_coordinates[dog_id1],
                self.dog_coordinates[dog_id2]
            )
        
        # If no data available, return a large time
        return float('inf')
    
    def safe_get_time(self, dog1_data, dog2_data):
        """Safely get time between two dogs with error handling"""
        try:
            dog1_id = dog1_data.get('dog_id', '')
            dog2_id = dog2_data.get('dog_id', '')
            
            if not dog1_id or not dog2_id:
                return float('inf')
                
            return self.get_time_with_fallback(dog1_id, dog2_id)
            
        except Exception:
            return float('inf')
    
    def load_dog_assignments(self):
        """Load current dog assignments from Google Sheets"""
        print("\n📋 Loading dog assignments...")
        
        try:
            # Get all data
            all_data = self.map_sheet.get_all_values()
            
            if len(all_data) < 2:
                print("❌ No dog data found")
                return
            
            # Find column indices
            headers = all_data[0]
            col_indices = {}
            
            # Map column names to indices
            for idx, header in enumerate(headers):
                if 'Dog Name' in header:
                    col_indices['dog_name'] = idx
                elif 'Combined' in header:
                    col_indices['combined'] = idx
                elif 'Dog ID' in header:
                    col_indices['dog_id'] = idx
                elif 'Callout' in header:
                    col_indices['callout'] = idx
            
            # Process each row
            for row_idx, row in enumerate(all_data[1:], 2):
                if len(row) > max(col_indices.values()):
                    dog_data = {
                        'row_index': row_idx,
                        'dog_name': row[col_indices.get('dog_name', 1)] if 'dog_name' in col_indices else '',
                        'combined': row[col_indices.get('combined', 7)] if 'combined' in col_indices else '',
                        'dog_id': row[col_indices.get('dog_id', 9)] if 'dog_id' in col_indices else '',
                        'callout': row[col_indices.get('callout', 10)] if 'callout' in col_indices else ''
                    }
                    
                    # Only add if has dog name and ID
                    if dog_data['dog_name'] and dog_data['dog_id']:
                        self.dog_assignments.append(dog_data)
                        
                        # Track active drivers
                        if dog_data['combined'] and ':' in dog_data['combined']:
                            driver = dog_data['combined'].split(':')[0]
                            if driver and driver not in ['Field', 'Parking']:
                                self.active_drivers.add(driver)
                                self.driver_assignment_counts[driver] += 1
            
            # Count assignments and callouts
            assigned = sum(1 for d in self.dog_assignments if d.get('combined'))
            callouts = sum(1 for d in self.dog_assignments 
                          if d.get('callout') and not d.get('combined'))
            
            print(f"✅ Loaded {len(self.dog_assignments)} dogs")
            print(f"   - {assigned} assigned")
            print(f"   - {callouts} callouts (unassigned)")
            print(f"   - {len(self.active_drivers)} active drivers")
            
        except Exception as e:
            print(f"❌ Error loading assignments: {e}")
            raise
    
    def calculate_driver_density(self, driver):
        """Calculate if a driver's route is dense and their capacity"""
        driver_dogs = []
        for assignment in self.dog_assignments:
            if (isinstance(assignment, dict) and 
                assignment.get('combined', '').startswith(f"{driver}:")):
                driver_dogs.append(assignment)
        
        if len(driver_dogs) < 2:
            return {
                'dog_count': len(driver_dogs),
                'avg_time': 0,
                'capacity': 8,  # Default capacity
                'is_dense': False
            }
        
        # Calculate average time between consecutive dogs
        total_time = 0
        time_count = 0
        
        for i in range(len(driver_dogs)):
            for j in range(i + 1, len(driver_dogs)):
                time = self.safe_get_time(driver_dogs[i], driver_dogs[j])
                if time < float('inf'):
                    total_time += time
                    time_count += 1
        
        avg_time = total_time / time_count if time_count > 0 else 0
        is_dense = avg_time < self.DENSE_ROUTE_THRESHOLD
        capacity = 12 if is_dense else 8
        
        return {
            'dog_count': len(driver_dogs),
            'avg_time': avg_time,
            'capacity': capacity,
            'is_dense': is_dense
        }
    
    def parse_dog_groups_from_callout(self, callout):
        """Parse group numbers from callout string"""
        if not callout:
            return []
        
        # Handle None type
        callout_str = str(callout) if callout else ""
        
        # Remove leading colon if present
        callout_clean = callout_str.lstrip(':')
        
        groups = []
        for char in callout_clean:
            if char in ['1', '2', '3']:
                group_num = int(char)
                if group_num not in groups:
                    groups.append(group_num)
        
        return sorted(groups)
    
    def optimize_routes(self):
        """Run the complete optimization with force assignment and dynamic capacity handling"""
        print("\n🚀 Starting route optimization with force assignment...")
        print("=" * 60)
        
        # Track all moves
        total_moves = 0
        
        # PHASE 1: Initial clustering of existing dogs
        print("\n📍 Initial clustering...")
        phase1_moves = self.phase1_cluster_existing_dogs()
        total_moves += phase1_moves
        
        # PHASE 2: Initial outlier removal
        print("\n📍 Initial outlier removal...")
        phase2_moves = self.phase2_remove_outliers_from_existing()
        total_moves += phase2_moves
        
        # PHASE 3: FORCE ASSIGN ALL CALLOUTS (ignore capacity)
        print("\n📍 Force assigning all callouts...")
        phase3_moves = self.phase3_force_assign_all_callouts()
        total_moves += phase3_moves
        
        # REPEAT PHASE 1: Re-cluster after all dogs are assigned
        print("\n🔄 RE-CLUSTERING after force assignment...")
        phase1_repeat_moves = self.phase1_cluster_existing_dogs()
        total_moves += phase1_repeat_moves
        
        # REPEAT PHASE 2: Remove new outliers created by force assignment
        print("\n🔄 RE-CHECKING OUTLIERS after force assignment...")
        phase2_repeat_moves = self.phase2_remove_outliers_from_existing()
        total_moves += phase2_repeat_moves
        
        # PHASE 4: Consolidate small drivers
        print("\n📍 Consolidating small drivers...")
        phase4_moves = self.phase4_consolidate_small_drivers()
        total_moves += phase4_moves
        
        # PHASE 5: Consolidate small groups
        print("\n📍 Consolidating small groups...")
        phase5_moves = self.phase5_consolidate_small_groups_constrained()
        total_moves += phase5_moves
        
        # PHASE 6: Final outlier sweep
        print("\n📍 Final outlier sweep...")
        phase6_moves = self.phase6_final_outlier_sweep()
        total_moves += phase6_moves
        
        # PHASE 7: DYNAMIC CAPACITY HANDLING WITH CASCADING
        print("\n📍 Handling over-capacity with dynamic cascading...")
        phase7_moves = self.phase7_dynamic_handle_over_capacity()
        total_moves += phase7_moves
        
        # Summary
        print("\n" + "=" * 60)
        print("🏁 OPTIMIZATION COMPLETE!")
        print("=" * 60)
        print(f"Phase 1 (Initial clustering): {phase1_moves} moves")
        print(f"Phase 2 (Initial outlier removal): {phase2_moves} moves")
        print(f"Phase 3 (Force assign callouts): {phase3_moves} assignments")
        print(f"Phase 1 Repeat (Re-clustering): {phase1_repeat_moves} moves")
        print(f"Phase 2 Repeat (Re-check outliers): {phase2_repeat_moves} moves")
        print(f"Phase 4 (Small drivers): {phase4_moves} moves")
        print(f"Phase 5 (Small groups): {phase5_moves} moves")
        print(f"Phase 6 (Final sweep): {phase6_moves} moves")
        print(f"Phase 7 (Dynamic capacity): {phase7_moves} moves")
        print(f"\n📊 Total optimizations: {total_moves}")
        
        return total_moves
    
    def phase1_cluster_existing_dogs(self):
        """Phase 1: Cluster nearby dogs from different drivers"""
        print("\n🔗 PHASE 1: Clustering nearby dogs")
        print("=" * 60)
        
        moves_made = 0
        
        # Process each group separately
        for group_num in [1, 2, 3]:
            print(f"\n📊 Processing Group {group_num}...")
            
            # Find all dogs in this group across all drivers
            dogs_by_driver = defaultdict(list)
            
            for assignment in self.dog_assignments:
                if not isinstance(assignment, dict) or not assignment.get('combined'):
                    continue
                    
                combined = assignment['combined']
                if ':' not in combined:
                    continue
                    
                driver = combined.split(':')[0]
                if driver not in self.active_drivers:
                    continue
                    
                # Parse groups from the combined field
                groups = self.parse_dog_groups_from_callout(combined.split(':', 1)[1])
                if group_num in groups:
                    dogs_by_driver[driver].append(assignment)
            
            # Find clusters using BFS
            processed = set()
            
            for driver_name, driver_dogs in dogs_by_driver.items():
                for dog in driver_dogs:
                    dog_id = dog.get('dog_id', '')
                    if dog_id in processed:
                        continue
                    
                    # Find all dogs within cluster threshold
                    cluster = []
                    queue = [dog]
                    cluster_dogs = {dog_id}
                    
                    while queue:
                        current = queue.pop(0)
                        current_id = current.get('dog_id', '')
                        cluster.append(current)
                        
                        # Check all other dogs
                        for other_driver in dogs_by_driver:
                            for other_dog in dogs_by_driver[other_driver]:
                                other_id = other_dog.get('dog_id', '')
                                
                                if (other_id not in cluster_dogs and 
                                    other_id not in processed):
                                    
                                    time = self.get_time_with_fallback(current_id, other_id)
                                    
                                    if time <= self.CLUSTER_THRESHOLD:
                                        queue.append(other_dog)
                                        cluster_dogs.add(other_id)
                    
                    # Mark all as processed
                    processed.update(cluster_dogs)
                    
                    # If cluster spans multiple drivers, consolidate
                    cluster_drivers = {}
                    for dog in cluster:
                        combined = dog.get('combined', '')
                        if ':' in combined:
                            driver = combined.split(':')[0]
                            if driver not in cluster_drivers:
                                cluster_drivers[driver] = []
                            cluster_drivers[driver].append(dog)
                    
                    if len(cluster_drivers) > 1:
                        # Find driver with most dogs in cluster
                        best_driver = max(cluster_drivers.keys(), 
                                        key=lambda d: len(cluster_drivers[d]))
                        
                        # Check capacity before moving
                        capacity_info = self.calculate_driver_density(best_driver)
                        current_count = capacity_info['dog_count']
                        capacity = capacity_info['capacity']
                        
                        # Move dogs from other drivers to best driver
                        for driver in cluster_drivers:
                            if driver == best_driver:
                                continue
                                
                            for dog in cluster_drivers[driver]:
                                if current_count < capacity:
                                    dog_name = dog.get('dog_name', 'Unknown')
                                    print(f"   ✅ Clustering: {dog_name} from {driver} → {best_driver}")
                                    
                                    # Update assignment
                                    old_combined = dog.get('combined', '')
                                    if ':' in old_combined:
                                        group_part = old_combined.split(':', 1)[1]
                                        dog['combined'] = f"{best_driver}:{group_part}"
                                        moves_made += 1
                                        current_count += 1
        
        print(f"\n✅ Phase 1 complete: {moves_made} dogs clustered")
        return moves_made
    
    def phase2_remove_outliers_from_existing(self):
        """Phase 2: Remove outliers from existing assignments"""
        print("\n🎯 PHASE 2: Removing outliers from existing assignments")
        print("=" * 60)
        
        moves_made = 0
        
        for driver in list(self.active_drivers):
            # Get all dogs for this driver
            driver_dogs = []
            for assignment in self.dog_assignments:
                if (isinstance(assignment, dict) and 
                    assignment.get('combined', '').startswith(f"{driver}:")):
                    driver_dogs.append(assignment)
            
            if len(driver_dogs) < 3:  # Need at least 3 to identify outliers
                continue
            
            # Group by time groups
            groups = defaultdict(list)
            for dog in driver_dogs:
                combined = dog.get('combined', '')
                if ':' in combined:
                    group_nums = self.parse_dog_groups_from_callout(combined.split(':', 1)[1])
                    for g in group_nums:
                        groups[g].append(dog)
            
            # Check each group for outliers
            for group_num, group_dogs in groups.items():
                if len(group_dogs) < 3:
                    continue
                
                # Calculate distances between all pairs
                for i, dog in enumerate(group_dogs):
                    dog_id = dog.get('dog_id', '')
                    if not dog_id:
                        continue
                    
                    distances = []
                    for j, other_dog in enumerate(group_dogs):
                        if i != j:
                            other_id = other_dog.get('dog_id', '')
                            if other_id:
                                dist = self.get_time_with_fallback(dog_id, other_id)
                                if dist < float('inf'):
                                    distances.append(dist)
                    
                    if not distances:
                        continue
                    
                    avg_distance = sum(distances) / len(distances)
                    min_distance = min(distances)
                    
                    # Check if outlier
                    is_outlier = (avg_distance > self.OUTLIER_MULTIPLIER * 
                                 (sum(distances) / len(distances)) or
                                 avg_distance > self.OUTLIER_ABSOLUTE or
                                 min_distance > self.OUTLIER_ABSOLUTE)
                    
                    if is_outlier:
                        # Find better placement
                        dog_name = dog.get('dog_name', 'Unknown')
                        
                        # Find nearest neighbor in other groups
                        min_distances = {}
                        
                        for g in [1, 2, 3]:
                            if g == group_num:
                                continue
                            
                            min_dist = float('inf')
                            for assignment in self.dog_assignments:
                                if not isinstance(assignment, dict):
                                    continue
                                combined = assignment.get('combined', '')
                                if not combined or not combined.startswith(f"{driver}:"):
                                    continue
                                if ':' not in combined:
                                    continue
                                other_groups = self.parse_dog_groups_from_callout(
                                    combined.split(':', 1)[1])
                                if g in other_groups:
                                    other_id = assignment.get('dog_id', '')
                                    if other_id and other_id != dog_id:
                                        dist = self.get_time_with_fallback(dog_id, other_id)
                                        min_dist = min(min_dist, dist)
                            
                            if min_dist < float('inf'):
                                min_distances[g] = min_dist
                        
                        # Also check other drivers
                        for other_driver in self.active_drivers:
                            if other_driver == driver:
                                continue
                            
                            min_dist = float('inf')
                            closest_dog = None
                            
                            for assignment in self.dog_assignments:
                                if (isinstance(assignment, dict) and 
                                    assignment.get('combined', '').startswith(f"{other_driver}:") and
                                    ':' in assignment.get('combined', '')):
                                    other_groups = self.parse_dog_groups_from_callout(
                                        assignment['combined'].split(':', 1)[1])
                                    if group_num in other_groups:
                                        other_id = assignment.get('dog_id', '')
                                        if other_id:
                                            dist = self.get_time_with_fallback(dog_id, other_id)
                                            if dist < min_dist:
                                                min_dist = dist
                                                closest_dog = assignment
                            
                            if min_dist < avg_distance * 0.5:  # Significantly closer
                                # Check capacity
                                capacity_info = self.calculate_driver_density(other_driver)
                                if capacity_info['dog_count'] < capacity_info['capacity']:
                                    print(f"   ✅ Moving outlier {dog_name}: "
                                          f"{driver} → {other_driver} "
                                          f"(closer by {avg_distance - min_dist:.1f} min)")
                                    
                                    dog['combined'] = f"{other_driver}:{group_num}"
                                    moves_made += 1
                                    break
        
        print(f"\n✅ Phase 2 complete: {moves_made} outliers removed")
        return moves_made
    
    def phase3_force_assign_all_callouts(self):
        """Phase 3: Force assign ALL callouts regardless of capacity"""
        print("\n💪 PHASE 3: Force assigning ALL callouts (ignoring capacity)")
        print("=" * 60)
        
        # Find all unassigned dogs with callouts
        callouts = []
        for assignment in self.dog_assignments:
            if isinstance(assignment, dict):
                callout = assignment.get('callout', '').strip()
                combined = assignment.get('combined', '').strip()
                
                if callout and not combined:
                    callouts.append(assignment)
        
        print(f"\n📊 Found {len(callouts)} unassigned dogs to force assign")
        
        assignments_made = 0
        
        for callout in callouts:
            dog_name = callout.get('dog_name', 'Unknown')
            dog_id = callout.get('dog_id', '')
            original_callout = callout.get('callout', '').strip()
            
            print(f"\n🐕 Force assigning {dog_name} (Callout: {original_callout})")
            
            # Parse what groups this dog needs
            callout_groups = self.parse_dog_groups_from_callout(original_callout)
            if not callout_groups:
                print(f"   ⚠️ No valid groups in callout")
                continue
                
            # Find ALL neighbors with matching groups
            neighbors = []
            
            for assignment in self.dog_assignments:
                if isinstance(assignment, dict) and assignment.get('combined', '').strip():
                    combined = assignment.get('combined', '')
                    if ':' not in combined:
                        continue
                        
                    neighbor_driver = combined.split(':')[0]
                    if neighbor_driver not in self.active_drivers:
                        continue
                        
                    # Parse neighbor's groups
                    neighbor_groups = self.parse_dog_groups_from_callout(
                        combined.split(':', 1)[1] if ':' in combined else ''
                    )
                    
                    # Check if any groups match
                    matching_groups = set(callout_groups) & set(neighbor_groups)
                    if matching_groups:
                        neighbor_id = assignment.get('dog_id', '')
                        if neighbor_id and dog_id:
                            time_to_neighbor = self.get_time_with_fallback(dog_id, neighbor_id)
                            
                            neighbors.append({
                                'driver': neighbor_driver,
                                'time': time_to_neighbor,
                                'dog_name': assignment.get('dog_name', 'Unknown'),
                                'dog_id': neighbor_id,
                                'matching_groups': matching_groups
                            })
            
            if not neighbors:
                print(f"   ❌ No neighbors found with matching groups")
                continue
                
            # Sort by distance and assign to the closest
            neighbors.sort(key=lambda x: x['time'])
            
            chosen_neighbor = neighbors[0]
            best_driver = chosen_neighbor['driver']
            
            # Determine which group(s) to assign
            if len(callout_groups) == 1:
                group_assignment = str(callout_groups[0])
            else:
                # If multiple groups needed and neighbor has multiple, match them
                available_groups = list(chosen_neighbor['matching_groups'])
                if len(available_groups) == 1:
                    group_assignment = str(available_groups[0])
                else:
                    # Assign to all matching groups
                    group_assignment = ''.join(map(str, sorted(available_groups)))
            
            # FORCE ASSIGN - ignore capacity
            new_combined = f"{best_driver}:{group_assignment}"
            callout['combined'] = new_combined
            
            print(f"   ✅ ASSIGNED to {best_driver} (closest to {chosen_neighbor['dog_name']}, "
                  f"{chosen_neighbor['time']:.1f} min away)")
            print(f"   📍 Groups: {group_assignment}")
            print(f"   ⚠️ Capacity check: IGNORED (force assignment)")
            
            # Update assignment on sheet
            self._update_assignment_on_sheet(callout)
            assignments_made += 1
        
        print(f"\n✅ Force assigned {assignments_made} dogs (out of {len(callouts)} callouts)")
        
        # Show any that couldn't be assigned
        unassigned = len(callouts) - assignments_made
        if unassigned > 0:
            print(f"⚠️ {unassigned} dogs had no valid neighbors to assign to")
        
        return assignments_made
    
    def phase4_consolidate_small_drivers(self):
        """Phase 4: Consolidate drivers with too few dogs"""
        print("\n🔄 PHASE 4: Consolidating small drivers")
        print("=" * 60)
        
        # Update driver counts
        self._update_driver_assignment_counts()
        
        # Find drivers with too few dogs
        small_drivers = []
        for driver, count in self.driver_assignment_counts.items():
            if count < self.MIN_DOGS_FOR_DRIVER and count > 0:
                small_drivers.append((driver, count))
        
        if not small_drivers:
            print("✅ No small drivers to consolidate")
            return 0
        
        small_drivers.sort(key=lambda x: x[1])  # Sort by count
        print(f"\n📊 Found {len(small_drivers)} drivers with < {self.MIN_DOGS_FOR_DRIVER} dogs")
        
        moves_made = 0
        
        for small_driver, count in small_drivers:
            print(f"\n🚚 Consolidating {small_driver} ({count} dogs)...")
            
            # Get all dogs for this driver
            dogs_to_move = []
            for assignment in self.dog_assignments:
                if (isinstance(assignment, dict) and 
                    assignment.get('combined', '').startswith(f"{small_driver}:")):
                    dogs_to_move.append(assignment)
            
            # Find best destinations for each dog
            for dog in dogs_to_move:
                dog_name = dog.get('dog_name', 'Unknown')
                dog_id = dog.get('dog_id', '')
                combined = dog.get('combined', '')
                
                if not dog_id or ':' not in combined:
                    continue
                
                groups = self.parse_dog_groups_from_callout(combined.split(':', 1)[1])
                
                # Find nearest dog in another driver
                best_option = None
                min_time = float('inf')
                
                for assignment in self.dog_assignments:
                    if not isinstance(assignment, dict):
                        continue
                    other_combined = assignment.get('combined', '')
                    if not other_combined or not other_combined.strip():
                        continue
                    if ':' not in other_combined:
                        continue
                    other_driver = other_combined.split(':')[0]
                    
                    if other_driver == small_driver or other_driver not in self.active_drivers:
                        continue
                    
                    # Check if groups match
                    other_groups = self.parse_dog_groups_from_callout(
                        other_combined.split(':', 1)[1])
                    
                    if any(g in other_groups for g in groups):
                        other_id = assignment.get('dog_id', '')
                        if other_id:
                            time = self.get_time_with_fallback(dog_id, other_id)
                            if time < min_time:
                                min_time = time
                                best_option = {
                                    'driver': other_driver,
                                    'time': time,
                                    'neighbor': assignment.get('dog_name', 'Unknown')
                                }
                
                if best_option and self.driver_assignment_counts[best_option['driver']] >= self.MIN_DOGS_FOR_DRIVER:
                    # Update assignment
                    group_str = ''.join(map(str, groups))
                    dog['combined'] = f"{best_option['driver']}:{group_str}"
                    
                    print(f"   ✅ {dog_name} → {best_option['driver']} "
                          f"(near {best_option['neighbor']}, {best_option['time']:.1f} min)")
                    
                    moves_made += 1
            
            # Remove driver from active set
            self.active_drivers.discard(small_driver)
            self.driver_assignment_counts[small_driver] = 0
        
        print(f"\n✅ Phase 4 complete: {moves_made} dogs redistributed")
        return moves_made
    
    def phase5_consolidate_small_groups_constrained(self):
        """Phase 5: Consolidate small groups with constraints"""
        print("\n🔄 PHASE 5: Consolidating small groups (with constraints)")
        print("=" * 60)
        
        moves_made = 0
        
        # Only process drivers with all 3 groups
        for driver in list(self.active_drivers):
            # Count dogs in each group
            driver_groups = {1: [], 2: [], 3: []}
            
            for assignment in self.dog_assignments:
                if (isinstance(assignment, dict) and 
                    assignment.get('combined', '').startswith(f"{driver}:")):
                    combined = assignment.get('combined', '')
                    if ':' in combined:
                        groups = self.parse_dog_groups_from_callout(combined.split(':', 1)[1])
                        for g in groups:
                            if g in driver_groups:
                                driver_groups[g].append(assignment)
            
            # Check if driver has all 3 groups
            has_all_groups = all(len(driver_groups[g]) > 0 for g in [1, 2, 3])
            
            if not has_all_groups:
                continue
            
            # Check Group 1 and Group 3 for small size
            for group_num in [1, 3]:
                if group_num not in driver_groups:
                    continue
                
                if len(driver_groups[group_num]) >= self.MIN_GROUP_SIZE:
                    continue
                    
                dogs_to_move = driver_groups[group_num]
                if not dogs_to_move:
                    continue
                    
                print(f"\n📊 {driver} Group {group_num} has only {len(dogs_to_move)} dogs")
                
                # Try to move each dog
                group_moves = 0
                for dog in dogs_to_move:
                    dog_name = dog.get('dog_name', 'Unknown')
                    dog_id = dog.get('dog_id', '')
                    
                    if not dog_id:
                        continue
                    
                    # Find all potential destinations
                    options = []
                    
                    for other_driver in self.active_drivers:
                        if other_driver == driver:
                            continue
                        
                        # Get capacity info
                        capacity_info = self.calculate_driver_density(other_driver)
                        capacity = capacity_info['capacity']
                        
                        # Count dogs in this group for other driver
                        other_group_dogs = []
                        for assignment in self.dog_assignments:
                            if not isinstance(assignment, dict):
                                continue
                            combined = assignment.get('combined', '')
                            if not combined.startswith(f"{other_driver}:"):
                                continue
                            if ':' not in combined:
                                continue
                            parts = combined.split(':', 1)
                            if len(parts) >= 2:
                                groups = self.parse_dog_groups_from_callout(parts[1])
                                if group_num in groups:
                                    other_group_dogs.append(assignment)
                        
                        available_capacity = capacity - len(other_group_dogs)
                        
                        # Only process if there's capacity
                        if available_capacity > 0:
                            # Find closest dog in other driver's group
                            min_time = float('inf')
                            closest_dog_name = None
                            
                            if other_group_dogs:
                                for other_dog in other_group_dogs:
                                    other_id = other_dog.get('dog_id', '')
                                    if other_id and dog_id:
                                        time = self.get_time_with_fallback(dog_id, other_id)
                                        if time < min_time:
                                            min_time = time
                                            closest_dog_name = other_dog.get('dog_name', 'Unknown')
                            else:
                                # Empty group - use default time
                                min_time = 5.0
                                closest_dog_name = "Empty group"
                            
                            if min_time < float('inf'):
                                options.append({
                                    'driver': other_driver,
                                    'time': min_time,
                                    'capacity': available_capacity,
                                    'closest_dog': closest_dog_name
                                })
                    
                    if options:
                        # Sort by time
                        options.sort(key=lambda x: x['time'])
                        
                        # Find all options within 1 minute of best
                        best_time = options[0]['time']
                        tied_options = [o for o in options 
                                      if o['time'] <= best_time + 1.0]
                        
                        # Among tied options, pick one with most capacity
                        if tied_options:
                            best_option = max(tied_options, key=lambda x: x['capacity'])
                            
                            # Check time increase constraint
                            time_increase = best_option['time']
                            
                            if time_increase < self.GROUP_CONSOLIDATION_TIME_LIMIT:
                                # Update assignment
                                new_combined = f"{best_option['driver']}:{group_num}"
                                dog['combined'] = new_combined
                                
                                print(f"   ✅ {dog_name} → {best_option['driver']} "
                                      f"(near {best_option['closest_dog']}, "
                                      f"{best_option['time']:.1f} min)")
                                
                                moves_made += 1
                                group_moves += 1
                            else:
                                print(f"   ❌ {dog_name}: Would increase time by {time_increase:.1f} min "
                                      f"(exceeds {self.GROUP_CONSOLIDATION_TIME_LIMIT} min limit)")
                        else:
                            print(f"   ❌ {dog_name}: No suitable destinations found")
                    else:
                        print(f"   ❌ {dog_name}: No destinations with capacity found")
                
                if group_moves == 0:
                    print(f"   ⚠️ Could not move any dogs from this small group")
        
        print(f"\n✅ Phase 5 complete: {moves_made} dogs moved")
        return moves_made
    
    def phase6_final_outlier_sweep(self):
        """Phase 6: Final sweep for any remaining outliers"""
        print("\n🧹 PHASE 6: Final outlier sweep across all groups")
        print("=" * 60)
        
        # Find ALL outliers across all groups
        all_outliers = []
        
        for driver in self.active_drivers:
            # Group dogs by their groups
            driver_groups = defaultdict(list)
            
            for assignment in self.dog_assignments:
                if (isinstance(assignment, dict) and 
                    assignment.get('combined', '').startswith(f"{driver}:")):
                    combined = assignment.get('combined', '')
                    if ':' in combined:
                        groups = self.parse_dog_groups_from_callout(combined.split(':', 1)[1])
                        for g in groups:
                            driver_groups[g].append(assignment)
            
            # Check each group for outliers
            for group_num, dogs in driver_groups.items():
                if len(dogs) < 3:  # Need at least 3 to identify outliers
                    continue
                
                # Analyze within-group times
                for i, dog in enumerate(dogs):
                    dog_id = dog.get('dog_id', '')
                    if not dog_id:
                        continue
                    
                    distances = []
                    for j, other_dog in enumerate(dogs):
                        if i != j:
                            other_id = other_dog.get('dog_id', '')
                            if other_id:
                                dist = self.get_time_with_fallback(dog_id, other_id)
                                if dist < float('inf'):
                                    distances.append(dist)
                    
                    if not distances:
                        continue
                    
                    avg_distance = sum(distances) / len(distances)
                    min_distance = min(distances)
                    
                    # More conservative criteria for final sweep
                    if (avg_distance > 2 * statistics.median(distances) or
                        avg_distance > 4 or
                        min_distance > 4):
                        
                        all_outliers.append({
                            'dog': dog,
                            'driver': driver,
                            'group': group_num,
                            'avg_distance': avg_distance,
                            'min_distance': min_distance,
                            'severity': avg_distance / (statistics.median(distances) + 0.1)
                        })
        
        if not all_outliers:
            print("✅ No significant outliers found")
            return 0
        
        # Sort by severity
        all_outliers.sort(key=lambda x: x['severity'], reverse=True)
        
        print(f"\n🔍 Found {len(all_outliers)} outliers across all groups")
        print("\nTop outliers:")
        for outlier_data in all_outliers[:5]:
            dog_name = outlier_data['dog'].get('dog_name', 'Unknown')
            print(f"   - {dog_name} ({outlier_data['driver']} G{outlier_data['group']}): "
                  f"avg {outlier_data['avg_distance']:.1f} min, "
                  f"nearest {outlier_data['min_distance']:.1f} min "
                  f"(severity: {outlier_data['severity']:.1f}x)")
        
        moves_made = 0
        max_moves = 10  # Limit final sweep moves
        
        for outlier_data in all_outliers[:max_moves]:
            dog = outlier_data['dog']
            current_driver = outlier_data['driver']
            group_num = outlier_data['group']
            dog_name = dog.get('dog_name', 'Unknown')
            dog_id = dog.get('dog_id', '')
            
            # Find best alternative placement
            best_dest = None
            best_improvement = 0
            
            for other_driver in self.active_drivers:
                if other_driver == current_driver:
                    continue
                
                # Get capacity info
                capacity_info = self.calculate_driver_density(other_driver)
                capacity = capacity_info['capacity']
                
                # Count current dogs in this group
                current_count = 0
                for a in self.dog_assignments:
                    if not isinstance(a, dict):
                        continue
                    combined = a.get('combined', '')
                    if not combined.startswith(f"{other_driver}:"):
                        continue
                    if ':' not in combined:
                        continue
                    parts = combined.split(':', 1)
                    if len(parts) >= 2:
                        groups = self.parse_dog_groups_from_callout(parts[1])
                        if group_num in groups:
                            current_count += 1
                
                available_capacity = max(0, capacity - current_count)
                
                # Check for cross-group proximity
                nearby_dogs = 0
                min_cross_group_dist = float('inf')
                
                for g in [1, 2, 3]:
                    if g == group_num:
                        continue
                    
                    for assignment in self.dog_assignments:
                        if not isinstance(assignment, dict):
                            continue
                        combined = assignment.get('combined', '')
                        if not combined.startswith(f"{other_driver}:"):
                            continue
                        if ':' not in combined:
                            continue
                        other_groups = self.parse_dog_groups_from_callout(
                            combined.split(':', 1)[1])
                        if g in other_groups:
                            other_id = assignment.get('dog_id', '')
                            if other_id:
                                dist = self.get_time_with_fallback(dog_id, other_id)
                                if dist < 3:  # Close enough
                                    nearby_dogs += 1
                                min_cross_group_dist = min(min_cross_group_dist, dist)
                
                # Calculate improvement
                improvement = outlier_data['avg_distance'] - min_cross_group_dist
                
                if improvement > 1 and (available_capacity > 0 or nearby_dogs > 0):
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_dest = {
                            'driver': other_driver,
                            'improvement': improvement,
                            'capacity': available_capacity,
                            'nearby_dogs': nearby_dogs
                        }
            
            if best_dest and best_improvement > 1:
                # Move the outlier
                new_combined = f"{best_dest['driver']}:{group_num}"
                dog['combined'] = new_combined
                
                improvement_pct = (best_dest['improvement'] / outlier_data['avg_distance']) * 100 if outlier_data['avg_distance'] > 0 else 0
                
                print(f"\n✅ Moved outlier {dog_name}:")
                print(f"   From: {current_driver} (avg {outlier_data['avg_distance']:.1f} min)")
                print(f"   To: {best_dest['driver']} (avg {outlier_data['avg_distance'] - best_dest['improvement']:.1f} min)")
                print(f"   Improvement: {best_dest['improvement']:.1f} min ({improvement_pct:.0f}% better)")
                
                moves_made += 1
        
        print(f"\n✅ Phase 6 complete: {moves_made} outliers moved")
        return moves_made
    
    def phase7_dynamic_handle_over_capacity(self):
        """Phase 7: Handle over-capacity with dynamic cascading"""
        print("\n⚖️ PHASE 7: Dynamic over-capacity handling with cascading")
        print("=" * 60)
        
        moves_made = 0
        max_iterations = 100  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Find the first over-capacity group
            over_capacity_group = self._find_over_capacity_group()
            
            if not over_capacity_group:
                print(f"\n✅ All groups within capacity after {moves_made} moves!")
                break
                
            driver = over_capacity_group['driver']
            group_num = over_capacity_group['group_num']
            current_count = over_capacity_group['current_count']
            capacity = over_capacity_group['capacity']
            dogs = over_capacity_group['dogs']
            
            print(f"\n🔍 Iteration {iteration}: {driver} Group {group_num} is over capacity ({current_count}/{capacity})")
            
            # Find outliers in this group (dogs with no close neighbors)
            outliers = self._find_outliers_by_connectivity(dogs, group_num)
            
            if not outliers:
                print(f"   ⚠️ No outliers found in over-capacity group. Breaking to avoid infinite loop.")
                break
                
            # Sort outliers by how isolated they are
            outliers.sort(key=lambda x: x['isolation_score'], reverse=True)
            
            # Move the most isolated outlier
            outlier = outliers[0]
            dog_name = outlier['dog']['dog_name']
            dog_id = outlier['dog']['dog_id']
            
            print(f"   🎯 Moving outlier: {dog_name} (isolation score: {outlier['isolation_score']:.1f})")
            
            # Find best destination
            best_destination = self._find_best_destination_dynamic(
                outlier['dog'], group_num, current_driver=driver
            )
            
            if not best_destination:
                print(f"   ❌ No valid destination found for {dog_name}")
                continue
                
            # Execute the move
            new_driver = best_destination['driver']
            new_combined = f"{new_driver}:{group_num}"
            
            print(f"   ✅ Moving {dog_name}: {driver} → {new_driver}")
            print(f"      Distance: {best_destination['distance']:.1f} min")
            print(f"      {new_driver}'s available capacity: {best_destination['available_capacity']}")
            
            # Update the assignment
            outlier['dog']['combined'] = new_combined
            self._update_assignment_on_sheet(outlier['dog'])
            moves_made += 1
            
            # CRITICAL: Re-evaluate everything after this move
            print(f"   🔄 Re-evaluating all groups after move...")
            
        if iteration >= max_iterations:
            print(f"\n⚠️ Reached maximum iterations ({max_iterations}). Stopping to prevent infinite loop.")
            
        return moves_made
    
    def _find_over_capacity_group(self):
        """Find the first over-capacity group"""
        for driver in self.active_drivers:
            capacity_info = self.calculate_driver_density(driver)
            capacity = capacity_info['capacity']
            
            # Check each group
            for group_num in [1, 2, 3]:
                dogs = []
                for assignment in self.dog_assignments:
                    if not isinstance(assignment, dict):
                        continue
                    combined = assignment.get('combined', '')
                    if not combined.startswith(f"{driver}:"):
                        continue
                    if ':' not in combined:
                        continue
                    combined_parts = combined.split(':', 1)
                    if len(combined_parts) >= 2:
                        groups = self.parse_dog_groups_from_callout(combined_parts[1])
                        if group_num in groups:
                            dogs.append(assignment)
                
                current_count = len(dogs)
                if current_count > capacity:
                    return {
                        'driver': driver,
                        'group_num': group_num,
                        'current_count': current_count,
                        'capacity': capacity,
                        'dogs': dogs
                    }
        
        return None
    
    def _find_outliers_by_connectivity(self, dogs, group_num):
        """Find outliers based on connectivity (dogs with no close neighbors)"""
        outliers = []
        CLOSE_NEIGHBOR_THRESHOLD = 3.0  # minutes
        
        for i, dog in enumerate(dogs):
            dog_id = dog.get('dog_id', '')
            if not dog_id:
                continue
                
            # Count close neighbors
            close_neighbors = 0
            min_neighbor_distance = float('inf')
            
            for j, other_dog in enumerate(dogs):
                if i == j:
                    continue
                    
                other_id = other_dog.get('dog_id', '')
                if not other_id:
                    continue
                    
                distance = self.get_time_with_fallback(dog_id, other_id)
                
                if distance < CLOSE_NEIGHBOR_THRESHOLD:
                    close_neighbors += 1
                    
                if distance < min_neighbor_distance:
                    min_neighbor_distance = distance
            
            # Calculate isolation score
            isolation_score = (1 / (close_neighbors + 1)) * min_neighbor_distance
            
            if close_neighbors == 0:  # True outlier with no close neighbors
                outliers.append({
                    'dog': dog,
                    'close_neighbors': close_neighbors,
                    'min_neighbor_distance': min_neighbor_distance,
                    'isolation_score': isolation_score
                })
        
        return outliers
    
    def _find_best_destination_dynamic(self, dog, group_num, current_driver):
        """Find best destination considering available space for tie-breaking"""
        dog_id = dog.get('dog_id', '')
        if not dog_id:
            return None
            
        destinations = []
        SIMILAR_DISTANCE_THRESHOLD = 1.0
        
        for driver in self.active_drivers:
            if driver == current_driver:
                continue
                
            # Get capacity info
            capacity_info = self.calculate_driver_density(driver)
            capacity = capacity_info['capacity']
            
            # Count current dogs in this group
            current_count = 0
            closest_dog = None
            min_distance = float('inf')
            
            for assignment in self.dog_assignments:
                if not isinstance(assignment, dict):
                    continue
                combined = assignment.get('combined', '')
                if not combined.startswith(f"{driver}:"):
                    continue
                if ':' not in combined:
                    continue
                combined_parts = combined.split(':', 1)
                if len(combined_parts) >= 2:
                    groups = self.parse_dog_groups_from_callout(combined_parts[1])
                    if group_num in groups:
                        current_count += 1
                        
                        # Find closest dog
                        other_id = assignment.get('dog_id', '')
                        if other_id:
                            distance = self.get_time_with_fallback(dog_id, other_id)
                            if distance < min_distance:
                                min_distance = distance
                                closest_dog = assignment.get('dog_name', 'Unknown')
            
            # If no dogs in group, estimate distance
            if min_distance == float('inf'):
                min_distance = 10.0
                
            available_capacity = capacity - current_count
            
            destinations.append({
                'driver': driver,
                'distance': min_distance,
                'available_capacity': available_capacity,
                'current_count': current_count,
                'closest_dog': closest_dog
            })
        
        if not destinations:
            return None
            
        # Sort by distance first
        destinations.sort(key=lambda x: x['distance'])
        
        # Find all destinations within similar distance
        best_distance = destinations[0]['distance']
        similar_destinations = [d for d in destinations 
                              if d['distance'] <= best_distance + SIMILAR_DISTANCE_THRESHOLD]
        
        # Among similar distances, pick the one with most available space
        if len(similar_destinations) > 1:
            similar_destinations.sort(key=lambda x: x['available_capacity'], reverse=True)
            
        return similar_destinations[0]
    
    def _update_driver_assignment_counts(self):
        """Update the count of dogs assigned to each driver"""
        self.driver_assignment_counts = defaultdict(int)
        for assignment in self.dog_assignments:
            if isinstance(assignment, dict) and assignment.get('combined'):
                combined = assignment['combined']
                if ':' in combined:
                    driver = combined.split(':')[0]
                    if driver in self.active_drivers:
                        self.driver_assignment_counts[driver] += 1
    
    def _update_assignment_on_sheet(self, dog_data):
        """Update a single assignment on the Google Sheet"""
        try:
            row_index = dog_data['row_index']
            combined_value = dog_data['combined']
            
            # Find the Combined column (usually column H, index 8)
            self.map_sheet.update_cell(row_index, 8, combined_value)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not update sheet for {dog_data.get('dog_name', 'Unknown')}: {e}")
    
    def write_results_to_sheets(self):
        """Write all optimized assignments back to Google Sheets"""
        print("\n💾 Saving results to Google Sheets...")
        
        try:
            # Prepare batch update
            updates = []
            combined_col_idx = 8  # Column H
            
            for assignment in self.dog_assignments:
                if isinstance(assignment, dict) and 'row_index' in assignment:
                    row_idx = assignment['row_index']
                    combined_value = assignment.get('combined', '')
                    
                    # Create cell reference
                    if combined_col_idx <= 26:
                        col_letter = chr(ord('A') + combined_col_idx - 1)
                    else:
                        # Handle columns beyond Z (AA, AB, etc.)
                        first_letter = chr(ord('A') + (combined_col_idx - 1) // 26 - 1)
                        second_letter = chr(ord('A') + (combined_col_idx - 1) % 26)
                        col_letter = first_letter + second_letter
                    
                    cell_ref = f"{col_letter}{row_idx}"
                    updates.append({
                        'range': cell_ref,
                        'values': [[combined_value]]
                    })
            
            # Batch update
            if updates:
                self.map_sheet.batch_update(updates)
                print(f"✅ Updated {len(updates)} assignments in Google Sheets")
            
        except Exception as e:
            print(f"❌ Error writing to sheets: {e}")
            # Fall back to individual updates
            print("   Attempting individual updates...")
            updated = 0
            for assignment in self.dog_assignments:
                if isinstance(assignment, dict) and 'row_index' in assignment:
                    try:
                        self._update_assignment_on_sheet(assignment)
                        updated += 1
                    except:
                        pass
            print(f"   ✅ Updated {updated} assignments individually")
    
    def run(self):
        """Main execution function"""
        try:
            # Load data
            self.load_distance_matrix()
            self.load_dog_coordinates()
            self.load_dog_assignments()
            
            # Run optimization
            total_moves = self.optimize_routes()
            
            # Save results
            self.write_results_to_sheets()
            
            # Print summary
            print("\n" + "=" * 60)
            print("✅ OPTIMIZATION COMPLETE!")
            print("=" * 60)
            print(f"Total changes made: {total_moves}")
            print(f"Results saved to Google Sheets")
            
            if hasattr(self, 'haversine_fallback_count') and self.haversine_fallback_count > 0:
                print(f"\n📍 Haversine Fallback Usage:")
                print(f"   Used {self.haversine_fallback_count} times for time calculations")
                print(f"   Consider updating distance matrix for better accuracy")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    system = DogReassignmentSystem()
    success = system.run()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
