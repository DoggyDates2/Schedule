#!/usr/bin/env python3
"""
Dog Assignment Optimization System - REVISED PHASE ORDER

NEW OPTIMIZATION STRATEGY (6 PHASES):
1. Phase 1: Cluster nearby dogs FIRST (< 1 min apart) - before any assignments
2. Phase 2: Remove outliers from existing assignments
3. Phase 3: Assign callouts intelligently with capacity management
4. Phase 4: Consolidate small drivers (< 7 total dogs)
5. Phase 5: Consolidate small groups with constraints
6. Phase 6: Final outlier sweep to clean up any remaining suboptimal assignments

KEY IMPROVEMENTS:
- Cluster BEFORE assigning (preserves natural groupings)
- Smarter callout assignment using neighbor availability
- Cascading logic for over-capacity situations
- Weighted scoring for placement decisions
- More conservative thresholds
- Final cleanup phase to catch any remaining outliers
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

try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    import requests
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("💡 Install with: pip install gspread oauth2client requests")
    sys.exit(1)

class DogReassignmentSystem:
    def __init__(self):
        print("🚀 Enhanced Dog Reassignment System - REVISED PHASE ORDER")
        print("   All distances are now in MINUTES of driving time")
        print("   Dense routes (< 2 min avg): 12 dogs per group")
        print("   Standard routes: 8 dogs per group")
        print("   Small drivers: < 7 dogs get consolidated")
        print("   Small groups: < 4 dogs in Group 1 or 3 get consolidated")
        print("   6-Phase optimization with clustering FIRST and final sweep LAST")
        
        # Google Sheets IDs
        self.MAP_SHEET_ID = "1-KTOfTKXk_sX7nO7eGmW73JLi8TJBvv5gobK6gyrc7U"
        self.DISTANCE_MATRIX_SHEET_ID = "1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg"
        self.MAP_TAB = "Map"
        self.MATRIX_TAB = "Matrix"
        
        # System parameters - NOW IN MINUTES
        self.PREFERRED_DISTANCE = 2      # 2 minutes preferred
        self.MAX_DISTANCE = 5           # 5 minutes max for normal moves
        self.ABSOLUTE_MAX_DISTANCE = 7  # 7 minutes = too far between dogs
        self.DETOUR_THRESHOLD = 7       # 7 minutes detour acceptable
        self.CASCADING_MOVE_MAX = 10    # 10 minutes for cascading moves
        self.ADJACENT_GROUP_DISTANCE = 2 # 2 minutes between adjacent groups
        self.EXCLUSION_DISTANCE = 100   # 100 minutes = placeholder for unreachable
        self.DENSE_ROUTE_THRESHOLD = 2  # Routes denser than 2 min get capacity 12
        self.OUTLIER_THRESHOLD = 5      # Dog is outlier if > 5 min from nearest neighbor
        self.CLUSTER_THRESHOLD = 1      # Dogs < 1 min apart should cluster together
        
        # Consolidation parameters
        self.MIN_DOGS_FOR_DRIVER = 7   # Drivers with fewer than 7 dogs get consolidated (reduced from 12)
        self.MIN_GROUP_SIZE = 4        # Minimum dogs to keep a Group 1 or Group 3
        self.CAPACITY_THRESHOLD = 2    # Minutes within which to consider equal
        self.OUTLIER_MULTIPLIER = 1.5  # Dog is outlier if > 1.5x average distance
        self.OUTLIER_ABSOLUTE = 3      # Dog is outlier if > 3 min from avg regardless of multiplier
        self.GROUP_CONSOLIDATION_TIME_LIMIT = 10  # Max time increase for group consolidation
        
        # Conversion for haversine fallback
        self.MILES_TO_MINUTES = 2.5    # 1 mile ≈ 2.5 minutes city driving
        
        # Initialize data structures
        self.distance_matrix = {}
        self.dog_assignments = []
        self.driver_capacities = {}
        self.dog_name_to_id = {}
        self.dog_id_to_name = {}
        self.driver_assignment_counts = defaultdict(int)
        self.active_drivers = set()
        self.optimization_swaps = []
        self.dog_coordinates = {}
        self.haversine_fallback_count = 0
        
        # Set up Google Sheets connection and load data
        self.setup_google_sheets()
        self.load_distance_matrix()
        self.load_dog_assignments()
        self.load_dog_coordinates()
    
    def setup_google_sheets(self):
        """Set up Google Sheets API connection"""
        try:
            scope = ['https://spreadsheets.google.com/feeds',
                    'https://www.googleapis.com/auth/drive']
            
            json_str = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
            if not json_str:
                raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON environment variable not set")
            
            creds_dict = json.loads(json_str)
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            self.gc = gspread.authorize(creds)
            print("✅ Google Sheets connection established")
        except Exception as e:
            print(f"❌ Failed to connect to Google Sheets: {e}")
            raise

    def load_distance_matrix(self):
        """Load time matrix from Google Sheets - NOW IN MINUTES"""
        try:
            spreadsheet = self.gc.open_by_key(self.DISTANCE_MATRIX_SHEET_ID)
            try:
                sheet = spreadsheet.worksheet(self.MATRIX_TAB)
                print(f"✅ Found sheet by name: '{self.MATRIX_TAB}'")
            except gspread.WorksheetNotFound:
                sheet = spreadsheet.get_worksheet_by_id(398422902)
                print(f"✅ Found sheet by ID: 398422902")
            
            all_values = sheet.get_all_values()
            
            if not all_values:
                print("❌ Time matrix sheet is empty")
                return
            
            # Get column headers from row 1, starting at B1
            column_headers = all_values[0]
            column_dog_ids = []
            for i in range(1, len(column_headers)):
                if column_headers[i].strip():
                    column_dog_ids.append(column_headers[i].strip())
            
            print(f"📋 Column headers (B1 onwards): {column_dog_ids[:5]}...")
            print(f"   Total column dogs: {len(column_dog_ids)}")
            
            # Build time matrix
            times_loaded = 0
            
            for row_index in range(1, len(all_values)):
                row = all_values[row_index]
                
                if len(row) == 0 or not row[0].strip():
                    continue
                    
                from_dog_id = row[0].strip()
                
                for col_index in range(1, min(len(row), len(column_dog_ids) + 1)):
                    if col_index < len(row) and row[col_index] and row[col_index].strip():
                        to_dog_id = column_dog_ids[col_index - 1]
                        
                        try:
                            time_minutes = float(row[col_index].strip())
                            
                            if from_dog_id not in self.distance_matrix:
                                self.distance_matrix[from_dog_id] = {}
                            
                            self.distance_matrix[from_dog_id][to_dog_id] = time_minutes
                            times_loaded += 1
                            
                        except ValueError:
                            pass
            
            print(f"✅ Loaded {times_loaded} driving times")
            print(f"✅ Matrix has {len(self.distance_matrix)} dogs with time data")
            
            if self.distance_matrix:
                sample_dog = list(self.distance_matrix.keys())[0]
                sample_times = list(self.distance_matrix[sample_dog].items())[:3]
                print(f"✅ Sample: {sample_dog} → {sample_times}")
                
        except Exception as e:
            print(f"❌ Error loading time matrix: {e}")
            import traceback
            traceback.print_exc()
            self.distance_matrix = {}

    def load_dog_coordinates(self):
        """Load dog coordinates from Map sheet for haversine fallback"""
        try:
            print("📍 Loading dog coordinates for time calculation fallback...")
            sheet = self.gc.open_by_key(self.MAP_SHEET_ID).worksheet(self.MAP_TAB)
            all_values = sheet.get_all_values()
            
            lat_idx = 3  # Column D
            lng_idx = 4  # Column E  
            dog_id_idx = 9  # Column J
            
            coordinates_loaded = 0
            for row in all_values[1:]:
                if len(row) > max(lat_idx, lng_idx, dog_id_idx):
                    try:
                        dog_id = row[dog_id_idx].strip() if row[dog_id_idx] else ""
                        lat_str = row[lat_idx].strip() if row[lat_idx] else ""
                        lng_str = row[lng_idx].strip() if row[lng_idx] else ""
                        
                        if dog_id and lat_str and lng_str:
                            lat = float(lat_str)
                            lng = float(lng_str)
                            if -90 <= lat <= 90 and -180 <= lng <= 180:  # Valid coordinates
                                self.dog_coordinates[dog_id] = (lat, lng)
                                coordinates_loaded += 1
                    except (ValueError, IndexError):
                        pass
            
            print(f"✅ Loaded coordinates for {coordinates_loaded} dogs")
            
        except Exception as e:
            print(f"❌ Error loading dog coordinates: {e}")
            self.dog_coordinates = {}

    def haversine_time(self, dog_id1, dog_id2):
        """Calculate driving time between two dogs using haversine formula"""
        if dog_id1 not in self.dog_coordinates or dog_id2 not in self.dog_coordinates:
            return float('inf')
        
        if dog_id1 == dog_id2:
            return 0.0
        
        lat1, lon1 = self.dog_coordinates[dog_id1]
        lat2, lon2 = self.dog_coordinates[dog_id2]
        
        # Check for same location
        if lat1 == lat2 and lon1 == lon2:
            return 0.0
        
        # Convert to radians
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        
        # Clamp to avoid numerical errors
        a = min(1.0, max(0.0, a))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        # Earth's radius in miles
        R = 3959
        distance_miles = R * c
        
        # Convert to driving time in minutes
        driving_factor = 1.3  # Account for roads vs straight line
        driving_miles = distance_miles * driving_factor
        driving_minutes = driving_miles * self.MILES_TO_MINUTES
        
        return max(0.0, driving_minutes)  # Ensure non-negative

    def get_time_with_fallback(self, dog_id1, dog_id2):
        """Get driving time between dogs with haversine fallback"""
        # First, try the time matrix
        time_minutes = self.safe_get_time(dog_id1, dog_id2)
        if time_minutes < float('inf'):
            return time_minutes
        
        # Fallback to haversine calculation
        haversine_time = self.haversine_time(dog_id1, dog_id2)
        if haversine_time < float('inf'):
            self.haversine_fallback_count += 1
            return haversine_time
        
        # Last resort
        return self.EXCLUSION_DISTANCE

    def safe_get_time(self, dog_id1, dog_id2):
        """Safely get driving time between two dogs from matrix"""
        if dog_id1 == dog_id2:
            return 0.0
        
        # Try multiple ID formats to handle 'x' suffix variations
        id1_variants = [dog_id1, dog_id1.rstrip('x'), f"{dog_id1}x"]
        id2_variants = [dog_id2, dog_id2.rstrip('x'), f"{dog_id2}x"]
        
        for id1 in id1_variants:
            for id2 in id2_variants:
                if id1 in self.distance_matrix and id2 in self.distance_matrix[id1]:
                    return self.distance_matrix[id1][id2]
                if id2 in self.distance_matrix and id1 in self.distance_matrix[id2]:
                    return self.distance_matrix[id2][id1]
        
        return float('inf')

    def load_dog_assignments(self):
        """Load dog assignments with correct column mappings"""
        try:
            print("📊 Loading assignments...")
            sheet = self.gc.open_by_key(self.MAP_SHEET_ID).worksheet(self.MAP_TAB)
            all_values = sheet.get_all_values()
            
            if not all_values:
                print("❌ No data found in assignments sheet")
                return
            
            # Column indices
            dog_name_idx = 1    # Column B
            combined_idx = 7    # Column H
            dog_id_idx = 9      # Column J  
            callout_idx = 10    # Column K
            
            # Load all dog data
            drivers_found = set()
            self.dog_assignments = []
            callouts_found = 0
            
            for row_idx, row in enumerate(all_values[1:], 2):
                if len(row) > max(dog_name_idx, combined_idx, dog_id_idx, callout_idx):
                    dog_name = row[dog_name_idx].strip() if row[dog_name_idx] else ""
                    combined = row[combined_idx].strip() if row[combined_idx] else ""
                    dog_id = row[dog_id_idx].strip() if row[dog_id_idx] else ""
                    callout = row[callout_idx].strip() if row[callout_idx] else ""
                    
                    if dog_name and dog_id:
                        assignment = {
                            'row_index': row_idx,
                            'dog_name': dog_name,
                            'combined': combined,
                            'dog_id': dog_id,
                            'callout': callout
                        }
                        self.dog_assignments.append(assignment)
                        
                        self.dog_name_to_id[dog_name] = dog_id
                        self.dog_id_to_name[dog_id] = dog_name
                        
                        if callout and not combined:
                            callouts_found += 1
                        
                        if combined and ':' in combined:
                            driver_name = combined.split(':')[0]
                            if driver_name not in ['Field', 'Parking']:
                                drivers_found.add(driver_name)
                                self.driver_assignment_counts[driver_name] += 1
            
            self.active_drivers = drivers_found
            
            print(f"✅ Found {len(drivers_found)} drivers from assignments")
            print(f"✅ Loaded {len(self.dog_assignments)} dog assignments")
            print(f"✅ Found {callouts_found} callouts needing assignment")
            print(f"✅ Capacity will be calculated dynamically based on route density:")
            print(f"   - Dense routes (avg < {self.DENSE_ROUTE_THRESHOLD} min): capacity 12")
            print(f"   - Standard routes: capacity 8")
            
        except Exception as e:
            print(f"❌ Error loading dog assignments: {e}")
            raise

    def calculate_driver_density(self, driver):
        """Calculate route density for a driver and return capacity info"""
        driver_dogs = []
        for assignment in self.dog_assignments:
            if isinstance(assignment, dict) and assignment.get('combined', '').startswith(f"{driver}:"):
                driver_dogs.append(assignment)
        
        if len(driver_dogs) < 2:
            return {
                'dog_count': len(driver_dogs),
                'avg_time': 0.0,
                'capacity': 8,
                'is_dense': False
            }
        
        # Calculate all pairwise times
        times = []
        for i in range(len(driver_dogs)):
            for j in range(i + 1, len(driver_dogs)):
                dog1_id = driver_dogs[i].get('dog_id', '')
                dog2_id = driver_dogs[j].get('dog_id', '')
                if dog1_id and dog2_id:
                    time_min = self.get_time_with_fallback(dog1_id, dog2_id)
                    if time_min < float('inf'):
                        times.append(time_min)
        
        if not times:
            avg_time = 0.0
        else:
            avg_time = sum(times) / len(times)
        
        # Dense route if average time is less than threshold
        # If no valid times, assume standard capacity
        is_dense = avg_time < self.DENSE_ROUTE_THRESHOLD and avg_time > 0
        capacity = 12 if is_dense else 8
        
        return {
            'dog_count': len(driver_dogs),
            'avg_time': avg_time,
            'capacity': capacity,
            'is_dense': is_dense
        }

    def parse_dog_groups_from_callout(self, callout):
        """Parse groups from callout for capacity checking"""
        if not callout:
            return []
        
        # Handle None and ensure string
        callout_str = str(callout) if callout else ""
        callout_clean = callout_str.lstrip(':')
        groups = []
        
        for char in callout_clean:
            if char in ['1', '2', '3']:
                group_num = int(char)
                if group_num not in groups:
                    groups.append(group_num)
        
        return sorted(groups)

    def phase1_cluster_existing_dogs(self):
        """NEW PHASE 1: Cluster nearby dogs BEFORE any new assignments
        
        This runs FIRST to identify natural clusters in existing assignments.
        Dogs < 1 minute apart should have the same driver.
        """
        print("\n🔗 PHASE 1: Clustering existing nearby dogs (< 1 minute apart)")
        print("=" * 60)
        
        moves_made = 0
        
        # Only process dogs that already have assignments (not callouts)
        assigned_dogs = []
        for assignment in self.dog_assignments:
            if isinstance(assignment, dict) and assignment.get('combined', '').strip():
                combined = assignment.get('combined', '')
                if ':' in combined:
                    driver = combined.split(':')[0]
                    if driver not in ['Field', 'Parking']:
                        assigned_dogs.append(assignment)
        
        print(f"📊 Analyzing {len(assigned_dogs)} assigned dogs for clusters")
        
        # Process each group separately
        for group_num in [1, 2, 3]:
            print(f"\n📊 Processing Group {group_num}:")
            
            # Get all dogs in this group
            group_dogs = []
            for assignment in assigned_dogs:
                combined = assignment.get('combined', '')
                if ':' in combined:
                    groups = self.parse_dog_groups_from_callout(combined.split(':', 1)[1])
                    if group_num in groups:
                        driver = combined.split(':')[0]
                        group_dogs.append({
                            'assignment': assignment,
                            'driver': driver,
                            'dog_id': assignment.get('dog_id', ''),
                            'dog_name': assignment.get('dog_name', 'Unknown')
                        })
            
            if len(group_dogs) < 2:
                print(f"   Only {len(group_dogs)} dogs in group - skipping")
                continue
            
            # Build adjacency list for dogs < 1 minute apart
            nearby_dogs = defaultdict(list)
            for i, dog1_data in enumerate(group_dogs):
                dog1_id = dog1_data['dog_id']
                
                for j, dog2_data in enumerate(group_dogs):
                    if i != j:
                        dog2_id = dog2_data['dog_id']
                        time_between = self.get_time_with_fallback(dog1_id, dog2_id)
                        
                        if time_between < self.CLUSTER_THRESHOLD:  # < 1 minute
                            nearby_dogs[dog1_id].append({
                                'dog_data': dog2_data,
                                'time': time_between
                            })
            
            # Find clusters using connected components
            visited = set()
            clusters = []
            
            for dog_data in group_dogs:
                dog_id = dog_data['dog_id']
                if dog_id not in visited:
                    # BFS to find all connected dogs
                    cluster = []
                    queue = [dog_data]
                    
                    while queue:
                        current = queue.pop(0)
                        current_id = current['dog_id']
                        
                        if current_id in visited:
                            continue
                            
                        visited.add(current_id)
                        cluster.append(current)
                        
                        # Add all nearby dogs to queue
                        for nearby in nearby_dogs.get(current_id, []):
                            if nearby['dog_data']['dog_id'] not in visited:
                                queue.append(nearby['dog_data'])
                    
                    if len(cluster) > 1:
                        clusters.append(cluster)
            
            # Process each cluster
            for cluster in clusters:
                drivers_in_cluster = set(dog['driver'] for dog in cluster)
                
                if len(drivers_in_cluster) > 1:
                    # Choose driver with most dogs in cluster AND capacity
                    driver_counts = Counter(dog['driver'] for dog in cluster)
                    
                    # Check capacity for each driver
                    valid_drivers = []
                    for driver, count in driver_counts.items():
                        capacity_info = self.calculate_driver_density(driver)
                        capacity = capacity_info['capacity']
                        current_in_group = sum(1 for d in group_dogs if d['driver'] == driver)
                        dogs_to_add = len(cluster) - count
                        
                        if current_in_group + dogs_to_add <= capacity:
                            valid_drivers.append((driver, count, capacity - current_in_group))
                    
                    if valid_drivers:
                        # Sort by: 1) most dogs already in cluster, 2) most capacity remaining
                        valid_drivers.sort(key=lambda x: (-x[1], -x[2]))
                        winning_driver = valid_drivers[0][0]
                        
                        print(f"\n   🔗 Found cluster of {len(cluster)} dogs:")
                        for dog in cluster:
                            print(f"      - {dog['dog_name']} (with {dog['driver']})")
                        print(f"      → Consolidating to {winning_driver}")
                        
                        # Move dogs to winning driver
                        for dog_data in cluster:
                            if dog_data['driver'] != winning_driver:
                                combined = dog_data['assignment'].get('combined', '')
                                if ':' in combined:
                                    original_groups = combined.split(':', 1)[1]
                                    dog_data['assignment']['combined'] = f"{winning_driver}:{original_groups}"
                                    moves_made += 1
        
        print(f"\n✅ Phase 1 Complete: {moves_made} dogs clustered")
        return moves_made

    def phase2_remove_outliers_from_existing(self):
        """NEW PHASE 2: Remove outliers from existing assignments only
        
        Runs AFTER clustering but BEFORE new callout assignments.
        Only processes dogs that already have drivers assigned.
        """
        print("\n🎯 PHASE 2: Removing outliers from existing assignments")
        print("=" * 60)
        
        moves_made = 0
        
        # Only process existing assignments, not callouts
        for driver in self.active_drivers:
            driver_groups = defaultdict(list)
            
            for assignment in self.dog_assignments:
                if isinstance(assignment, dict) and assignment.get('combined', '').startswith(f"{driver}:"):
                    combined = assignment['combined']
                    if ':' in combined:
                        groups = self.parse_dog_groups_from_callout(combined.split(':', 1)[1])
                        for group in groups:
                            driver_groups[group].append(assignment)
            
            # Analyze each group
            for group_num, dogs in driver_groups.items():
                if len(dogs) < 3:  # Need at least 3 dogs to identify outliers
                    continue
                
                # Calculate distances between all dogs
                all_distances = []
                for i in range(len(dogs)):
                    for j in range(i + 1, len(dogs)):
                        dog1_id = dogs[i].get('dog_id', '')
                        dog2_id = dogs[j].get('dog_id', '')
                        if dog1_id and dog2_id:
                            time_min = self.get_time_with_fallback(dog1_id, dog2_id)
                            if time_min < float('inf'):
                                all_distances.append(time_min)
                
                if not all_distances:
                    continue
                
                avg_distance = sum(all_distances) / len(all_distances)
                
                print(f"\n📊 {driver} Group {group_num}: {len(dogs)} dogs")
                print(f"   Average distance: {avg_distance:.1f} min")
                
                # Find outliers
                outliers = []
                for i, dog in enumerate(dogs):
                    dog_id = dog.get('dog_id', '')
                    if not dog_id:
                        continue
                    
                    # Calculate metrics for this dog
                    distances_to_others = []
                    min_distance = float('inf')
                    
                    for j, other_dog in enumerate(dogs):
                        if i != j:
                            other_id = other_dog.get('dog_id', '')
                            if other_id:
                                time_min = self.get_time_with_fallback(dog_id, other_id)
                                if time_min < float('inf'):
                                    distances_to_others.append(time_min)
                                    min_distance = min(min_distance, time_min)
                    
                    if distances_to_others:
                        avg_to_others = sum(distances_to_others) / len(distances_to_others)
                        
                        # Check outlier criteria
                        is_outlier = (
                            avg_to_others > avg_distance * 1.5 or  # > 1.5x average
                            avg_to_others > 3 or                    # > 3 min absolute
                            min_distance > 3                        # > 3 min to nearest
                        )
                        
                        if is_outlier:
                            outliers.append({
                                'dog': dog,
                                'avg_distance': avg_to_others,
                                'min_distance': min_distance,
                                'dog_name': dog.get('dog_name', 'Unknown')
                            })
                
                if outliers:
                    # Sort by how far they are from the group
                    outliers.sort(key=lambda x: -x['avg_distance'])
                    
                    print(f"   ⚠️  Found {len(outliers)} outlier(s):")
                    for outlier_data in outliers[:3]:  # Show top 3
                        print(f"      - {outlier_data['dog_name']}: avg {outlier_data['avg_distance']:.1f} min, "
                              f"nearest {outlier_data['min_distance']:.1f} min")
                    
                    for outlier_data in outliers:
                        dog = outlier_data['dog']
                        dog_id = dog.get('dog_id', '')
                        
                        # Find better placement
                        best_option = None
                        best_avg_distance = outlier_data['avg_distance']
                        
                        for other_driver in self.active_drivers:
                            if other_driver == driver:
                                continue
                            
                            # Get dogs in same group for other driver
                            other_dogs = []
                            for assignment in self.dog_assignments:
                                if isinstance(assignment, dict) and assignment.get('combined', '').startswith(f"{other_driver}:"):
                                    combined = assignment.get('combined', '')
                                    if ':' in combined:
                                        groups = self.parse_dog_groups_from_callout(combined.split(':', 1)[1])
                                        if group_num in groups:
                                            other_dogs.append(assignment)
                            
                            if not other_dogs:
                                continue
                            
                            # Calculate average distance
                            distances = []
                            for other_dog in other_dogs:
                                other_id = other_dog.get('dog_id', '')
                                if other_id and dog_id:
                                    time_min = self.get_time_with_fallback(dog_id, other_id)
                                    if time_min < float('inf'):
                                        distances.append(time_min)
                            
                            if distances:
                                avg_dist = sum(distances) / len(distances)
                                if avg_dist < best_avg_distance:
                                    best_avg_distance = avg_dist
                                    best_option = other_driver
                        
                        # Move if better location found
                        if best_option:
                            combined = dog.get('combined', '')
                            if ':' in combined:
                                original_groups = combined.split(':', 1)[1]
                                dog['combined'] = f"{best_option}:{original_groups}"
                                print(f"   ✅ Moved outlier {outlier_data['dog_name']} to {best_option} "
                                      f"(new avg: {best_avg_distance:.1f} min)")
                                moves_made += 1
        
        print(f"\n✅ Phase 2 Complete: {moves_made} outliers moved")
        return moves_made

    def phase3_assign_callouts_with_capacity(self):
        """NEW PHASE 3: Assign callouts intelligently with capacity management
        
        - Find closest neighbor dog (not driver)
        - If multiple neighbors within 2 minutes, choose based on driver availability
        - Handle over-capacity situations with cascading logic
        """
        print("\n🎯 PHASE 3: Assigning callouts with intelligent capacity management")
        print("=" * 60)
        
        # Find all callouts
        callouts = []
        for assignment in self.dog_assignments:
            if isinstance(assignment, dict):
                callout = assignment.get('callout', '').strip()
                combined = assignment.get('combined', '').strip()
                if callout and not combined:
                    callouts.append(assignment)
        
        print(f"📊 Found {len(callouts)} callouts to assign")
        
        assignments_made = 0
        
        for callout in callouts:
            dog_name = callout.get('dog_name', 'Unknown')
            dog_id = callout.get('dog_id', '')
            original_callout = callout.get('callout', '').strip()
            
            # Parse which groups this dog needs
            callout_groups = self.parse_dog_groups_from_callout(original_callout)
            
            if not callout_groups:
                print(f"❌ {dog_name}: Invalid callout format '{original_callout}'")
                continue
            
            # Find all potential neighbor dogs
            neighbors = []
            
            for assignment in self.dog_assignments:
                if isinstance(assignment, dict) and assignment.get('combined', '').strip():
                    if ':' in assignment['combined']:
                        neighbor_driver = assignment['combined'].split(':')[0]
                        if neighbor_driver in ['Field', 'Parking']:
                            continue
                        
                        neighbor_groups = self.parse_dog_groups_from_callout(
                            assignment['combined'].split(':', 1)[1]
                        )
                        
                        # Check if neighbor has any matching groups
                        matching_groups = set(callout_groups) & set(neighbor_groups)
                        if matching_groups:
                            neighbor_id = assignment.get('dog_id', '')
                            if neighbor_id and dog_id:
                                time_to_neighbor = self.get_time_with_fallback(dog_id, neighbor_id)
                                if time_to_neighbor < float('inf'):
                                    neighbors.append({
                                        'driver': neighbor_driver,
                                        'time': time_to_neighbor,
                                        'dog_name': assignment.get('dog_name', 'Unknown'),
                                        'matching_groups': matching_groups
                                    })
            
            if not neighbors:
                print(f"❌ {dog_name}: No valid neighbors found")
                continue
            
            # Sort neighbors by distance
            neighbors.sort(key=lambda x: x['time'])
            
            # Find neighbors within 2 minutes
            close_neighbors = [n for n in neighbors if n['time'] <= 2]
            
            best_driver = None
            chosen_neighbor = None
            
            if close_neighbors:
                # Multiple close neighbors - check driver availability
                driver_scores = defaultdict(lambda: {'min_time': float('inf'), 'capacity_score': 0})
                
                for neighbor in close_neighbors:
                    driver = neighbor['driver']
                    
                    # Calculate capacity for each matching group
                    capacity_info = self.calculate_driver_density(driver)
                    capacity = capacity_info['capacity']
                    
                    for group_num in neighbor['matching_groups']:
                        current_count = sum(1 for a in self.dog_assignments
                                          if isinstance(a, dict) and 
                                          a.get('combined', '').startswith(f"{driver}:") and
                                          group_num in self.parse_dog_groups_from_callout(
                                              a.get('combined', '').split(':', 1)[1] if ':' in a.get('combined', '') else ''))
                        
                        available_capacity = max(0, capacity - current_count)
                        
                        # Update driver score
                        if neighbor['time'] < driver_scores[driver]['min_time']:
                            driver_scores[driver]['min_time'] = neighbor['time']
                        driver_scores[driver]['capacity_score'] = max(
                            driver_scores[driver]['capacity_score'], 
                            available_capacity
                        )
                
                # Choose best driver (weighted by distance and availability)
                best_score = float('-inf')
                
                for driver, scores in driver_scores.items():
                    # Weight: closer is better, more capacity is better
                    weighted_score = (2 - scores['min_time']) * 2 + scores['capacity_score']
                    if weighted_score > best_score:
                        best_score = weighted_score
                        best_driver = driver
                
                if best_driver:
                    chosen_neighbor = next(n for n in close_neighbors if n['driver'] == best_driver)
            else:
                # No close neighbors - just use closest
                chosen_neighbor = neighbors[0]
                best_driver = chosen_neighbor['driver']
            
            # Assign to chosen driver
            if best_driver:
                if original_callout.startswith(':'):
                    group_part = original_callout
                else:
                    group_part = ':' + original_callout
                
                callout['combined'] = f"{best_driver}{group_part}"
                assignments_made += 1
                
                print(f"   ✅ {dog_name} → {best_driver} (closest to {chosen_neighbor['dog_name']}, "
                      f"{chosen_neighbor['time']:.1f} min)")
            else:
                print(f"   ❌ {dog_name}: Could not find suitable driver")
        
        # Now handle over-capacity situations
        print(f"\n🔄 Checking for over-capacity groups...")
        moves_made = self._handle_over_capacity_cascading()
        
        print(f"\n✅ Phase 3 Complete: {assignments_made} callouts assigned, "
              f"{moves_made} capacity adjustments")
        return assignments_made + moves_made

    def _handle_over_capacity_cascading(self):
        """Handle over-capacity groups with cascading logic"""
        moves_made = 0
        max_iterations = 20
        iteration = 0
        moved_dogs = set()  # Track moved dogs to prevent infinite loops
        
        while iteration < max_iterations:
            iteration += 1
            over_capacity_found = False
            moves_this_iteration = 0
            max_moves_per_iteration = 5  # Prevent runaway cascading
            
            # Find all over-capacity groups
            for driver in self.active_drivers:
                if moves_this_iteration >= max_moves_per_iteration:
                    break
                    
                capacity_info = self.calculate_driver_density(driver)
                capacity = capacity_info['capacity']
                
                driver_groups = defaultdict(list)
                for assignment in self.dog_assignments:
                    if isinstance(assignment, dict) and assignment.get('combined', '').startswith(f"{driver}:"):
                        combined = assignment.get('combined', '')
                        if ':' in combined:
                            groups = self.parse_dog_groups_from_callout(combined.split(':', 1)[1])
                            for group in groups:
                                driver_groups[group].append(assignment)
                
                # Check each group
                for group_num, dogs in driver_groups.items():
                    if moves_this_iteration >= max_moves_per_iteration:
                        break
                        
                    if len(dogs) > capacity:
                        over_capacity_found = True
                        print(f"\n   ⚠️  {driver} Group {group_num}: {len(dogs)}/{capacity} dogs (over by {len(dogs) - capacity})")
                        
                        # Find outliers in this group
                        outliers = self._find_outliers_in_group(dogs)
                        
                        if outliers:
                            # Try to move outliers starting with the worst
                            moved_this_group = False
                            for outlier in outliers:
                                if moved_this_group:
                                    break
                                    
                                dog = outlier['dog']
                                dog_id = dog.get('dog_id', '')
                                dog_name = dog.get('dog_name', 'Unknown')
                                
                                # Skip if we've already moved this dog
                                if dog_id in moved_dogs:
                                    continue
                            
                            # Find alternative placements within 5 minutes
                            alternatives = []
                            
                            for other_driver in self.active_drivers:
                                if other_driver == driver:
                                    continue
                                
                                # Check if other driver has this group and capacity
                                other_capacity_info = self.calculate_driver_density(other_driver)
                                other_capacity = other_capacity_info['capacity']
                                
                                other_group_count = sum(1 for a in self.dog_assignments
                                                      if isinstance(a, dict) and 
                                                      a.get('combined', '').startswith(f"{other_driver}:") and
                                                      group_num in self.parse_dog_groups_from_callout(
                                                          a.get('combined', '').split(':', 1)[1] if ':' in a.get('combined', '') else ''))
                                
                                if other_group_count < other_capacity:
                                    # Double-check this won't immediately create another over-capacity
                                    if other_group_count + 1 <= other_capacity:
                                        # Find closest dog in other driver's group
                                    min_time = float('inf')
                                    for a in self.dog_assignments:
                                        if (isinstance(a, dict) and 
                                            a.get('combined', '').startswith(f"{other_driver}:") and
                                            group_num in self.parse_dog_groups_from_callout(
                                                a['combined'].split(':', 1)[1])):
                                            other_id = a.get('dog_id', '')
                                            if other_id and dog_id:
                                                time = self.get_time_with_fallback(dog_id, other_id)
                                                min_time = min(min_time, time)
                                    
                                    if min_time <= 5:  # Within 5 minutes
                                        alternatives.append({
                                            'driver': other_driver,
                                            'time': min_time,
                                            'available_capacity': other_capacity - other_group_count
                                        })
                            
                                if alternatives:
                                    # Sort by weighted score (distance + availability)
                                    alternatives.sort(key=lambda x: x['time'] - x['available_capacity'] * 0.5)
                                    best_alt = alternatives[0]
                                    
                                    # Move the dog
                                    combined = dog.get('combined', '')
                                    if ':' in combined:
                                        original_groups = combined.split(':', 1)[1]
                                        dog['combined'] = f"{best_alt['driver']}:{original_groups}"
                                        moves_made += 1
                                        moved_dogs.add(dog_id)  # Track this move
                                        moved_this_group = True
                                        moves_this_iteration += 1
                                        
                                        print(f"      ✅ Moved outlier {dog_name} to {best_alt['driver']} "
                                              f"({best_alt['time']:.1f} min)")
                                        
                                        if moves_this_iteration >= max_moves_per_iteration:
                                            print(f"   ⚠️  Reached max moves per iteration ({max_moves_per_iteration})")
                                            break
        
        if not over_capacity_found:
            print("   ✅ All groups within capacity")
        
        return moves_made

    def _find_outliers_in_group(self, dogs):
        """Find outliers in a group of dogs"""
        if len(dogs) < 2:
            return []
        
        # Calculate average distance between all dogs
        all_times = []
        for i in range(len(dogs)):
            for j in range(i + 1, len(dogs)):
                dog1_id = dogs[i].get('dog_id', '')
                dog2_id = dogs[j].get('dog_id', '')
                if dog1_id and dog2_id:
                    time = self.get_time_with_fallback(dog1_id, dog2_id)
                    if time < float('inf'):
                        all_times.append(time)
        
        if not all_times:
            return []
        
        avg_time = sum(all_times) / len(all_times)
        
        # Find dogs that are outliers
        outliers = []
        for dog in dogs:
            dog_id = dog.get('dog_id', '')
            if not dog_id:
                continue
            
            # Ensure dog has a combined field
            if not dog.get('combined'):
                continue
            
            # Calculate this dog's average distance to others
            times_to_others = []
            for other_dog in dogs:
                if other_dog != dog:
                    other_id = other_dog.get('dog_id', '')
                    if other_id:
                        time = self.get_time_with_fallback(dog_id, other_id)
                        if time < float('inf'):
                            times_to_others.append(time)
            
            if times_to_others:
                avg_to_others = sum(times_to_others) / len(times_to_others)
                min_to_others = min(times_to_others)
                
                # Check if outlier
                if avg_to_others > avg_time * 1.5 or avg_to_others > 3 or min_to_others > 3:
                    outliers.append({
                        'dog': dog,
                        'avg_distance': avg_to_others,
                        'min_distance': min_to_others
                    })
        
        # Sort by average distance (worst outliers first)
        outliers.sort(key=lambda x: -x['avg_distance'])
        return outliers

    def phase4_consolidate_small_drivers(self):
        """NEW PHASE 4: Consolidate drivers with < 7 total dogs
        
        Reduced threshold from 12 to 7 dogs.
        """
        print("\n🔄 PHASE 4: Consolidating small drivers (< 7 dogs)")
        print("=" * 60)
        
        # Count dogs per driver
        driver_dog_counts = defaultdict(list)
        for assignment in self.dog_assignments:
            if isinstance(assignment, dict) and assignment.get('combined', '').strip():
                combined = assignment['combined']
                if ':' in combined:
                    driver = combined.split(':')[0]
                    if driver not in ['Field', 'Parking']:
                        driver_dog_counts[driver].append(assignment)
        
        # Find drivers to consolidate
        drivers_to_consolidate = []
        for driver, dogs in driver_dog_counts.items():
            if len(dogs) < self.MIN_DOGS_FOR_DRIVER:
                drivers_to_consolidate.append((driver, dogs))
        
        print(f"🔍 Found {len(drivers_to_consolidate)} drivers with < {self.MIN_DOGS_FOR_DRIVER} dogs")
        
        if not drivers_to_consolidate:
            print("✅ No small drivers to consolidate")
            return 0
        
        dogs_moved = 0
        
        for driver_to_remove, dogs_to_move in drivers_to_consolidate:
            print(f"\n📤 Consolidating {driver_to_remove} ({len(dogs_to_move)} dogs):")
            
            for dog in dogs_to_move:
                dog_name = dog.get('dog_name', 'Unknown')
                dog_id = dog.get('dog_id', '')
                combined = dog.get('combined', '')
                
                if not combined or ':' not in combined:
                    print(f"   ❌ {dog_name}: Invalid assignment format")
                    continue
                    
                original_groups = combined.split(':', 1)[1]
                
                # Find best alternative driver
                best_driver = None
                best_time = float('inf')
                
                for other_driver in list(self.active_drivers):  # Use list() to avoid modification issues
                    if other_driver == driver_to_remove:
                        continue
                    
                    # Calculate average time to other driver's dogs
                    other_driver_dogs = driver_dog_counts.get(other_driver, [])
                    if not other_driver_dogs:
                        continue
                    
                    times = []
                    for other_dog in other_driver_dogs:
                        other_dog_id = other_dog.get('dog_id', '')
                        if other_dog_id and dog_id:
                            time_min = self.get_time_with_fallback(dog_id, other_dog_id)
                            if time_min < float('inf'):
                                times.append(time_min)
                    
                    if times:
                        avg_time = sum(times) / len(times)
                        if avg_time < best_time:
                            best_time = avg_time
                            best_driver = other_driver
                
                if best_driver:
                    dog['combined'] = f"{best_driver}:{original_groups}"
                    dogs_moved += 1
                    print(f"   ✅ {dog_name} → {best_driver} ({best_time:.1f} min avg)")
                else:
                    print(f"   ❌ {dog_name}: No suitable driver found")
            
            # Remove driver from active list
            self.active_drivers.discard(driver_to_remove)
            print(f"   🏠 {driver_to_remove} can take the day off!")
        
        print(f"\n✅ Phase 4 Complete: {dogs_moved} dogs moved, "
              f"{len(drivers_to_consolidate)} drivers eliminated")
        return dogs_moved

    def phase5_consolidate_small_groups_constrained(self):
        """NEW PHASE 5: Consolidate small groups with constraints
        
        - Only move if total drive time increase < 10 minutes
        - Choose neighbor with most space when there's a tie
        """
        print("\n🔄 PHASE 5: Consolidating small groups (< 4 dogs) with constraints")
        print("=" * 60)
        
        moves_made = 0
        
        for driver in list(self.active_drivers):
            driver_groups = defaultdict(list)
            
            for assignment in self.dog_assignments:
                if isinstance(assignment, dict) and assignment.get('combined', '').startswith(f"{driver}:"):
                    combined = assignment.get('combined', '')
                    if ':' in combined:
                        groups = self.parse_dog_groups_from_callout(combined.split(':', 1)[1])
                        for group in groups:
                            driver_groups[group].append(assignment)
            
            # Check which groups exist
            has_group1 = 1 in driver_groups and len(driver_groups[1]) > 0
            has_group2 = 2 in driver_groups and len(driver_groups[2]) > 0
            has_group3 = 3 in driver_groups and len(driver_groups[3]) > 0
            
            # Only consolidate if driver has all 3 groups
            if not (has_group1 and has_group2 and has_group3):
                continue
            
            # Check Group 1 and Group 3 for small size
            for group_num in [1, 3]:
                if group_num not in driver_groups:
                    continue
                    
                if len(driver_groups[group_num]) >= self.MIN_GROUP_SIZE:
                    continue
                    
                dogs_to_move = driver_groups[group_num]
                print(f"\n📊 {driver} Group {group_num} has only {len(dogs_to_move)} dogs")
                
                # Calculate current total drive time for this group
                current_total_time = 0
                for i in range(len(dogs_to_move)):
                    for j in range(i + 1, len(dogs_to_move)):
                        dog1_id = dogs_to_move[i].get('dog_id', '')
                        dog2_id = dogs_to_move[j].get('dog_id', '')
                        if dog1_id and dog2_id:
                            time = self.get_time_with_fallback(dog1_id, dog2_id)
                            if time < float('inf'):
                                current_total_time += time
                
                # Try to move each dog
                group_moves = 0
                for dog in dogs_to_move:
                    dog_name = dog.get('dog_name', 'Unknown')
                    dog_id = dog.get('dog_id', '')
                    
                    if not dog.get('combined') or ':' not in dog.get('combined', ''):
                        print(f"   ❌ {dog_name}: Invalid assignment format")
                        continue
                    
                    # Find all potential destinations
                    options = []
                    
                    for other_driver in self.active_drivers:
                        if other_driver == driver:
                            continue
                        
                        try:
                            # Get capacity info
                            capacity_info = self.calculate_driver_density(other_driver)
                            capacity = capacity_info['capacity']
                            
                            # Count dogs in this group for other driver
                            other_group_dogs = []
                            if self.dog_assignments:  # Check if there are assignments
                                for assignment in self.dog_assignments:
                                    if (isinstance(assignment, dict) and 
                                        assignment.get('combined', '').startswith(f"{other_driver}:") and
                                        group_num in self.parse_dog_groups_from_callout(
                                            assignment.get('combined', '').split(':', 1)[1] if ':' in assignment.get('combined', '') else '')):
                                        other_group_dogs.append(assignment)
                            
                            available_capacity = capacity - len(other_group_dogs)
                            
                            # Only process if there's capacity
                            if available_capacity > 0:
                                # Find closest dog in other driver's group
                                min_time = float('inf')
                                closest_dog_name = None
                                
                                if other_group_dogs:  # Only if there are dogs to compare to
                                    for other_dog in other_group_dogs:
                                        other_id = other_dog.get('dog_id', '')
                                        if other_id and dog_id:
                                            time = self.get_time_with_fallback(dog_id, other_id)
                                            if time < min_time:
                                                min_time = time
                                                closest_dog_name = other_dog.get('dog_name', 'Unknown')
                                else:
                                    # Empty group - use a default time
                                    min_time = 5.0  # Assume 5 minutes for empty groups
                                    closest_dog_name = "Empty group"
                                
                                if min_time < float('inf'):
                                    options.append({
                                        'driver': other_driver,
                                        'time': min_time,
                                        'capacity': available_capacity,
                                        'closest_dog': closest_dog_name
                                    })
                        except Exception as e:
                            print(f"   ⚠️  Error processing {other_driver}: {e}")
                            continue
                    
                    # Sort options by time, but consider ties
                    if options:
                        options.sort(key=lambda x: x['time'])
                        
                        # Find all options within 1 minute of best
                        best_time = options[0]['time']
                        tied_options = [opt for opt in options if opt['time'] <= best_time + 1]
                        
                        # Among tied options, choose one with most capacity
                        if tied_options:
                            tied_options.sort(key=lambda x: -x['capacity'])
                            best_option = tied_options[0]
                            
                            # Calculate time increase more accurately
                            # Current setup: all dogs in small group travel together
                            # New setup: this dog travels with new group
                            # Increase is roughly the distance to new group
                            time_increase = best_option['time']
                            
                            if time_increase < self.GROUP_CONSOLIDATION_TIME_LIMIT:
                                combined = dog.get('combined', '')
                                if ':' in combined:
                                    original_groups = combined.split(':', 1)[1]
                                    dog['combined'] = f"{best_option['driver']}:{original_groups}"
                                    print(f"   ✅ {dog_name} → {best_option['driver']} "
                                          f"(closest to {best_option['closest_dog']}, "
                                          f"{best_option['time']:.1f} min, "
                                          f"+{time_increase:.1f} min total)")
                                    moves_made += 1
                                    group_moves += 1
                            else:
                                print(f"   ❌ {dog_name}: Would increase time by {time_increase:.1f} min "
                                      f"(limit: {self.GROUP_CONSOLIDATION_TIME_LIMIT} min)")
                        else:
                            print(f"   ❌ {dog_name}: No options within tie threshold")
                    else:
                        print(f"   ❌ {dog_name}: No suitable destination found")
                
                if group_moves == 0:
                    print(f"   ℹ️  Could not consolidate any dogs from this small group")
        
        print(f"\n✅ Phase 5 Complete: {moves_made} dogs moved from small groups")
        return moves_made
        """NEW PHASE 5: Consolidate small groups with constraints
        
        - Only move if total drive time increase < 10 minutes
        - Choose neighbor with most space when there's a tie
        """
        print("\n🔄 PHASE 5: Consolidating small groups (< 4 dogs) with constraints")
        print("=" * 60)
        
        moves_made = 0
        
        for driver in list(self.active_drivers):
            driver_groups = defaultdict(list)
            
            for assignment in self.dog_assignments:
                if isinstance(assignment, dict) and assignment.get('combined', '').startswith(f"{driver}:"):
                    combined = assignment.get('combined', '')
                    if ':' in combined:
                        groups = self.parse_dog_groups_from_callout(combined.split(':', 1)[1])
                        for group in groups:
                            driver_groups[group].append(assignment)
            
            # Check which groups exist
            has_group1 = 1 in driver_groups and len(driver_groups[1]) > 0
            has_group2 = 2 in driver_groups and len(driver_groups[2]) > 0
            has_group3 = 3 in driver_groups and len(driver_groups[3]) > 0
            
            # Only consolidate if driver has all 3 groups
            if not (has_group1 and has_group2 and has_group3):
                continue
            
            # Check Group 1 and Group 3 for small size
            for group_num in [1, 3]:
                if group_num not in driver_groups:
                    continue
                    
                if len(driver_groups[group_num]) < self.MIN_GROUP_SIZE:
                    dogs_to_move = driver_groups[group_num]
                    print(f"\n📊 {driver} Group {group_num} has only {len(dogs_to_move)} dogs")
                    
                    # Calculate current total drive time for this group
                    current_total_time = 0
                    for i in range(len(dogs_to_move)):
                        for j in range(i + 1, len(dogs_to_move)):
                            dog1_id = dogs_to_move[i].get('dog_id', '')
                            dog2_id = dogs_to_move[j].get('dog_id', '')
                            if dog1_id and dog2_id:
                                time = self.get_time_with_fallback(dog1_id, dog2_id)
                                if time < float('inf'):
                                    current_total_time += time
                    
                    # Try to move each dog
                    for dog in dogs_to_move:
                        dog_name = dog.get('dog_name', 'Unknown')
                        dog_id = dog.get('dog_id', '')
                        
                        if not dog.get('combined') or ':' not in dog.get('combined', ''):
                            print(f"   ❌ {dog_name}: Invalid assignment format")
                            continue
                        
                        # Find all potential destinations
                        options = []
                        
                        for other_driver in self.active_drivers:
                            if other_driver == driver:
                                continue
                            
                            try:
                                # Get capacity info
                                capacity_info = self.calculate_driver_density(other_driver)
                                capacity = capacity_info.get('capacity', 8)
                                
                                # Count dogs in this group for other driver
                                other_group_dogs = []
                                for assignment in self.dog_assignments:
                                    if (isinstance(assignment, dict) and 
                                        assignment.get('combined', '').startswith(f"{other_driver}:") and
                                        ':' in assignment.get('combined', '')):
                                        assignment_groups = self.parse_dog_groups_from_callout(
                                            assignment.get('combined', '').split(':', 1)[1])
                                        if group_num in assignment_groups:
                                            other_group_dogs.append(assignment)
                                
                                available_capacity = capacity - len(other_group_dogs)
                                
                                # Only process if there's capacity
                                if available_capacity > 0:
                                    # Find closest dog in other driver's group
                                    min_time = float('inf')
                                    closest_dog_name = None
                                    
                                    if len(other_group_dogs) > 0:
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
                            except Exception as e:
                                print(f"   ⚠️  Error processing {other_driver}: {e}")
                                continue
                        
            # Find all options within 1 minute of best
            if options:
                best_time = options[0]['time']
                tied_options = [opt for opt in options if opt['time'] <= best_time + 1]
                
                # Among tied options, choose one with most capacity
                if tied_options:
                    tied_options.sort(key=lambda x: -x['capacity'])
                    best_option = tied_options[0]
                    
                    # Calculate time increase more accurately
                    # Current setup: all dogs in small group travel together
                    # New setup: this dog travels with new group
                    # Increase is roughly the distance to new group
                    time_increase = best_option['time']
                    
                    if time_increase < self.GROUP_CONSOLIDATION_TIME_LIMIT:
                        combined = dog.get('combined', '')
                        if ':' in combined:
                            original_groups = combined.split(':', 1)[1]
                            dog['combined'] = f"{best_option['driver']}:{original_groups}"
                            print(f"   ✅ {dog_name} → {best_option['driver']} "
                                  f"(closest to {best_option['closest_dog']}, "
                                  f"{best_option['time']:.1f} min, "
                                  f"+{time_increase:.1f} min total)")
                            moves_made += 1
                    else:
                        print(f"   ❌ {dog_name}: Would increase time by {time_increase:.1f} min "
                              f"(limit: {self.GROUP_CONSOLIDATION_TIME_LIMIT} min)")
        
        print(f"\n✅ Phase 5 Complete: {moves_made} dogs moved from small groups")
        return moves_made

    def phase6_final_outlier_sweep(self):
        """NEW PHASE 6: Final sweep to clean up any remaining outliers
        
        Looks across ALL groups (not just over-capacity) to find outliers and move them to:
        1. Groups with extra capacity that can easily absorb them
        2. Drivers with neighboring dogs (better geographic fit)
        
        This is a final cleanup phase to catch any suboptimal assignments.
        """
        print("\n🧹 PHASE 6: Final outlier sweep across all groups")
        print("=" * 60)
        
        moves_made = 0
        all_outliers = []
        
        # Find all outliers across all drivers and groups
        for driver in self.active_drivers:
            driver_groups = defaultdict(list)
            
            for assignment in self.dog_assignments:
                if isinstance(assignment, dict) and assignment.get('combined', '').startswith(f"{driver}:"):
                    combined = assignment.get('combined', '')
                    if ':' in combined:
                        groups = self.parse_dog_groups_from_callout(combined.split(':', 1)[1])
                        for group in groups:
                            driver_groups[group].append(assignment)
            
            # Check each group for outliers
            for group_num, dogs in driver_groups.items():
                if len(dogs) < 3:  # Need at least 3 dogs to identify outliers
                    continue
                
                # Calculate average distance in group
                all_distances = []
                for i in range(len(dogs)):
                    for j in range(i + 1, len(dogs)):
                        dog1_id = dogs[i].get('dog_id', '')
                        dog2_id = dogs[j].get('dog_id', '')
                        if dog1_id and dog2_id:
                            time_min = self.get_time_with_fallback(dog1_id, dog2_id)
                            if time_min < float('inf'):
                                all_distances.append(time_min)
                
                if not all_distances:
                    continue
                
                avg_distance = sum(all_distances) / len(all_distances)
                
                # Find outliers in this group
                for dog in dogs:
                    dog_id = dog.get('dog_id', '')
                    if not dog_id:
                        continue
                    
                    # Ensure dog has a combined field
                    if not dog.get('combined') or ':' not in dog.get('combined', ''):
                        continue
                    
                    # Calculate metrics
                    distances_to_others = []
                    min_distance = float('inf')
                    
                    for other_dog in dogs:
                        if other_dog != dog:
                            other_id = other_dog.get('dog_id', '')
                            if other_id:
                                time_min = self.get_time_with_fallback(dog_id, other_id)
                                if time_min < float('inf'):
                                    distances_to_others.append(time_min)
                                    min_distance = min(min_distance, time_min)
                    
                    if distances_to_others:
                        avg_to_others = sum(distances_to_others) / len(distances_to_others)
                        
                        # More conservative criteria for final sweep
                        is_outlier = (
                            avg_to_others > avg_distance * 2 or  # > 2x average (more conservative)
                            avg_to_others > 4 or                  # > 4 min absolute
                            min_distance > 4                      # > 4 min to nearest
                        )
                        
                        if is_outlier:
                            all_outliers.append({
                                'dog': dog,
                                'current_driver': driver,
                                'current_group': group_num,
                                'avg_distance': avg_to_others,
                                'min_distance': min_distance,
                                'dog_name': dog.get('dog_name', 'Unknown'),
                                'dog_id': dog_id,
                                'group_avg': avg_distance,
                                'severity': avg_to_others / avg_distance  # How bad is this outlier
                            })
        
        if not all_outliers:
            print("✅ No significant outliers found - assignments look good!")
            return 0
        
        # Sort outliers by severity (worst first)
        all_outliers.sort(key=lambda x: -x['severity'])
        
        print(f"🔍 Found {len(all_outliers)} outliers across all groups")
        print("\nTop outliers:")
        for outlier in all_outliers[:5]:
            print(f"   - {outlier['dog_name']} ({outlier['current_driver']} G{outlier['current_group']}): "
                  f"avg {outlier['avg_distance']:.1f} min, nearest {outlier['min_distance']:.1f} min "
                  f"(severity: {outlier['severity']:.1f}x)")
        
        # Try to relocate outliers
        for outlier_data in all_outliers:
            dog = outlier_data['dog']
            dog_id = outlier_data['dog_id']
            current_driver = outlier_data['current_driver']
            group_num = outlier_data['current_group']
            
            # Find all possible destinations
            destinations = []
            
            for driver in self.active_drivers:
                if driver == current_driver:
                    continue
                
                # Get capacity info
                capacity_info = self.calculate_driver_density(driver)
                capacity = capacity_info['capacity']
                
                # Count current dogs in this group
                current_count = sum(1 for a in self.dog_assignments
                                  if isinstance(a, dict) and 
                                  a.get('combined', '').startswith(f"{driver}:") and
                                  group_num in self.parse_dog_groups_from_callout(
                                      a.get('combined', '').split(':', 1)[1] if ':' in a.get('combined', '') else ''))
                
                available_capacity = capacity - current_count
                
                # Get dogs in same group for this driver
                driver_group_dogs = []
                for assignment in self.dog_assignments:
                    if (isinstance(assignment, dict) and 
                        assignment.get('combined', '').startswith(f"{driver}:") and
                        group_num in self.parse_dog_groups_from_callout(
                            assignment.get('combined', '').split(':', 1)[1] if ':' in assignment.get('combined', '') else '')):
                        driver_group_dogs.append(assignment)
                
                if not driver_group_dogs and available_capacity <= 0:
                    continue  # Skip if no dogs and no capacity
                
                # Calculate fit with this group
                if driver_group_dogs:
                    distances = []
                    min_dist = float('inf')
                    for other_dog in driver_group_dogs:
                        other_id = other_dog.get('dog_id', '')
                        if other_id and dog_id:
                            time_min = self.get_time_with_fallback(dog_id, other_id)
                            if time_min < float('inf'):
                                distances.append(time_min)
                                min_dist = min(min_dist, time_min)
                    
                    if distances:
                        avg_dist = sum(distances) / len(distances)
                        
                        # Calculate improvement score
                        improvement = outlier_data['avg_distance'] - avg_dist
                        
                        destinations.append({
                            'driver': driver,
                            'avg_distance': avg_dist,
                            'min_distance': min_dist,
                            'available_capacity': available_capacity,
                            'improvement': improvement,
                            'has_neighbors': True,
                            'neighbor_count': len(driver_group_dogs)
                        })
                elif available_capacity > 2:  # Empty group with good capacity
                    # Check if there are dogs nearby in other groups
                    nearby_dogs = 0
                    min_cross_group_dist = float('inf')
                    
                    for g in [1, 2, 3]:
                        if g == group_num:
                            continue
                        for assignment in self.dog_assignments:
                            if (isinstance(assignment, dict) and 
                                assignment.get('combined', '').startswith(f"{driver}:") and
                                g in self.parse_dog_groups_from_callout(
                                    assignment.get('combined', '').split(':', 1)[1] if ':' in assignment.get('combined', '') else '')):
                                other_id = assignment.get('dog_id', '')
                                if other_id and dog_id:
                                    time_min = self.get_time_with_fallback(dog_id, other_id)
                                    if time_min < 3:  # Within 3 minutes
                                        nearby_dogs += 1
                                    min_cross_group_dist = min(min_cross_group_dist, time_min)
                    
                    if nearby_dogs > 0:
                        destinations.append({
                            'driver': driver,
                            'avg_distance': min_cross_group_dist,
                            'min_distance': min_cross_group_dist,
                            'available_capacity': available_capacity,
                            'improvement': outlier_data['avg_distance'] - min_cross_group_dist,
                            'has_neighbors': False,
                            'neighbor_count': 0,
                            'nearby_in_other_groups': nearby_dogs
                        })
            
            # Sort destinations by criteria
            if destinations:
                # Sort by: 1) significant improvement, 2) has neighbors, 3) capacity
                destinations.sort(key=lambda x: (
                    -max(x['improvement'], 0),  # Improvement (positive is better)
                    -x['has_neighbors'],         # Prefer groups with existing dogs
                    -x['available_capacity']     # Prefer groups with more space
                ))
                
                # Only move if there's significant improvement
                best_dest = destinations[0]
                if best_dest['improvement'] > 1 and best_dest['avg_distance'] < outlier_data['avg_distance'] - 1:
                    # Make the move
                    combined = dog.get('combined', '')
                    if ':' in combined:
                        original_groups = combined.split(':', 1)[1]
                        dog['combined'] = f"{best_dest['driver']}:{original_groups}"
                        
                        improvement_pct = (best_dest['improvement'] / outlier_data['avg_distance']) * 100 if outlier_data['avg_distance'] > 0 else 0
                        
                        print(f"\n✅ Moved outlier {outlier_data['dog_name']}:")
                        print(f"   From: {current_driver} (avg {outlier_data['avg_distance']:.1f} min)")
                        print(f"   To: {best_dest['driver']} (avg {best_dest['avg_distance']:.1f} min)")
                        print(f"   Improvement: {best_dest['improvement']:.1f} min ({improvement_pct:.0f}% better)")
                        if not best_dest['has_neighbors']:
                            print(f"   Note: First dog in this group, but has {best_dest.get('nearby_in_other_groups', 0)} "
                                  f"dogs nearby in other groups")
                        
                        moves_made += 1
                        
                        # Stop after moving 10 outliers to avoid over-optimization
                        if moves_made >= 10:
                            print(f"\n📊 Moved {moves_made} outliers - stopping to avoid over-optimization")
                            break
        
        if moves_made == 0 and all_outliers:
            print("\n📊 Found outliers but no significantly better placements available")
            print("   (Would need to compromise other objectives to improve further)")
        
        print(f"\n✅ Phase 6 Complete: {moves_made} outliers relocated")
        return moves_made
        """Main optimization function with revised phase order"""
        print("\n🚀 STARTING REVISED OPTIMIZATION STRATEGY (5 PHASES)")
        print("=" * 60)
        
        # Phase 1: Cluster existing dogs FIRST
        cluster_moves = self.phase1_cluster_existing_dogs()
        
        # Phase 2: Remove outliers from existing assignments
        outlier_moves = self.phase2_remove_outliers_from_existing()
        
        # Phase 3: Assign callouts with intelligent capacity management
        callout_moves = self.phase3_assign_callouts_with_capacity()
        
        # Phase 4: Consolidate small drivers (< 7 dogs)
        consolidation_moves = self.phase4_consolidate_small_drivers()
        
        # Phase 5: Consolidate small groups with constraints
        small_group_moves = self.phase5_consolidate_small_groups_constrained()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 OPTIMIZATION COMPLETE - REVISED 5 PHASES")
        print("=" * 60)
        print(f"✅ Phase 1: {cluster_moves} dogs clustered with nearby neighbors")
        print(f"✅ Phase 2: {outlier_moves} outliers moved from existing assignments")
        print(f"✅ Phase 3: {callout_moves} callouts assigned with capacity management")
        print(f"✅ Phase 4: {consolidation_moves} dogs moved from small drivers")
        print(f"✅ Phase 5: {small_group_moves} dogs moved from small groups")
        print(f"✅ Active drivers: {len(self.active_drivers)}")
        
        return cluster_moves + outlier_moves + callout_moves + consolidation_moves + small_group_moves

    def analyze_within_group_times(self):
        """Analyze driving times between dogs within same driver AND same group"""
        print("\n🔍 WITHIN-GROUP TIME ANALYSIS")
        print("=" * 50)
        
        max_time = 0
        max_info = None
        group_stats = []
        
        for driver in self.active_drivers:
            print(f"\nDriver: {driver}")
            print("=" * (8 + len(driver)))
            
            # Get all dogs for this driver
            driver_dogs = []
            for assignment in self.dog_assignments:
                if isinstance(assignment, dict) and assignment.get('combined', '').startswith(f"{driver}:"):
                    driver_dogs.append(assignment)
            
            if not driver_dogs:
                print("  No dogs assigned")
                continue
            
            # Group dogs by their group numbers
            groups = defaultdict(list)
            for dog in driver_dogs:
                combined = dog.get('combined', '')
                if ':' in combined:
                    groups_part = combined.split(':', 1)[1]
                    dog_groups = self.parse_dog_groups_from_callout(groups_part)
                    for g in dog_groups:
                        groups[g].append(dog)
            
            # Group capacity indicator
            capacity_info = self.calculate_driver_density(driver)
            capacity = capacity_info['capacity']
            is_dense = capacity_info['is_dense']
            avg_route_time = capacity_info['avg_time']
            
            print(f"  Route density: {avg_route_time:.1f} min avg → Capacity: {capacity} "
                  f"{'(DENSE)' if is_dense else '(STANDARD)'}")
            
            # Analyze each group
            for group_num in sorted(groups.keys()):
                dogs = groups[group_num]
                if len(dogs) < 2:
                    print(f"  Group {group_num}: {len(dogs)} dog{'s' if len(dogs) != 1 else ''} "
                          f"(no times to calculate)")
                    continue
                
                times = []
                for i in range(len(dogs)):
                    for j in range(i+1, len(dogs)):
                        time_min = self.get_time_with_fallback(
                            dogs[i].get('dog_id', ''), 
                            dogs[j].get('dog_id', '')
                        )
                        if time_min < float('inf'):
                            times.append(time_min)
                            if time_min > max_time:
                                max_time = time_min
                                max_info = (driver, group_num, dogs[i].get('dog_name', ''), 
                                          dogs[j].get('dog_name', ''))
                
                if times:
                    avg = sum(times) / len(times)
                    min_time = min(times)
                    max_time_group = max(times)
                    
                    # Visual indicator
                    if avg < 2:
                        indicator = "✅"
                    elif avg < 5:
                        indicator = "⚠️"
                    else:
                        indicator = "🚨"
                    
                    # Capacity pressure indicator
                    capacity_pct = (len(dogs) / capacity) * 100
                    if capacity_pct >= 100:
                        capacity_indicator = "🔴 OVER"
                    elif capacity_pct >= 87.5:
                        capacity_indicator = "🟡 NEAR"
                    else:
                        capacity_indicator = "🟢 GOOD"
                    
                    print(f"  Group {group_num}: {len(dogs)}/{capacity} dogs ({capacity_pct:.0f}% {capacity_indicator})")
                    print(f"    Average time: {avg:.1f} minutes {indicator}")
                    print(f"    Min time: {min_time:.1f} minutes")
                    print(f"    Max time: {max_time_group:.1f} minutes")
                    
                    # Store for summary
                    group_stats.append({
                        'driver': driver,
                        'group': group_num,
                        'dog_count': len(dogs),
                        'avg_time': avg,
                        'min_time': min_time,
                        'max_time': max_time_group
                    })
                    
                    print(f"    Dogs in group:")
                    for dog in dogs:
                        dog_name = dog.get('dog_name', 'Unknown')
                        dog_id = dog.get('dog_id', 'No ID')
                        
                        # Check if this dog is an outlier
                        min_time_to_neighbor = float('inf')
                        for other_dog in dogs:
                            if other_dog != dog:
                                other_id = other_dog.get('dog_id', '')
                                if other_id and dog_id:
                                    time_to_other = self.get_time_with_fallback(dog_id, other_id)
                                    if time_to_other < min_time_to_neighbor:
                                        min_time_to_neighbor = time_to_other
                        
                        # Determine status
                        if min_time_to_neighbor >= self.OUTLIER_THRESHOLD:
                            status = " ⚠️ EXTREME OUTLIER"
                        elif min_time_to_neighbor < self.CLUSTER_THRESHOLD:
                            status = " 🔗 CLUSTERED"
                        else:
                            # Check if it's a regular outlier using new aggressive rules
                            if times and len(times) > 0:
                                # Use more aggressive threshold
                                threshold = min(avg * self.OUTLIER_MULTIPLIER, self.OUTLIER_ABSOLUTE)
                                # Calculate this dog's average to others
                                dog_times_to_others = []
                                for other_dog in dogs:
                                    if other_dog != dog:
                                        other_id = other_dog.get('dog_id', '')
                                        if other_id and dog_id:
                                            time_to_other = self.get_time_with_fallback(dog_id, other_id)
                                            if time_to_other < float('inf'):
                                                dog_times_to_others.append(time_to_other)
                                
                                if dog_times_to_others:
                                    avg_to_others = sum(dog_times_to_others) / len(dog_times_to_others)
                                    if avg_to_others > threshold or min_time_to_neighbor > 3:
                                        status = " ⚠️ OUTLIER"
                                    else:
                                        status = ""
                                else:
                                    status = ""
                            else:
                                status = ""
                            
                        print(f"      - {dog_name} (ID: {dog_id}){status}")
        
        # Summary statistics
        if group_stats:
            print(f"\n📊 SUMMARY STATISTICS")
            print("=" * 50)
            
            all_avgs = [stat['avg_time'] for stat in group_stats]
            overall_avg = sum(all_avgs) / len(all_avgs)
            
            print(f"🎯 OVERALL AVERAGE TIME: {overall_avg:.1f} minutes")
            
            # Group-specific averages
            group_averages = defaultdict(list)
            for stat in group_stats:
                group_averages[stat['group']].append(stat['avg_time'])
            
            print(f"\n🎯 AVERAGE TIMES BY GROUP:")
            for group_num in sorted(group_averages.keys()):
                group_avg = sum(group_averages[group_num]) / len(group_averages[group_num])
                print(f"  Group {group_num}: {group_avg:.1f} minutes average")
            
            if max_info:
                print(f"\n🚨 MAXIMUM TIME BETWEEN ANY TWO DOGS:")
                print(f"  Driver: {max_info[0]}")
                print(f"  Group: {max_info[1]}")
                print(f"  Dogs: {max_info[2]} ↔ {max_info[3]}")
                print(f"  Time: {max_time:.1f} minutes")
            
            # Problem groups
            problem_groups = [stat for stat in group_stats if stat['avg_time'] > 2]
            if problem_groups:
                print(f"\n⚠️  GROUPS NEEDING ATTENTION (avg > 2 min):")
                for stat in problem_groups:
                    print(f"  {stat['driver']} Group {stat['group']}: "
                          f"{stat['avg_time']:.1f} min avg, {stat['dog_count']} dogs")
            
            print(f"\n💡 OPTIMIZATION THRESHOLDS:")
            print(f"   Cluster together: Dogs < {self.CLUSTER_THRESHOLD} min apart")
            print(f"   Phase 2 outliers: Dogs > 1.5x avg OR > 3 min from avg OR > 3 min to nearest")
            print(f"   Small drivers: Drivers with < {self.MIN_DOGS_FOR_DRIVER} total dogs")
            print(f"   Small groups: Group 1 or 3 with < {self.MIN_GROUP_SIZE} dogs")
            print(f"   Group consolidation: Max {self.GROUP_CONSOLIDATION_TIME_LIMIT} min time increase")
        else:
            print("\n📊 No group statistics available")

    def write_results_to_sheets(self):
        """Write the optimized assignments back to Google Sheets"""
        try:
            print("\n💾 Writing results back to Google Sheets...")
            sheet = self.gc.open_by_key(self.MAP_SHEET_ID).worksheet(self.MAP_TAB)
            
            updates = []
            combined_col_idx = 8  # Column H (1-indexed)
            
            for assignment in self.dog_assignments:
                if isinstance(assignment, dict):
                    row_idx = assignment.get('row_index')
                    combined = assignment.get('combined', '')
                    
                    if row_idx:
                        # Handle column indices > 26 (AA, AB, etc.)
                        if combined_col_idx <= 26:
                            cell_ref = f"{chr(ord('A') + combined_col_idx - 1)}{row_idx}"
                        else:
                            first_letter = chr(ord('A') + (combined_col_idx - 1) // 26 - 1)
                            second_letter = chr(ord('A') + (combined_col_idx - 1) % 26)
                            cell_ref = f"{first_letter}{second_letter}{row_idx}"
                        
                        updates.append({
                            'range': cell_ref,
                            'values': [[combined]]
                        })
            
            if updates:
                # Batch update with rate limiting
                try:
                    for i in range(0, len(updates), 25):
                        batch = updates[i:i+25]
                        for update in batch:
                            sheet.update(update['values'], update['range'])
                            time.sleep(1)
                        time.sleep(5)
                        print(f"📊 Updated batch {i//25 + 1}/{(len(updates)-1)//25 + 1}")
                    
                    print(f"✅ Updated {len(updates)} assignments in Google Sheets")
                except Exception as e:
                    print(f"⚠️  Error during batch update: {e}")
                    print(f"   Successfully updated {i} assignments before error")
            else:
                print("ℹ️  No updates needed")
                
        except Exception as e:
            print(f"❌ Error writing to Google Sheets: {e}")

    def send_slack_notification(self, total_changes):
        """Send notification to Slack about optimization results"""
        webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
        if not webhook_url:
            print("ℹ️  No Slack webhook configured")
            return
        
        try:
            active_drivers_count = len(self.active_drivers)
            total_dogs = len(self.dog_assignments)
            
            # Calculate utilization (approximate, as capacity is dynamic)
            # Assume average capacity of 10 dogs per group
            avg_capacity_per_group = 10
            total_capacity = active_drivers_count * avg_capacity_per_group * 3
            utilization = (total_dogs / total_capacity * 100) if total_capacity > 0 else 0
            
            message = {
                "text": f"🐕 Dog Assignment Optimization Complete",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Dog Assignment Optimization Results*\n"
                                   f"• Total changes: {total_changes}\n"
                                   f"• Active drivers: {active_drivers_count}\n"
                                   f"• Total dogs: {total_dogs}\n"
                                   f"• Utilization: {utilization:.1f}%\n"
                                   f"• Optimization strategy: Time-based (minutes)\n"
                                   f"• 6-phase optimization with clustering FIRST and final sweep LAST"
                        }
                    }
                ]
            }
            
            if hasattr(self, 'haversine_fallback_count') and self.haversine_fallback_count > 0:
                message["blocks"][0]["text"]["text"] += f"\n• Time fallbacks used: {self.haversine_fallback_count}"
            
            response = requests.post(webhook_url, json=message, timeout=10)
            if response.status_code == 200:
                print("✅ Slack notification sent")
            else:
                print(f"⚠️  Slack notification failed: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️  Slack notification error: {e}")

def main():
    """Main execution function"""
    try:
        system = DogReassignmentSystem()
        
        # Validate we have necessary data
        if not system.dog_assignments:
            print("❌ No dog assignments loaded. Cannot proceed.")
            sys.exit(1)
        
        if not system.active_drivers:
            print("❌ No active drivers found. Cannot proceed.")
            sys.exit(1)
        
        # Check if running in GitHub Actions
        is_github_actions = os.environ.get('GITHUB_ACTIONS') == 'true'
        
        if is_github_actions:
            choice = '1'
            print("🤖 Running in GitHub Actions - auto-selecting option 1 (optimization)")
        else:
            print("\nWhat would you like to do?")
            print("1. Run optimization with revised strategy")
            print("2. Analyze within-group times only")
            print("3. Both (analyze first, then optimize)")
            choice = input("Enter choice (1/2/3): ").strip()
        
        total_changes = 0
        
        if choice in ['2', '3']:
            system.analyze_within_group_times()
        
        if choice in ['1', '3']:
            total_changes = system.optimize_routes()
            system.write_results_to_sheets()
        
        if hasattr(system, 'haversine_fallback_count') and system.haversine_fallback_count > 0:
            print(f"\n📍 Haversine Fallback Usage:")
            print(f"   Used {system.haversine_fallback_count} times for time calculations")
            print(f"   These represent dog pairs not in your time matrix")
        
        system.send_slack_notification(total_changes)
        
        print(f"\n🎉 Process completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
