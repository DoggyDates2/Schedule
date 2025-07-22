# For outlier, note that it's about connectivity not center distance
            # Example: A→B→C→D→E where each arrow is 1 min = no outliers
            # But if F is 6 min from its nearest neighbor = F is outlier#!/usr/bin/env python3
"""
Dog Assignment Optimization System - TIME-BASED (MINUTES)

UPDATE: Added Phase 3 to cluster nearby dogs BEFORE capacity balancing.
This creates more efficient routes by keeping natural clusters together.

OPTIMIZATION STRATEGY (5 PHASES):
1. Phase 1: Assign ALL callout dogs to closest driver (ignore capacity)
2. Phase 2: Consolidate drivers with < 8 total dogs (give them day off)
3. Phase 3: Cluster nearby dogs (< 1 min apart) to same driver
   - Creates natural clusters for efficiency
   - Handles chains: if A near B and B near C, all go together
4. Phase 4: Balance capacity by moving outliers from over-capacity groups
   - Outliers = dogs > 5 min from nearest neighbor
5. Phase 5: Balance workloads between drivers (max 2 min added time)
   - Even out dog counts across drivers

WHY THIS WORKS:
- Phase 3 prevents many outliers from forming in the first place
- Natural clusters stay together = more efficient routes
- Capacity issues handled AFTER clustering = better overall routes
- Workload balancing ensures fairness without compromising efficiency

KEY CONCEPTS:
- All distances now in MINUTES of driving time
- Dogs NEVER change groups (1, 2, 3) - only drivers
- Nearby dogs (< 1 min) should go to same driver for efficiency
- Outliers = dogs > 5 min from nearest neighbor (NOT center distance)
- Chain routes are OK: A→B→C→D where each link ≤ 5 min = no outliers
- Dense routes (avg < 2 min): capacity 12 dogs
- Standard routes: capacity 8 dogs
- Detour value: Must collect > 0.7 dogs/minute for detour to be worthwhile
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
        print("🚀 Enhanced Dog Reassignment System - TIME-BASED OPTIMIZATION")
        print("   All distances are now in MINUTES of driving time")
        print("   Dense routes (< 2 min avg): 12 dogs per group")
        print("   Standard routes: 8 dogs per group")
        print("   Outliers: Dogs > 5 min from nearest neighbor in group")
        print("   Clusters: Dogs < 1 min apart go to same driver")
        print("   5-Phase optimization with clustering and workload balancing")
        
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
        self.MIN_DOGS_FOR_DRIVER = 8   # Drivers with fewer than 8 dogs get consolidated
        self.CAPACITY_THRESHOLD = 2    # Minutes within which to consider equal
        
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
                    if row[col_index].strip():
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
                    dog_id = row[dog_id_idx].strip() if row[dog_id_idx] else ""
                    lat_str = row[lat_idx].strip() if row[lat_idx] else ""
                    lng_str = row[lng_idx].strip() if row[lng_idx] else ""
                    
                    if dog_id and lat_str and lng_str:
                        try:
                            lat = float(lat_str)
                            lng = float(lng_str)
                            self.dog_coordinates[dog_id] = (lat, lng)
                            coordinates_loaded += 1
                        except ValueError:
                            pass
            
            print(f"✅ Loaded coordinates for {coordinates_loaded} dogs")
            
        except Exception as e:
            print(f"❌ Error loading dog coordinates: {e}")
            self.dog_coordinates = {}

    def haversine_time(self, dog_id1, dog_id2):
        """Calculate driving time between two dogs using haversine formula"""
        if dog_id1 not in self.dog_coordinates or dog_id2 not in self.dog_coordinates:
            return float('inf')
        
        lat1, lon1 = self.dog_coordinates[dog_id1]
        lat2, lon2 = self.dog_coordinates[dog_id2]
        
        # Convert to radians
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        # Earth's radius in miles
        R = 3959
        distance_miles = R * c
        
        # Convert to driving time in minutes
        driving_factor = 1.3  # Account for roads vs straight line
        driving_miles = distance_miles * driving_factor
        driving_minutes = driving_miles * self.MILES_TO_MINUTES
        
        return driving_minutes

    def get_time_with_fallback(self, dog_id1, dog_id2):
        """Get driving time between dogs with haversine fallback"""
        # First, try the time matrix
        time_minutes = self.safe_get_time(dog_id1, dog_id2)
        if time_minutes < float('inf'):
            return time_minutes
        
        # Fallback to haversine calculation
        haversine_time = self.haversine_time(dog_id1, dog_id2)
        if haversine_time < float('inf'):
            if not hasattr(self, 'haversine_fallback_count'):
                self.haversine_fallback_count = 0
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
        is_dense = avg_time < self.DENSE_ROUTE_THRESHOLD
        capacity = 12 if is_dense else 8
        
        return {
            'dog_count': len(driver_dogs),
            'avg_time': avg_time,
            'capacity': capacity,
            'is_dense': is_dense
        }

    def calculate_detour_value(self, detour_minutes, dogs_collected):
        """
        Calculate if a detour is worth it based on dogs collected
        Detour Value = Dogs collected / Additional minutes
        Threshold: > 0.7 dogs/minute is acceptable
        Example: 5 dogs for 7 minutes = 0.71 ✓
        """
        if detour_minutes <= 0:
            return float('inf')  # No detour needed
        
        value = dogs_collected / detour_minutes
        is_acceptable = value > 0.7
        
        return {
            'value': value,
            'is_acceptable': is_acceptable,
            'dogs_per_minute': value
        }
        """Parse groups from callout for capacity checking"""
        if not callout:
            return []
        
        callout_clean = callout.lstrip(':')
        groups = []
        
        for char in callout_clean:
            if char in ['1', '2', '3']:
                group_num = int(char)
                if group_num not in groups:
                    groups.append(group_num)
        
        return sorted(groups)

    def phase1_assign_all_callouts(self):
        """PHASE 1: Assign ALL callout dogs to closest driver (ignore capacity)
        
        Gets all unassigned dogs into the system quickly.
        Capacity issues will be fixed in later phases.
        """
        print("\n🎯 PHASE 1: Assigning ALL callouts to closest driver (ignoring capacity)")
        print("=" * 60)
        
        # Find all callouts that need assignment
        callouts = []
        for assignment in self.dog_assignments:
            if isinstance(assignment, dict):
                callout = assignment.get('callout', '').strip()
                combined = assignment.get('combined', '').strip()
                
                if callout and not combined:
                    callouts.append(assignment)
        
        print(f"📊 Found {len(callouts)} callout dogs to assign")
        
        assignments_made = 0
        
        for callout in callouts:
            dog_name = callout.get('dog_name', 'Unknown')
            dog_id = callout.get('dog_id', '')
            original_callout = callout.get('callout', '').strip()
            
            if not original_callout:
                print(f"❌ {dog_name}: No callout specified")
                continue
            
            # Find closest driver (considering ALL drivers)
            best_driver = None
            best_time = float('inf')
            
            for driver in self.active_drivers:
                # Get driver's current dogs
                driver_dogs = []
                for assignment in self.dog_assignments:
                    if isinstance(assignment, dict) and assignment.get('combined', '').startswith(f"{driver}:"):
                        driver_dogs.append(assignment)
                
                if not driver_dogs:
                    # Empty driver - assign with default time
                    avg_time = 5.0  # Default 5 minutes
                else:
                    # Calculate average time to driver's existing dogs
                    times = []
                    for existing_dog in driver_dogs:
                        existing_dog_id = existing_dog.get('dog_id', '')
                        if existing_dog_id and dog_id:
                            time_min = self.get_time_with_fallback(dog_id, existing_dog_id)
                            if time_min < float('inf'):
                                times.append(time_min)
                    
                    if times:
                        avg_time = sum(times) / len(times)
                    else:
                        avg_time = 5.0  # Default if can't calculate
                
                if avg_time < best_time:
                    best_time = avg_time
                    best_driver = driver
            
            if best_driver:
                # Preserve exact original callout format
                if original_callout.startswith(':'):
                    group_part = original_callout
                else:
                    group_part = ':' + original_callout
                
                new_assignment = f"{best_driver}{group_part}"
                callout['combined'] = new_assignment
                
                # Check detour value if this was far
                if best_time > 5:
                    detour_calc = self.calculate_detour_value(best_time, 1)
                    detour_msg = f" [Detour: {detour_calc['dogs_per_minute']:.2f} dogs/min - "
                    detour_msg += "✓]" if detour_calc['is_acceptable'] else "✗]"
                else:
                    detour_msg = ""
                
                print(f"   ✅ {dog_name} → {best_driver} ({best_time:.1f} min avg) "
                      f"for callout '{original_callout}'{detour_msg}")
                assignments_made += 1
            else:
                print(f"❌ {dog_name}: No driver found")
        
        print(f"\n✅ Phase 1 Complete: {assignments_made} dogs assigned")
        return assignments_made

    def phase2_consolidate_small_drivers(self):
        """PHASE 2: Consolidate drivers with < 8 total dogs
        
        Gives small drivers the day off by redistributing their dogs.
        Dogs keep their original group assignments (1, 2, 3).
        """
        print("\n🔄 PHASE 2: Consolidating small drivers (< 8 dogs)")
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
            if len(dogs) < self.MIN_DOGS_FOR_DRIVER:  # Less than 8 dogs
                drivers_to_consolidate.append((driver, dogs))
        
        print(f"🔍 Found {len(drivers_to_consolidate)} drivers to consolidate")
        
        dogs_moved = 0
        
        for driver_to_remove, dogs_to_move in drivers_to_consolidate:
            print(f"\n📤 Consolidating {driver_to_remove} ({len(dogs_to_move)} dogs):")
            
            # Calculate average position of this driver's dogs
            driver_center_lat = 0
            driver_center_lng = 0
            valid_coords = 0
            
            for dog in dogs_to_move:
                dog_id = dog.get('dog_id', '')
                if dog_id in self.dog_coordinates:
                    lat, lng = self.dog_coordinates[dog_id]
                    driver_center_lat += lat
                    driver_center_lng += lng
                    valid_coords += 1
            
            if valid_coords > 0:
                driver_center_lat /= valid_coords
                driver_center_lng /= valid_coords
            
            # Reassign each dog to the best alternative driver
            for dog in dogs_to_move:
                dog_name = dog.get('dog_name', 'Unknown')
                dog_id = dog.get('dog_id', '')
                original_groups = dog['combined'].split(':', 1)[1]
                
                # Find best alternative driver
                best_driver = None
                best_time = float('inf')
                
                for other_driver in self.active_drivers:
                    if other_driver == driver_to_remove:
                        continue
                    
                    # Calculate average time to this driver's dogs
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
                    # CRITICAL: Preserve original group assignment
                    dog['combined'] = f"{best_driver}:{original_groups}"
                    dogs_moved += 1
                    print(f"   ✅ {dog_name} → {best_driver} ({best_time:.1f} min)")
            
            # Remove this driver from active drivers
            self.active_drivers.discard(driver_to_remove)
            print(f"   🏠 {driver_to_remove} can take the day off!")
        
        print(f"\n✅ Phase 2 Complete: {dogs_moved} dogs moved, "
              f"{len(drivers_to_consolidate)} drivers eliminated")
        return dogs_moved

    def phase3_cluster_nearby_dogs(self):
        """PHASE 3: Ensure dogs < 1 minute apart in same group go to same driver
        
        This phase runs BEFORE capacity balancing to create natural clusters.
        Helps prevent outliers by keeping nearby dogs together.
        
        Uses connected components to find clusters:
        - If A is < 1 min from B, and B is < 1 min from C, then A, B, C form a cluster
        - All dogs in a cluster should go to the same driver
        - Driver with most dogs in the cluster wins (if capacity allows)
        - Respects capacity limits - won't cluster if it would exceed capacity
        """
        print("\n🔗 PHASE 3: Clustering nearby dogs (< 1 minute apart)")
        print("=" * 60)
        
        moves_made = 0
        
        # Process each group separately
        for group_num in [1, 2, 3]:
            print(f"\n📊 Processing Group {group_num}:")
            
            # Get all dogs in this group with their current drivers
            group_dogs = []
            for assignment in self.dog_assignments:
                if isinstance(assignment, dict) and assignment.get('combined', '').strip():
                    groups = self.parse_dog_groups_from_callout(assignment['combined'].split(':', 1)[1])
                    if group_num in groups:
                        combined = assignment['combined']
                        driver = combined.split(':')[0] if ':' in combined else None
                        if driver and driver not in ['Field', 'Parking']:
                            group_dogs.append({
                                'assignment': assignment,
                                'driver': driver,
                                'dog_id': assignment.get('dog_id', ''),
                                'dog_name': assignment.get('dog_name', 'Unknown')
                            })
            
            if len(group_dogs) < 2:
                print(f"   Only {len(group_dogs)} dogs in group - skipping")
                continue
            
            # Build adjacency list of dogs < 1 minute apart
            nearby_dogs = defaultdict(list)
            for i, dog1_data in enumerate(group_dogs):
                dog1_id = dog1_data['dog_id']
                
                for j, dog2_data in enumerate(group_dogs):
                    if i != j:
                        dog2_id = dog2_data['dog_id']
                        time_between = self.get_time_with_fallback(dog1_id, dog2_id)
                        
                        if time_between < self.CLUSTER_THRESHOLD:
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
            for cluster_idx, cluster in enumerate(clusters):
                # Check if all dogs in cluster have same driver
                drivers_in_cluster = set(dog['driver'] for dog in cluster)
                # Check if all dogs in cluster have same driver
                drivers_in_cluster = set(dog['driver'] for dog in cluster)
                
                if len(drivers_in_cluster) > 1:
                    # Need to consolidate - choose driver with most dogs in cluster
                    driver_counts = Counter(dog['driver'] for dog in cluster)
                    
                    # Check capacity constraints for each potential driver
                    valid_drivers = []
                    for driver, count in driver_counts.items():
                        # Get current capacity for this driver
                        capacity_info = self.calculate_driver_density(driver)
                        capacity = capacity_info['capacity']
                        
                        # Count current dogs in this group for this driver
                        current_in_group = sum(1 for d in group_dogs if d['driver'] == driver)
                        
                        # Calculate how many additional dogs would be added
                        dogs_to_add = len(cluster) - count
                        
                        # Check if driver can handle the additional dogs
                        if current_in_group + dogs_to_add <= capacity:
                            valid_drivers.append((driver, count))
                    
                    if not valid_drivers:
                        print(f"\n   ⚠️  Cannot cluster {len(cluster)} dogs - would exceed all drivers' capacity")
                        print(f"      Cluster contains:")
                        for dog in cluster:
                            print(f"      - {dog['dog_name']} (with {dog['driver']})")
                        continue
                    
                    # Choose driver with most dogs in cluster from valid options
                    valid_drivers.sort(key=lambda x: -x[1])
                    winning_driver = valid_drivers[0][0]
                    
                    print(f"\n   🔗 Found cluster of {len(cluster)} dogs (< {self.CLUSTER_THRESHOLD} min chain):")
                    for dog in cluster:
                        print(f"      - {dog['dog_name']} (currently with {dog['driver']})")
                    print(f"      → Consolidating all to {winning_driver}")
                    
                    # Move all dogs to winning driver
                    for dog_data in cluster:
                        if dog_data['driver'] != winning_driver:
                            # Update assignment
                            original_groups = dog_data['assignment']['combined'].split(':', 1)[1]
                            dog_data['assignment']['combined'] = f"{winning_driver}:{original_groups}"
                            dog_data['driver'] = winning_driver
                            moves_made += 1
                            print(f"      ✅ Moved {dog_data['dog_name']} to {winning_driver}")
            
            if not clusters:
                print(f"   No clusters found (no dogs < {self.CLUSTER_THRESHOLD} min apart with different drivers)")
            else:
                print(f"   Found {len(clusters)} total clusters in group {group_num}")
        
        print(f"\n✅ Phase 3 Complete: {moves_made} dogs clustered with nearby neighbors")
        if moves_made > 0:
            print(f"   (Creates efficient routes by keeping nearby dogs together)")
        return moves_made

    def phase4_balance_capacity(self):
        """PHASE 4 (was 3): Balance capacity by moving outliers from dense groups
        
        Fixes over-capacity issues by identifying and moving outlier dogs.
        Outliers are dogs > 5 min from their nearest neighbor.
        Dogs stay in their assigned groups (1, 2, 3).
        """
        print("\n⚖️ PHASE 4: Balancing capacity")
        print("=" * 60)
        
        moves_made = 0
        iteration = 0
        max_iterations = 50
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}:")
            
            # Find over-capacity groups
            over_capacity_groups = []
            
            for driver in self.active_drivers:
                capacity_info = self.calculate_driver_density(driver)
                capacity = capacity_info['capacity']
                
                # Count dogs per group for this driver
                group_counts = defaultdict(int)
                driver_dogs = defaultdict(list)
                
                for assignment in self.dog_assignments:
                    if isinstance(assignment, dict) and assignment.get('combined', '').startswith(f"{driver}:"):
                        combined = assignment['combined']
                        groups = self.parse_dog_groups_from_callout(combined.split(':', 1)[1])
                        for group in groups:
                            group_counts[group] += 1
                            driver_dogs[group].append(assignment)
                
                # Check each group's capacity
                for group_num, count in group_counts.items():
                    if count > capacity:
                        over_capacity_groups.append({
                            'driver': driver,
                            'group': group_num,
                            'count': count,
                            'capacity': capacity,
                            'over_by': count - capacity,
                            'dogs': driver_dogs[group_num],
                            'avg_time': capacity_info['avg_time']
                        })
            
            if not over_capacity_groups:
                print("✅ All groups within capacity!")
                break
            
            # Sort by density (most dense first) and over-capacity amount
            over_capacity_groups.sort(key=lambda x: (x['avg_time'], -x['over_by']))
            
            print(f"Found {len(over_capacity_groups)} over-capacity groups")
            
            # Process the most problematic group
            group_to_fix = over_capacity_groups[0]
            print(f"\n🎯 Fixing {group_to_fix['driver']} Group {group_to_fix['group']}: "
                  f"{group_to_fix['count']}/{group_to_fix['capacity']} dogs")
            
            # Find outliers in this group
            outliers = self.find_group_outliers(group_to_fix['dogs'])
            
            if not outliers:
                print(f"   No outliers found (all dogs within {self.OUTLIER_THRESHOLD} min of a neighbor)")
                continue
            
            print(f"   Found {len(outliers)} outlier(s) in group:")
            for outlier in outliers[:3]:  # Show top 3 outliers
                print(f"     - {outlier['dog'].get('dog_name', 'Unknown')}: "
                      f"{outlier['min_time_to_nearest']:.1f} min from nearest neighbor")
            
            # Try to move the most distant outlier
            moved = False
            for outlier in outliers[:1]:  # Move one at a time
                dog_name = outlier['dog'].get('dog_name', 'Unknown')
                dog_id = outlier['dog'].get('dog_id', '')
                
                print(f"   🎯 Attempting to move outlier: {dog_name} "
                      f"({outlier['min_time_to_nearest']:.1f} min from nearest neighbor)")
                
                # Find best alternative placement
                best_option = self.find_best_alternative_placement(
                    outlier['dog'], group_to_fix['group'], group_to_fix['driver']
                )
                
                if best_option:
                    # CRITICAL: Dogs keep their group numbers - only driver changes
                    # Dogs MUST stay in their assigned group (1, 2, or 3)
                    old_combined = outlier['dog']['combined']
                    original_groups = old_combined.split(':', 1)[1]
                    new_combined = f"{best_option['driver']}:{original_groups}"
                    outlier['dog']['combined'] = new_combined
                    
                    print(f"   ✅ Moved {dog_name}: {old_combined} → {new_combined} "
                          f"({best_option['time']:.1f} min)")
                    moves_made += 1
                    moved = True
                    break
                else:
                    print(f"   ❌ No suitable alternative found for {dog_name}")
            
            if not moved:
                print("   ⚠️  Could not resolve this group - routes may be well-connected")
        
        print(f"\n✅ Phase 4 Complete: {moves_made} moves made")
        return moves_made

    def phase5_balance_driver_workload(self):
        """PHASE 5 (was 4): Balance workload between drivers to even out dog counts
        
        Moves dogs from overloaded to underloaded drivers.
        Only moves if it adds ≤ 2 minutes to the route.
        Dogs stay in their assigned groups (1, 2, 3).
        """
        print("\n⚖️ PHASE 5: Balancing driver workloads")
        print("=" * 60)
        
        # Calculate current capacity for each driver
        driver_loads = {}
        driver_dogs_by_group = defaultdict(lambda: defaultdict(list))
        
        for driver in self.active_drivers:
            total_dogs = 0
            for assignment in self.dog_assignments:
                if isinstance(assignment, dict) and assignment.get('combined', '').startswith(f"{driver}:"):
                    total_dogs += 1
                    # Track dogs by group for this driver
                    combined = assignment['combined']
                    groups = self.parse_dog_groups_from_callout(combined.split(':', 1)[1])
                    for group in groups:
                        driver_dogs_by_group[driver][group].append(assignment)
            
            driver_loads[driver] = total_dogs
        
        if not driver_loads:
            print("❌ No active drivers to balance")
            return 0
        
        # Find average load
        avg_load = sum(driver_loads.values()) / len(driver_loads)
        
        print(f"📊 Current driver loads:")
        for driver, load in sorted(driver_loads.items(), key=lambda x: -x[1]):
            deviation = load - avg_load
            indicator = "🔴" if deviation > 3 else "🟡" if deviation > 1 else "🟢"
            print(f"   {driver}: {load} dogs ({deviation:+.1f} from avg) {indicator}")
        
        print(f"\n🎯 Target average: {avg_load:.1f} dogs per driver")
        
        # Find overloaded and underloaded drivers
        overloaded = [(d, load) for d, load in driver_loads.items() if load > avg_load + 1]
        underloaded = [(d, load) for d, load in driver_loads.items() if load < avg_load - 1]
        
        if not overloaded or not underloaded:
            print("✅ Workload is already well balanced!")
            return 0
        
        moves_made = 0
        max_moves = 10  # Limit to prevent excessive changes
        
        # Try to move dogs from overloaded to underloaded drivers
        for over_driver, over_load in sorted(overloaded, key=lambda x: -x[1]):
            if moves_made >= max_moves:
                break
                
            for under_driver, under_load in sorted(underloaded, key=lambda x: x[1]):
                if moves_made >= max_moves:
                    break
                    
                if driver_loads[over_driver] <= avg_load + 0.5:
                    break  # This driver is now balanced
                
                print(f"\n🔍 Checking moves from {over_driver} ({driver_loads[over_driver]} dogs) "
                      f"to {under_driver} ({driver_loads[under_driver]} dogs)")
                
                # Look for dogs that could be moved within same group
                best_move = None
                best_added_time = float('inf')
                
                # Check each group
                for group_num in [1, 2, 3]:
                    over_dogs = driver_dogs_by_group[over_driver][group_num]
                    under_dogs = driver_dogs_by_group[under_driver][group_num]
                    
                    if not over_dogs:
                        continue
                    
                    # Check capacity for underloaded driver
                    capacity_info = self.calculate_driver_density(under_driver)
                    capacity = capacity_info['capacity']
                    
                    if len(under_dogs) >= capacity:
                        continue  # No room in this group
                    
                    # Find best dog to move from this group
                    for dog in over_dogs:
                        dog_id = dog.get('dog_id', '')
                        if not dog_id:
                            continue
                        
                        # Calculate how much time this would add to under_driver's route
                        if not under_dogs:
                            added_time = 0  # First dog in group
                        else:
                            times = []
                            for other_dog in under_dogs:
                                other_id = other_dog.get('dog_id', '')
                                if other_id:
                                    time_min = self.get_time_with_fallback(dog_id, other_id)
                                    if time_min < float('inf'):
                                        times.append(time_min)
                            
                            added_time = min(times) if times else 0
                        
                        # Only consider if it adds less than 2 minutes
                        if added_time <= 2:
                            if added_time < best_added_time:
                                best_added_time = added_time
                                best_move = {
                                    'dog': dog,
                                    'group': group_num,
                                    'added_time': added_time,
                                    'from_driver': over_driver,
                                    'to_driver': under_driver
                                }
                
                # Make the best move if found
                if best_move:
                    dog = best_move['dog']
                    dog_name = dog.get('dog_name', 'Unknown')
                    
                    # CRITICAL: Update assignment - KEEP SAME GROUP NUMBER
                    # Dogs CANNOT change groups (1, 2, 3) - only drivers
                    original_groups = dog['combined'].split(':', 1)[1]
                    dog['combined'] = f"{best_move['to_driver']}:{original_groups}"
                    
                    # Update tracking
                    driver_loads[best_move['from_driver']] -= 1
                    driver_loads[best_move['to_driver']] += 1
                    
                    # Update group tracking
                    driver_dogs_by_group[over_driver][best_move['group']].remove(dog)
                    driver_dogs_by_group[under_driver][best_move['group']].append(dog)
                    
                    print(f"   ✅ Moved {dog_name} (Group {best_move['group']}) "
                          f"from {best_move['from_driver']} to {best_move['to_driver']} "
                          f"(+{best_move['added_time']:.1f} min)")
                    moves_made += 1
                else:
                    print(f"   ❌ No suitable moves found (all would add > 2 min)")
        
        # Final summary
        print(f"\n📊 Final driver loads:")
        for driver, load in sorted(driver_loads.items(), key=lambda x: -x[1]):
            deviation = load - avg_load
            indicator = "🔴" if deviation > 3 else "🟡" if deviation > 1 else "🟢"
            print(f"   {driver}: {load} dogs ({deviation:+.1f} from avg) {indicator}")
        
        print(f"\n✅ Phase 5 Complete: {moves_made} moves made to balance workload")
        return moves_made

    def find_group_outliers(self, dogs):
        """Find outlier dogs in a group based on minimum distance to nearest neighbor
        
        A dog is an outlier if it's far from ALL other dogs, not just far from center.
        Example: In chain A→B→C→D→E where each arrow is 1 min, no dogs are outliers.
        But if F is 6 min from its nearest neighbor, F is an outlier.
        """
        if len(dogs) < 2:
            return []
        
        outliers = []
        
        for i, dog in enumerate(dogs):
            dog_id = dog.get('dog_id', '')
            if not dog_id:
                continue
            
            # Find MINIMUM time to any other dog in group
            min_time = float('inf')
            times_to_others = []
            
            for j, other_dog in enumerate(dogs):
                if i != j:
                    other_id = other_dog.get('dog_id', '')
                    if other_id:
                        time_min = self.get_time_with_fallback(dog_id, other_id)
                        if time_min < float('inf'):
                            times_to_others.append(time_min)
                            if time_min < min_time:
                                min_time = time_min
            
            # A dog is an outlier if its nearest neighbor is far away
            if min_time >= self.OUTLIER_THRESHOLD:
                outliers.append({
                    'dog': dog,
                    'min_time_to_nearest': min_time,
                    'avg_time': sum(times_to_others) / len(times_to_others) if times_to_others else float('inf')
                })
        
        # Sort by minimum time to nearest neighbor (most isolated first)
        outliers.sort(key=lambda x: -x['min_time_to_nearest'])
        return outliers

    def find_best_alternative_placement(self, dog, current_group, current_driver):
        """Find best alternative placement for a dog - SAME GROUP ONLY"""
        dog_id = dog.get('dog_id', '')
        if not dog_id:
            return None
        
        best_options = []
        
        # Check all other drivers for the SAME group only
        for driver in self.active_drivers:
            if driver == current_driver:
                continue
            
            # Get capacity info for this driver
            capacity_info = self.calculate_driver_density(driver)
            capacity = capacity_info['capacity']
            
            # Count current dogs in the SAME group for this driver
            group_count = 0
            group_dogs = []
            
            for assignment in self.dog_assignments:
                if isinstance(assignment, dict) and assignment.get('combined', '').startswith(f"{driver}:"):
                    groups = self.parse_dog_groups_from_callout(assignment['combined'].split(':', 1)[1])
                    if current_group in groups:
                        group_count += 1
                        group_dogs.append(assignment)
            
            # Check if there's capacity
            if group_count >= capacity:
                continue
            
            # Calculate average time to dogs in this group
            if not group_dogs:
                avg_time = 10.0  # Default for empty groups
            else:
                times = []
                for other_dog in group_dogs:
                    other_id = other_dog.get('dog_id', '')
                    if other_id:
                        time_min = self.get_time_with_fallback(dog_id, other_id)
                        if time_min < float('inf'):
                            times.append(time_min)
                
                if times:
                    avg_time = sum(times) / len(times)
                else:
                    avg_time = 10.0
            
            best_options.append({
                'driver': driver,
                'group': current_group,  # Always same group
                'time': avg_time,
                'capacity_remaining': capacity - group_count - 1
            })
        
        if not best_options:
            return None
        
        # Sort options by time, but consider capacity within threshold
        best_options.sort(key=lambda x: x['time'])
        
        # If multiple options are within CAPACITY_THRESHOLD minutes, prefer the one with more capacity
        best_time = best_options[0]['time']
        close_options = [opt for opt in best_options if opt['time'] <= best_time + self.CAPACITY_THRESHOLD]
        
        if len(close_options) > 1:
            close_options.sort(key=lambda x: -x['capacity_remaining'])
        
        return close_options[0] if close_options else best_options[0]

    def phase4_balance_driver_capacity(self):
        """PHASE 4: Balance capacity between drivers to even out workload"""
        print("\n⚖️ PHASE 4: Balancing driver workloads")
        print("=" * 60)
        
        # Calculate current capacity for each driver
        driver_loads = {}
        driver_dogs_by_group = defaultdict(lambda: defaultdict(list))
        
        for driver in self.active_drivers:
            total_dogs = 0
            for assignment in self.dog_assignments:
                if isinstance(assignment, dict) and assignment.get('combined', '').startswith(f"{driver}:"):
                    total_dogs += 1
                    # Track dogs by group for this driver
                    combined = assignment['combined']
                    groups = self.parse_dog_groups_from_callout(combined.split(':', 1)[1])
                    for group in groups:
                        driver_dogs_by_group[driver][group].append(assignment)
            
            driver_loads[driver] = total_dogs
        
        if not driver_loads:
            print("❌ No active drivers to balance")
            return 0
        
        # Find average load
        avg_load = sum(driver_loads.values()) / len(driver_loads)
        
        print(f"📊 Current driver loads:")
        for driver, load in sorted(driver_loads.items(), key=lambda x: -x[1]):
            deviation = load - avg_load
            indicator = "🔴" if deviation > 3 else "🟡" if deviation > 1 else "🟢"
            print(f"   {driver}: {load} dogs ({deviation:+.1f} from avg) {indicator}")
        
        print(f"\n🎯 Target average: {avg_load:.1f} dogs per driver")
        
        # Find overloaded and underloaded drivers
        overloaded = [(d, load) for d, load in driver_loads.items() if load > avg_load + 1]
        underloaded = [(d, load) for d, load in driver_loads.items() if load < avg_load - 1]
        
        if not overloaded or not underloaded:
            print("✅ Workload is already well balanced!")
            return 0
        
        moves_made = 0
        max_moves = 10  # Limit to prevent excessive changes
        
        # Try to move dogs from overloaded to underloaded drivers
        for over_driver, over_load in sorted(overloaded, key=lambda x: -x[1]):
            if moves_made >= max_moves:
                break
                
            for under_driver, under_load in sorted(underloaded, key=lambda x: x[1]):
                if moves_made >= max_moves:
                    break
                    
                if driver_loads[over_driver] <= avg_load + 0.5:
                    break  # This driver is now balanced
                
                print(f"\n🔍 Checking moves from {over_driver} ({driver_loads[over_driver]} dogs) "
                      f"to {under_driver} ({driver_loads[under_driver]} dogs)")
                
                # Look for dogs that could be moved within same group
                best_move = None
                best_added_time = float('inf')
                
                # Check each group
                for group_num in [1, 2, 3]:
                    over_dogs = driver_dogs_by_group[over_driver][group_num]
                    under_dogs = driver_dogs_by_group[under_driver][group_num]
                    
                    if not over_dogs:
                        continue
                    
                    # Check capacity for underloaded driver
                    capacity_info = self.calculate_driver_density(under_driver)
                    capacity = capacity_info['capacity']
                    
                    if len(under_dogs) >= capacity:
                        continue  # No room in this group
                    
                    # Find best dog to move from this group
                    for dog in over_dogs:
                        dog_id = dog.get('dog_id', '')
                        if not dog_id:
                            continue
                        
                        # Calculate how much time this would add to under_driver's route
                        if not under_dogs:
                            added_time = 0  # First dog in group
                        else:
                            times = []
                            for other_dog in under_dogs:
                                other_id = other_dog.get('dog_id', '')
                                if other_id:
                                    time_min = self.get_time_with_fallback(dog_id, other_id)
                                    if time_min < float('inf'):
                                        times.append(time_min)
                            
                            added_time = min(times) if times else 0
                        
                        # Only consider if it adds less than 2 minutes
                        if added_time <= 2:
                            if added_time < best_added_time:
                                best_added_time = added_time
                                best_move = {
                                    'dog': dog,
                                    'group': group_num,
                                    'added_time': added_time,
                                    'from_driver': over_driver,
                                    'to_driver': under_driver
                                }
                
                # Make the best move if found
                if best_move:
                    dog = best_move['dog']
                    dog_name = dog.get('dog_name', 'Unknown')
                    
                    # Update assignment - KEEP SAME GROUP
                    original_groups = dog['combined'].split(':', 1)[1]
                    dog['combined'] = f"{best_move['to_driver']}:{original_groups}"
                    
                    # Update tracking
                    driver_loads[best_move['from_driver']] -= 1
                    driver_loads[best_move['to_driver']] += 1
                    
                    # Update group tracking
                    driver_dogs_by_group[over_driver][best_move['group']].remove(dog)
                    driver_dogs_by_group[under_driver][best_move['group']].append(dog)
                    
                    print(f"   ✅ Moved {dog_name} (Group {best_move['group']}) "
                          f"from {best_move['from_driver']} to {best_move['to_driver']} "
                          f"(+{best_move['added_time']:.1f} min)")
                    moves_made += 1
                else:
                    print(f"   ❌ No suitable moves found (all would add > 2 min)")
        
        # Final summary
        print(f"\n📊 Final driver loads:")
        for driver, load in sorted(driver_loads.items(), key=lambda x: -x[1]):
            deviation = load - avg_load
            indicator = "🔴" if deviation > 3 else "🟡" if deviation > 1 else "🟢"
            print(f"   {driver}: {load} dogs ({deviation:+.1f} from avg) {indicator}")
        
        print(f"\n✅ Phase 4 Complete: {moves_made} moves made to balance workload")
        return moves_made
        """Main optimization function following the new strategy"""
        print("\n🚀 STARTING NEW OPTIMIZATION STRATEGY")
        print("=" * 60)
        
        # Phase 1: Assign all callouts (ignore capacity)
        callouts_assigned = self.phase1_assign_all_callouts()
        
        # Phase 2: Consolidate small drivers
        dogs_consolidated = self.phase2_consolidate_small_drivers()
        
        # Phase 3: Balance capacity
        moves_made = self.phase3_balance_capacity()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 OPTIMIZATION COMPLETE - 5 PHASES")
        print("=" * 60)
        print(f"✅ Phase 1: {callouts_assigned} callouts assigned")
        print(f"✅ Phase 2: {dogs_consolidated} dogs consolidated")
        print(f"✅ Phase 3: {moves_made} capacity balancing moves")
        print(f"✅ Active drivers: {len(self.active_drivers)}")
        
        return callouts_assigned + dogs_consolidated + moves_made

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
                            status = " ⚠️ OUTLIER"
                        elif min_time_to_neighbor < self.CLUSTER_THRESHOLD:
                            status = " 🔗 CLUSTERED"
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
            print(f"   Outlier definition: Dogs > {self.OUTLIER_THRESHOLD} min from nearest neighbor")
            print("   (Not about distance from center - chains of dogs are OK)")
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
                        cell_ref = f"{chr(ord('A') + combined_col_idx - 1)}{row_idx}"
                        updates.append({
                            'range': cell_ref,
                            'values': [[combined]]
                        })
            
            if updates:
                # Batch update with rate limiting
                for i in range(0, len(updates), 25):
                    batch = updates[i:i+25]
                    for update in batch:
                        sheet.update(update['values'], update['range'])
                        time.sleep(1)
                    time.sleep(5)
                    print(f"📊 Updated batch {i//25 + 1}/{(len(updates)-1)//25 + 1}")
                
                print(f"✅ Updated {len(updates)} assignments in Google Sheets")
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
            
            # Calculate utilization
            total_capacity = active_drivers_count * 8 * 3
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
                                   f"• 5-phase optimization with clustering"
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
        
        # Check if running in GitHub Actions
        is_github_actions = os.environ.get('GITHUB_ACTIONS') == 'true'
        
        if is_github_actions:
            choice = '1'
            print("🤖 Running in GitHub Actions - auto-selecting option 1 (optimization)")
        else:
            print("\nWhat would you like to do?")
            print("1. Run optimization with new strategy")
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
