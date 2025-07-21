#!/usr/bin/env python3
"""
Dog Assignment Optimization System with Dynamic Capacity - VERBOSE SWAP LOGGING
CRITICAL FIX: Preserves exact original group assignments (only changes driver names)
- Default capacity: 8 dogs per group
- Dense routes (avg < 0.5mi): 12 dogs per group
- Automatic capacity adjustment based on route density
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
        print("🚀 Enhanced Dog Reassignment System - DYNAMIC CAPACITY")
        print("   Dense routes (< 0.5mi avg): 12 dogs per group")
        print("   Standard routes: 8 dogs per group")
        
        # Google Sheets IDs
        self.MAP_SHEET_ID = "1-KTOfTKXk_sX7nO7eGmW73JLi8TJBvv5gobK6gyrc7U"
        self.DISTANCE_MATRIX_SHEET_ID = "1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg"
        self.MAP_TAB = "Map"
        self.MATRIX_TAB = "Matrix"  # Try these if Matrix doesn't work:
        # Common possibilities: "Sheet1", "Distance Matrix", "Distances", "Matrix", or check the tab finder script
        
        # System parameters (MUST BE BEFORE LOADING DATA)
        self.PREFERRED_DISTANCE = 0.2
        self.MAX_DISTANCE = 0.5
        self.ABSOLUTE_MAX_DISTANCE = 1.5
        self.CASCADING_MOVE_MAX = 0.7
        self.ADJACENT_GROUP_DISTANCE = 0.1
        self.EXCLUSION_DISTANCE = 200.0
        self.DENSE_ROUTE_THRESHOLD = 0.5  # Routes denser than this get capacity 12
        
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
        """Load distance matrix from Google Sheets"""
        try:
            # Try by tab name first, then by sheet ID as fallback
            spreadsheet = self.gc.open_by_key(self.DISTANCE_MATRIX_SHEET_ID)
            try:
                sheet = spreadsheet.worksheet(self.MATRIX_TAB)
                print(f"✅ Found sheet by name: '{self.MATRIX_TAB}'")
            except gspread.WorksheetNotFound:
                # Try by sheet ID (from your URL: gid=398422902)
                sheet = spreadsheet.get_worksheet_by_id(398422902)
                print(f"✅ Found sheet by ID: 398422902")
                if sheet:
                    print(f"✅ Sheet title is actually: '{sheet.title}'")
            
            all_values = sheet.get_all_values()
            
            if not all_values:
                print("❌ Distance matrix sheet is empty")
                return
            
            # Get dog IDs from the first row (headers)
            dog_ids = [val for val in all_values[0][1:] if val.strip()]
            print(f"🔍 Found {len(dog_ids)} dog IDs in first row")
            
            if len(dog_ids) == 0:
                print("❌ No dog IDs found in distance matrix headers")
                return
            
            # Show sample IDs for verification
            print(f"📋 Sample dog IDs: {dog_ids[:5]}...")
            
            # Build distance matrix
            distances_loaded = 0
            for i, row in enumerate(all_values[1:], 1):
                if i <= len(dog_ids):
                    from_dog = dog_ids[i-1]
                    for j, distance_str in enumerate(row[1:], 0):
                        if j < len(dog_ids) and distance_str.strip():
                            to_dog = dog_ids[j]
                            try:
                                distance = float(distance_str)
                                if from_dog not in self.distance_matrix:
                                    self.distance_matrix[from_dog] = {}
                                self.distance_matrix[from_dog][to_dog] = distance
                                distances_loaded += 1
                            except (ValueError, TypeError):
                                pass
            
            print(f"✅ Loaded distance matrix with {len(dog_ids)} dogs")
            print(f"✅ Loaded {distances_loaded} distance values")
            if dog_ids:
                sample_id = dog_ids[0]
                if sample_id.endswith('x'):
                    print(f"   ℹ️  Note: Matrix IDs have 'x' suffix (e.g., '{sample_id}')")
                    
        except Exception as e:
            print(f"❌ Error loading distance matrix: {e}")
            print(f"   Sheet ID: {self.DISTANCE_MATRIX_SHEET_ID}")
            print(f"   Tab name: {self.MATRIX_TAB}")
            print("   💡 Try running the sheet tab finder script to verify tab names")
            self.distance_matrix = {}

    def load_dog_coordinates(self):
        """Load dog coordinates from Map sheet for haversine fallback"""
        try:
            print("📍 Loading dog coordinates for distance fallback...")
            sheet = self.gc.open_by_key(self.MAP_SHEET_ID).worksheet(self.MAP_TAB)
            all_values = sheet.get_all_values()
            
            # Column indices (based on user's column explanation)
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

    def haversine_distance(self, dog_id1, dog_id2):
        """Calculate distance between two dogs using haversine formula"""
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
        distance = R * c
        
        # Add a factor to approximate driving distance from straight-line distance
        driving_factor = 1.3
        return distance * driving_factor

    def get_distance_with_fallback(self, dog_id1, dog_id2):
        """Get distance between dogs with haversine fallback"""
        # First, try the distance matrix
        distance = self.safe_get_distance(dog_id1, dog_id2)
        if distance < float('inf'):
            return distance
        
        # Fallback to haversine calculation
        haversine_dist = self.haversine_distance(dog_id1, dog_id2)
        if haversine_dist < float('inf'):
            # Track haversine usage
            if not hasattr(self, 'haversine_fallback_count'):
                self.haversine_fallback_count = 0
            self.haversine_fallback_count += 1
            return haversine_dist
        
        # Last resort
        return self.EXCLUSION_DISTANCE

    def load_dog_assignments(self):
        """Load dog assignments with correct column mappings and dynamic capacity"""
        try:
            print("📊 Loading assignments (dynamic capacity based on route density)...")
            sheet = self.gc.open_by_key(self.MAP_SHEET_ID).worksheet(self.MAP_TAB)
            all_values = sheet.get_all_values()
            
            if not all_values:
                print("❌ No data found in assignments sheet")
                return
            
            # Column indices based on user's explanation
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
                        
                        # Track name-to-ID mapping
                        self.dog_name_to_id[dog_name] = dog_id
                        self.dog_id_to_name[dog_id] = dog_name
                        
                        # Count callouts needing assignment
                        if callout and not combined:
                            callouts_found += 1
                        
                        # Count driver assignments and find drivers
                        if combined and ':' in combined:
                            driver_name = combined.split(':')[0]
                            if driver_name not in ['Field', 'Parking']:  # Skip non-driver entries
                                drivers_found.add(driver_name)
                                self.driver_assignment_counts[driver_name] += 1
            
            # Set active drivers (all drivers found in assignments)
            self.active_drivers = drivers_found
            
            print(f"✅ Found {len(drivers_found)} drivers from assignments")
            print(f"✅ Loaded {len(self.dog_assignments)} dog assignments")
            print(f"✅ Found {callouts_found} callouts needing assignment")
            print(f"✅ Capacity will be calculated dynamically based on route density:")
            print(f"   - Dense routes (avg < {self.DENSE_ROUTE_THRESHOLD}mi): capacity 12")
            print(f"   - Standard routes: capacity 8")
            
        except Exception as e:
            print(f"❌ Error loading dog assignments: {e}")
            raise

    def calculate_driver_density(self, driver):
        """Calculate route density for a driver and return capacity info"""
        # Get all dogs currently assigned to this driver
        driver_dogs = []
        for assignment in self.dog_assignments:
            if isinstance(assignment, dict) and assignment.get('combined', '').startswith(f"{driver}:"):
                driver_dogs.append(assignment)
        
        if len(driver_dogs) < 2:
            # Not enough dogs to calculate density, use default capacity
            return {
                'dog_count': len(driver_dogs),
                'avg_distance': 0.0,
                'capacity': 8,
                'is_dense': False
            }
        
        # Calculate all pairwise distances
        distances = []
        for i in range(len(driver_dogs)):
            for j in range(i + 1, len(driver_dogs)):
                dog1_id = driver_dogs[i].get('dog_id', '')
                dog2_id = driver_dogs[j].get('dog_id', '')
                if dog1_id and dog2_id:
                    dist = self.get_distance_with_fallback(dog1_id, dog2_id)
                    if dist < float('inf'):
                        distances.append(dist)
        
        if not distances:
            avg_distance = 0.0
        else:
            avg_distance = sum(distances) / len(distances)
        
        # Dense route if average distance is less than threshold
        is_dense = avg_distance < self.DENSE_ROUTE_THRESHOLD
        capacity = 12 if is_dense else 8
        
        return {
            'dog_count': len(driver_dogs),
            'avg_distance': avg_distance,
            'capacity': capacity,
            'is_dense': is_dense
        }

    def check_driver_capacity_for_groups(self, driver, needed_groups):
        """Check if driver has capacity for the specified groups with dynamic capacity"""
        if driver not in self.active_drivers:
            return False
        
        # Get current group usage
        current_groups = defaultdict(int)
        for assignment in self.dog_assignments:
            if isinstance(assignment, dict) and assignment.get('combined', '').startswith(f"{driver}:"):
                combined = assignment['combined']
                if ':' in combined:
                    groups_part = combined.split(':', 1)[1]
                    # Parse groups from the assignment
                    groups = self.parse_dog_groups_from_callout(groups_part)
                    for group in groups:
                        current_groups[group] += 1
        
        # Get dynamic capacity based on route density
        capacity_info = self.calculate_driver_density(driver)
        capacity = capacity_info['capacity']
        
        # Check if driver has room for all needed groups
        for group in needed_groups:
            if current_groups[group] >= capacity:
                return False
        
        return True

    def parse_dog_groups_from_callout(self, callout):
        """Parse groups from callout for capacity checking - but preserve original format"""
        if not callout:
            return []
        
        # Remove leading ':' for parsing
        callout_clean = callout.lstrip(':')
        groups = []
        
        # Parse numbers 1, 2, 3 from the callout string
        for char in callout_clean:
            if char in ['1', '2', '3']:
                group_num = int(char)
                if group_num not in groups:
                    groups.append(group_num)
        
        return sorted(groups)

    def find_drivers_with_group_compatibility(self, needed_groups):
        """Find drivers that can handle the specified groups"""
        compatible_drivers = []
        
        for driver in self.active_drivers:
            if self.check_driver_capacity_for_groups(driver, needed_groups):
                compatible_drivers.append(driver)
        
        # Sort by current capacity (dense route drivers first)
        def driver_priority(driver):
            capacity_info = self.calculate_driver_density(driver)
            return (-capacity_info['capacity'], capacity_info['dog_count'])
        
        compatible_drivers.sort(key=driver_priority)
        return compatible_drivers

    def safe_get_distance(self, dog_id1, dog_id2):
        """Safely get distance between two dogs from matrix"""
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

    def reassign_dogs_closest_first_strategy(self):
        """Assign all callouts to closest driver - FIXED to preserve original group formats"""
        print("\n🎯 Starting CLOSEST-FIRST assignment strategy")
        
        # Build current state
        current_assignments = defaultdict(list)
        for assignment in self.dog_assignments:
            if isinstance(assignment, dict) and assignment.get('combined', '').strip():
                combined = assignment['combined']
                if ':' in combined:
                    driver = combined.split(':')[0]
                    current_assignments[driver].append(assignment)
        
        # Find all callouts that need assignment
        callouts = []
        for assignment in self.dog_assignments:
            if isinstance(assignment, dict):
                callout = assignment.get('callout', '').strip()
                combined = assignment.get('combined', '').strip()
                
                # Need assignment if has callout but no driver assigned
                if callout and not combined:
                    callouts.append(assignment)
        
        print(f"📊 Found {len(callouts)} callout dogs to assign")
        
        # Initialize tracking
        if not hasattr(self, 'optimization_swaps'):
            self.optimization_swaps = []
        
        assignments_made = 0
        
        for callout in callouts:
            dog_name = callout.get('dog_name', 'Unknown')
            dog_id = callout.get('dog_id', '')
            original_callout = callout.get('callout', '').strip()  # PRESERVE ORIGINAL FORMAT
            
            if not original_callout:
                print(f"❌ {dog_name}: No callout specified")
                continue
            
            # Parse groups from callout (for capacity checking only)
            groups = self.parse_dog_groups_from_callout(original_callout)
            if not groups:
                print(f"❌ {dog_name}: No groups found in callout '{original_callout}'")
                continue
            
            # Find compatible drivers
            compatible_drivers = self.find_drivers_with_group_compatibility(groups)
            if not compatible_drivers:
                print(f"❌ {dog_name}: No driver can handle groups {groups}")
                continue
            
            # Find closest driver
            best_driver = None
            best_distance = float('inf')
            
            for driver in compatible_drivers:
                # Get driver's current dogs
                driver_dogs = current_assignments[driver]
                
                if not driver_dogs:
                    # Driver has no dogs yet - use default distance or assign anyway
                    distance = 0.5  # Default distance for new drivers
                else:
                    # Calculate average distance to driver's existing dogs
                    distances = []
                    for existing_dog in driver_dogs:
                        existing_dog_id = existing_dog.get('dog_id', '')
                        if existing_dog_id and dog_id:
                            dist = self.get_distance_with_fallback(dog_id, existing_dog_id)
                            if dist < float('inf'):
                                distances.append(dist)
                    
                    if distances:
                        distance = sum(distances) / len(distances)
                    else:
                        distance = 0.5  # Default if can't calculate
                
                if distance < best_distance:
                    best_distance = distance
                    best_driver = driver
            
            if best_driver:
                # CRITICAL FIX: Preserve exact original callout format
                # DON'T reconstruct groups - use original callout string after ':'
                if original_callout.startswith(':'):
                    group_part = original_callout  # Keep the whole thing including ':'
                else:
                    group_part = ':' + original_callout  # Add ':' if missing
                
                new_assignment = f"{best_driver}{group_part}"
                
                # Update the dog assignment
                callout['combined'] = new_assignment
                current_assignments[best_driver].append(callout)
                
                # Check capacity after assignment
                capacity_info = self.calculate_driver_density(best_driver)
                capacity = capacity_info['capacity']
                
                print(f"   ✅ {dog_name} → {best_driver} ({best_distance:.2f}mi) "
                      f"for callout '{original_callout}' [cap {capacity}]")
                assignments_made += 1
            else:
                print(f"❌ {dog_name}: No available driver found")
        
        print(f"\n📊 Assignment Summary: {assignments_made} dogs assigned")
        return assignments_made

    def optimize_existing_assignments_with_swaps(self):
        """Optimize existing assignments by swapping dogs between drivers - VERBOSE VERSION"""
        print("\n🔄 Starting swap optimization...")
        
        if not hasattr(self, 'optimization_swaps'):
            self.optimization_swaps = []
        
        swap_count = 0
        max_swaps = 20  # Reduced to prevent excessive swapping
        swapped_pairs = set()  # Track swapped pairs to prevent duplicates
        
        # Get all current assignments
        driver_assignments = defaultdict(list)
        for assignment in self.dog_assignments:
            if isinstance(assignment, dict) and assignment.get('combined', '').strip():
                combined = assignment['combined']
                if ':' in combined:
                    driver = combined.split(':')[0]
                    if driver not in ['Field', 'Parking']:
                        driver_assignments[driver].append(assignment)
        
        drivers = list(driver_assignments.keys())
        print(f"🔍 Testing {len(drivers)} drivers for beneficial swaps...")
        
        if len(drivers) < 2:
            print("❌ Need at least 2 drivers to perform swaps")
            return 0
        
        print(f"📊 Current driver assignments:")
        for driver in drivers:
            dog_count = len(driver_assignments[driver])
            capacity_info = self.calculate_driver_density(driver)
            print(f"   {driver}: {dog_count} dogs (capacity: {capacity_info['capacity']})")
        
        swaps_tested = 0
        for i, driver1 in enumerate(drivers):
            for driver2 in drivers[i+1:]:
                if swap_count >= max_swaps:
                    print(f"⏹️  Reached maximum swaps limit ({max_swaps})")
                    break
                
                dogs1 = driver_assignments[driver1]
                dogs2 = driver_assignments[driver2]
                
                print(f"\n🔍 Testing swaps between {driver1} ({len(dogs1)} dogs) ↔ {driver2} ({len(dogs2)} dogs)")
                
                driver_pair_swaps = 0
                for dog1 in dogs1[:]:  # Use slice to avoid modification during iteration
                    for dog2 in dogs2[:]:
                        if swap_count >= max_swaps:
                            break
                        
                        # Create unique pair identifier to prevent duplicate swaps
                        pair_id = tuple(sorted([dog1['dog_id'], dog2['dog_id']]))
                        if pair_id in swapped_pairs:
                            continue
                        
                        swaps_tested += 1
                        
                        # Check if swap would reduce total distance
                        would_improve = self.would_swap_reduce_distance(dog1, dog2, driver1, driver2)
                        
                        if would_improve:
                            # Mark this pair as swapped
                            swapped_pairs.add(pair_id)
                            
                            # Preserve the original group assignments
                            dog1_groups = dog1['combined'].split(':', 1)[1]  # Keep original format
                            dog2_groups = dog2['combined'].split(':', 1)[1]  # Keep original format
                            
                            # Swap the drivers but keep original group formats
                            dog1['combined'] = f"{driver2}:{dog1_groups}"
                            dog2['combined'] = f"{driver1}:{dog2_groups}"
                            
                            self.optimization_swaps.append({
                                'dog1': dog1['dog_name'],
                                'dog2': dog2['dog_name'],
                                'old_driver1': driver1,
                                'old_driver2': driver2,
                                'new_driver1': driver2,
                                'new_driver2': driver1
                            })
                            
                            print(f"   🔄 SWAPPED: {dog1['dog_name']} ({driver1}→{driver2}) ↔ {dog2['dog_name']} ({driver2}→{driver1})")
                            swap_count += 1
                            driver_pair_swaps += 1
                            
                            # Update the driver assignments for next iterations
                            driver_assignments[driver1] = [d for d in driver_assignments[driver1] if d != dog1]
                            driver_assignments[driver1].append(dog2)
                            driver_assignments[driver2] = [d for d in driver_assignments[driver2] if d != dog2]
                            driver_assignments[driver2].append(dog1)
                        
                        # Print progress every 100 tests
                        if swaps_tested % 100 == 0:
                            print(f"   📊 Progress: {swaps_tested} swaps tested, {swap_count} beneficial swaps found")
                
                if driver_pair_swaps == 0:
                    print(f"   ❌ No beneficial swaps found between {driver1} and {driver2}")
            
            if swap_count >= max_swaps:
                break
        
        print(f"\n📊 Swap Optimization Summary:")
        print(f"   - Total swaps tested: {swaps_tested}")
        print(f"   - Beneficial swaps found: {swap_count}")
        print(f"   - Optimization complete: {swap_count} swaps made")
        
        if swap_count == 0:
            print("   ℹ️  No swaps provided sufficient benefit (0.2+ mile savings)")
            print("   ℹ️  This suggests routes are already well-optimized!")
        
        return swap_count

    def would_swap_reduce_distance(self, dog1, dog2, driver1, driver2):
        """Check if swapping two dogs would reduce total distance"""
        dog1_id = dog1.get('dog_id', '')
        dog2_id = dog2.get('dog_id', '')
        
        if not dog1_id or not dog2_id:
            return False
        
        # Get other dogs for each driver
        driver1_others = [d for d in self.dog_assignments 
                         if (isinstance(d, dict) and 
                             d.get('combined', '').startswith(f"{driver1}:") and 
                             d.get('dog_id') != dog1_id)]
        
        driver2_others = [d for d in self.dog_assignments 
                         if (isinstance(d, dict) and 
                             d.get('combined', '').startswith(f"{driver2}:") and 
                             d.get('dog_id') != dog2_id)]
        
        # Calculate current total distance
        current_distance = 0
        for other in driver1_others:
            other_id = other.get('dog_id', '')
            if other_id:
                current_distance += self.get_distance_with_fallback(dog1_id, other_id)
        
        for other in driver2_others:
            other_id = other.get('dog_id', '')
            if other_id:
                current_distance += self.get_distance_with_fallback(dog2_id, other_id)
        
        # Calculate distance after swap
        swap_distance = 0
        for other in driver1_others:
            other_id = other.get('dog_id', '')
            if other_id:
                swap_distance += self.get_distance_with_fallback(dog2_id, other_id)
        
        for other in driver2_others:
            other_id = other.get('dog_id', '')
            if other_id:
                swap_distance += self.get_distance_with_fallback(dog1_id, other_id)
        
        # Check if swap reduces distance by at least threshold
        savings = current_distance - swap_distance
        return savings >= self.PREFERRED_DISTANCE

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
                # Batch update for efficiency with better rate limiting
                for i in range(0, len(updates), 25):  # Even smaller batches
                    batch = updates[i:i+25]
                    for update in batch:
                        sheet.update(update['values'], update['range'])
                        time.sleep(0.5)  # Small delay between each update
                    time.sleep(3)  # Longer delay between batches
                    print(f"📊 Updated batch {i//25 + 1}/{(len(updates)-1)//25 + 1}")
                
                print(f"✅ Updated {len(updates)} assignments in Google Sheets")
            else:
                print("ℹ️  No updates needed")
                
        except Exception as e:
            print(f"❌ Error writing to Google Sheets: {e}")

    def send_slack_notification(self, assignments_made, swaps_made):
        """Send notification to Slack about optimization results"""
        webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
        if not webhook_url:
            print("ℹ️  No Slack webhook configured")
            return
        
        try:
            active_drivers_count = len(self.active_drivers)
            total_dogs = len(self.dog_assignments)
            
            # Calculate utilization
            total_capacity = active_drivers_count * 8 * 3  # Assuming 8 capacity per group, 3 groups
            utilization = (total_dogs / total_capacity * 100) if total_capacity > 0 else 0
            
            message = {
                "text": f"🐕 Dog Assignment Optimization Complete",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Dog Assignment Optimization Results*\n"
                                   f"• New assignments: {assignments_made}\n"
                                   f"• Optimizations: {swaps_made} swaps\n"
                                   f"• Active drivers: {active_drivers_count}\n"
                                   f"• Total dogs: {total_dogs}\n"
                                   f"• Utilization: {utilization:.1f}%"
                        }
                    }
                ]
            }
            
            if hasattr(self, 'haversine_fallback_count') and self.haversine_fallback_count > 0:
                message["blocks"][0]["text"]["text"] += f"\n• Distance fallbacks: {self.haversine_fallback_count}"
            
            response = requests.post(webhook_url, json=message, timeout=10)
            if response.status_code == 200:
                print("✅ Slack notification sent")
            else:
                print(f"⚠️  Slack notification failed: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️  Slack notification error: {e}")

    def analyze_within_group_distances(self):
        """Analyze distances between dogs within same driver AND same group"""
        print("\n🔍 WITHIN-GROUP DISTANCE ANALYSIS")
        print("=" * 50)
        
        max_distance = 0
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
            
            # Get current capacity info
            capacity_info = self.calculate_driver_density(driver)
            capacity = capacity_info['capacity']
            is_dense = capacity_info['is_dense']
            avg_route_distance = capacity_info['avg_distance']
            
            print(f"  Route density: {avg_route_distance:.2f}mi avg → Capacity: {capacity} "
                  f"{'(DENSE)' if is_dense else '(STANDARD)'}")
            
            # Analyze each group
            for group_num in sorted(groups.keys()):
                dogs = groups[group_num]
                if len(dogs) < 2:
                    print(f"  Group {group_num}: {len(dogs)} dog{'s' if len(dogs) != 1 else ''} "
                          f"(no distances to calculate)")
                    continue
                
                distances = []
                for i in range(len(dogs)):
                    for j in range(i+1, len(dogs)):
                        dist = self.get_distance_with_fallback(
                            dogs[i].get('dog_id', ''), 
                            dogs[j].get('dog_id', '')
                        )
                        if dist < float('inf'):
                            distances.append(dist)
                            if dist > max_distance:
                                max_distance = dist
                                max_info = (driver, group_num, dogs[i].get('dog_name', ''), 
                                          dogs[j].get('dog_name', ''))
                
                if distances:
                    avg = sum(distances) / len(distances)
                    min_dist = min(distances)
                    max_dist = max(distances)
                    
                    # Visual indicator
                    if avg < 0.3:
                        indicator = "✅"
                    elif avg < 0.5:
                        indicator = "⚠️"
                    else:
                        indicator = "🚨"
                    
                    print(f"  Group {group_num}: {len(dogs)} dogs")
                    print(f"    Average distance: {avg:.2f} miles {indicator}")
                    print(f"    Min distance: {min_dist:.2f} miles")
                    print(f"    Max distance: {max_dist:.2f} miles")
                    
                    # Store for summary
                    group_stats.append({
                        'driver': driver,
                        'group': group_num,
                        'dog_count': len(dogs),
                        'avg_distance': avg,
                        'min_distance': min_dist,
                        'max_distance': max_dist
                    })
                    
                    print(f"    Dogs in group:")
                    for dog in dogs:
                        dog_name = dog.get('dog_name', 'Unknown')
                        dog_id = dog.get('dog_id', 'No ID')
                        print(f"      - {dog_name} (ID: {dog_id})")
        
        # Summary statistics
        if group_stats:
            print(f"\n📊 SUMMARY STATISTICS")
            print("=" * 50)
            
            all_avgs = [stat['avg_distance'] for stat in group_stats]
            overall_avg = sum(all_avgs) / len(all_avgs)
            
            print(f"🎯 OVERALL AVERAGE DISTANCE: {overall_avg:.2f} miles")
            
            # Group-specific averages
            group_averages = defaultdict(list)
            for stat in group_stats:
                group_averages[stat['group']].append(stat['avg_distance'])
            
            print(f"\n🎯 AVERAGE DISTANCES BY GROUP:")
            for group_num in sorted(group_averages.keys()):
                group_avg = sum(group_averages[group_num]) / len(group_averages[group_num])
                print(f"  Group {group_num}: {group_avg:.2f} miles average")
            
            if max_info:
                print(f"\n🚨 MAXIMUM DISTANCE BETWEEN ANY TWO DOGS:")
                print(f"  Driver: {max_info[0]}")
                print(f"  Group: {max_info[1]}")
                print(f"  Dogs: {max_info[2]} ↔ {max_info[3]}")
                print(f"  Distance: {max_distance:.2f} miles")
            
            # Problem groups
            problem_groups = [stat for stat in group_stats if stat['avg_distance'] > 0.5]
            if problem_groups:
                print(f"\n⚠️  GROUPS NEEDING ATTENTION (avg > 0.5mi):")
                for stat in problem_groups:
                    print(f"  {stat['driver']} Group {stat['group']}: "
                          f"{stat['avg_distance']:.2f}mi avg, {stat['dog_count']} dogs")
        else:
            print("\n📊 No group statistics available")

    def debug_distance_issues(self):
        """Debug distance matrix issues and ID mismatches"""
        try:
            print("\n🔍 DEBUGGING DISTANCE MATRIX ISSUES")
            print("=" * 40)
            
            # Get distance matrix dog IDs
            matrix_dog_ids = set(self.distance_matrix.keys())
            print(f"📊 Distance Matrix Info:")
            print(f"   Total dog IDs: {len(matrix_dog_ids)}")
            print(f"   Sample IDs: {list(matrix_dog_ids)[:10]}")
            
            # Get assignment sheet dog IDs
            assignment_dog_ids = set()
            callout_dog_ids = set()
            
            for assignment in self.dog_assignments:
                if isinstance(assignment, dict):
                    dog_id = assignment.get('dog_id', '').strip()
                    if dog_id:
                        assignment_dog_ids.add(dog_id)
                        # Check if this dog has a callout
                        callout = assignment.get('callout', '').strip()
                        combined = assignment.get('combined', '').strip()
                        if callout and not combined:  # Has callout but no assignment
                            callout_dog_ids.add(dog_id)
            
            print(f"\n📊 Assignment Sheet Info:")
            print(f"   Total dog IDs: {len(assignment_dog_ids)}")
            print(f"   Callout dog IDs: {len(callout_dog_ids)}")
            print(f"   Sample IDs: {list(assignment_dog_ids)[:10]}")
            
            # Find mismatches
            missing_from_matrix = assignment_dog_ids - matrix_dog_ids
            missing_from_assignments = matrix_dog_ids - assignment_dog_ids
            callouts_missing_from_matrix = callout_dog_ids - matrix_dog_ids
            
            print(f"\n🔍 ID MISMATCH ANALYSIS:")
            print(f"   Dogs in assignments but NOT in matrix: {len(missing_from_matrix)}")
            print(f"   Dogs in matrix but NOT in assignments: {len(missing_from_assignments)}")
            print(f"   Callout dogs missing from matrix: {len(callouts_missing_from_matrix)}")
            
            if missing_from_matrix:
                print(f"\n❌ MISSING FROM MATRIX (first 10):")
                for dog_id in list(missing_from_matrix)[:10]:
                    print(f"   - {dog_id}")
            
            if callouts_missing_from_matrix:
                print(f"\n🚨 CALLOUT DOGS MISSING FROM MATRIX (first 10):")
                for dog_id in list(callouts_missing_from_matrix)[:10]:
                    print(f"   - {dog_id}")
            
            # Check the specific dogs that got extreme distances
            extreme_assignment_dogs = [
                ('4x', 'Binky', '67.43mi'),
                ('1695x', 'Bauer', '86.10mi'), 
                ('55x', 'Ollie', 'should be close'),
                ('56x', 'Jagger', 'should be close')
            ]
            
            print(f"\n🔍 CHECKING SPECIFIC PROBLEM DOGS:")
            for dog_id, dog_name, note in extreme_assignment_dogs:
                in_matrix = dog_id in matrix_dog_ids
                in_assignments = dog_id in assignment_dog_ids
                print(f"   {dog_name} ({dog_id}): Matrix={in_matrix}, Assignments={in_assignments} - {note}")
                
                if not in_matrix:
                    print(f"     ❌ {dog_name} NOT FOUND in distance matrix - will use haversine!")
            
            # Your matrix rule: ≤3mi = actual, >3mi = 100mi
            print(f"\n🎯 MATRIX VALIDATION (should be ≤3mi or =100mi):")
            sample_distances = []
            count_by_range = {'0-3mi': 0, '100mi': 0, 'other': 0}
            
            # Sample some distances from the matrix
            for i, (from_dog, to_dict) in enumerate(self.distance_matrix.items()):
                if i >= 5:  # Just check first 5 dogs
                    break
                for j, (to_dog, distance) in enumerate(to_dict.items()):
                    if j >= 5:  # Just check first 5 distances per dog
                        break
                    sample_distances.append((from_dog, to_dog, distance))
                    
                    if distance <= 3:
                        count_by_range['0-3mi'] += 1
                    elif distance == 100:
                        count_by_range['100mi'] += 1
                    else:
                        count_by_range['other'] += 1
            
            print(f"   Sample distances from matrix:")
            for from_dog, to_dog, dist in sample_distances[:10]:
                print(f"     {from_dog} → {to_dog}: {dist}mi")
            
            print(f"   Distance distribution in matrix:")
            print(f"     0-3 miles: {count_by_range['0-3mi']}")
            print(f"     100 miles: {count_by_range['100mi']}")
            print(f"     Other (unexpected): {count_by_range['other']}")
            
            if count_by_range['other'] > 0:
                print(f"     ⚠️  Found unexpected distances (should only be ≤3 or =100)")
            
            print(f"\n💡 KEY INSIGHT:")
            print(f"   If callout dogs show 60-80mi distances, they're NOT in your matrix!")
            print(f"   Your matrix: ≤3mi (real) or 100mi (distant)")
            print(f"   Haversine fallback: 60-80mi (real straight-line)")
            print(f"   → Distances like 67mi prove haversine is being used")
            
            print(f"\n💡 RECOMMENDATIONS:")
            if len(callouts_missing_from_matrix) > 0:
                print(f"   1. ❗ {len(callouts_missing_from_matrix)} callout dogs are missing from distance matrix")
                print(f"      These will use haversine fallback (less accurate)")
            
            if len(missing_from_matrix) > len(callouts_missing_from_matrix):
                print(f"   2. ❗ Many assigned dogs also missing from matrix")
                print(f"      This could explain why haversine is used so much")
                
        except Exception as e:
            print(f"❌ Error in debug: {e}")
            import traceback
            traceback.print_exc()

    def report_haversine_usage(self):
        """Report on haversine fallback usage"""
        if hasattr(self, 'haversine_fallback_count') and self.haversine_fallback_count > 0:
            print(f"\n📍 Haversine Fallback Usage:")
            print(f"   Used {self.haversine_fallback_count} times for distance calculations")
            print(f"   These represent dog pairs not in your distance matrix")

def main():
    """Main execution function"""
    try:
        system = DogReassignmentSystem()
        
        # Check if running in GitHub Actions
        is_github_actions = os.environ.get('GITHUB_ACTIONS') == 'true'
        
        if is_github_actions:
            choice = '1'  # Just run optimization in GitHub Actions
            print("🤖 Running in GitHub Actions - performing optimization")
        else:
            print("\nWhat would you like to do?")
            print("1. Run optimization and reassignments")
            print("2. Analyze within-group distances only")
            print("3. Both (analyze first, then optimize)")
            print("4. Debug distance matrix issues")
            choice = input("Enter choice (1/2/3/4): ").strip()
        
        assignments_made = 0
        swaps_made = 0
        
        if choice in ['2', '3']:
            system.analyze_within_group_distances()
        
        if choice in ['1', '3']:
            assignments_made = system.reassign_dogs_closest_first_strategy()
            swaps_made = system.optimize_existing_assignments_with_swaps()
            system.write_results_to_sheets()
        
        if choice == '4':
            system.debug_distance_issues()
            return  # Exit after debug, don't continue with other operations
        
        system.report_haversine_usage()
        system.send_slack_notification(assignments_made, swaps_made)
        
        print(f"\n🎉 Process completed successfully!")
        print(f"📊 Final Summary:")
        print(f"   - New assignments: {assignments_made}")
        print(f"   - Swaps made: {swaps_made}")
        
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
