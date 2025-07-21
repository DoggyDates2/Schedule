#!/usr/bin/env python3
"""
Dog Assignment Optimization System with Dynamic Capacity
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
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DogReassignmentSystem:
    """Dog reassignment system with dynamic capacity based on route density"""
    
    def __init__(self):
        """Initialize the dog reassignment system"""
        print("🚀 Enhanced Dog Reassignment System - DYNAMIC CAPACITY")
        print("   Dense routes (< 0.5mi avg): 12 dogs per group")
        print("   Standard routes: 8 dogs per group")
        
        # Google Sheets IDs
        self.MATRIX_SHEET_ID = "1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg"
        self.MAP_SHEET_ID = "1-KTOfTKXk_sX7nO7eGmW73JLi8TJBvv5gobK6gyrc7U"
        self.MATRIX_TAB = "Matrix"
        self.MAP_TAB = "Map"
        
        # Initialize Google Sheets connection
        self.setup_google_sheets()
        
        # Initialize data structures
        self.distance_matrix = {}
        self.dog_assignments = []
        self.driver_capacities = {}
        self.dog_name_to_id = {}
        self.dog_id_to_name = {}
        self.driver_assignment_counts = defaultdict(int)
        self.all_capacity_rows = []
        self.dog_coordinates = {}
        
        # System parameters (MUST BE BEFORE LOADING DATA)
        self.PREFERRED_DISTANCE = 0.2
        self.MAX_DISTANCE = 0.5
        self.ABSOLUTE_MAX_DISTANCE = 1.5
        self.CASCADING_MOVE_MAX = 0.7
        self.ADJACENT_GROUP_DISTANCE = 0.1
        self.EXCLUSION_DISTANCE = 200.0
        self.DENSE_ROUTE_THRESHOLD = 0.5  # Routes with avg < 0.5mi are dense
        
        # Load data from sheets
        self.load_distance_matrix()
        self.load_dog_assignments()
        self.load_dog_coordinates()
        
        # Swap optimization parameters
        self.SWAP_THRESHOLD = 0.2
        
        # Tracking variables
        self.active_drivers = set()
        self.driver_densities = {}
        self.optimization_swaps = []
        self.callouts_assigned = 0
        self.total_miles_saved = 0
        self.assignments_made = []
        self.emergency_assignments = []
        
        # Performance optimization
        self._distance_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Rate limiting
        self.last_sheet_update = 0
        self.MIN_UPDATE_INTERVAL = 1.0
        
        # Haversine tracking
        self.haversine_fallback_count = 0
        self.haversine_pairs = set()

    def setup_google_sheets(self):
        """Initialize Google Sheets connection with better error handling"""
        try:
            credentials_json = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
            if not credentials_json:
                raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON environment variable not set")
            
            try:
                credentials_data = json.loads(credentials_json)
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON in GOOGLE_SERVICE_ACCOUNT_JSON: {e}")
                print("   Make sure to export the full JSON, wrapped in single quotes")
                raise
            
            scope = ['https://spreadsheets.google.com/feeds',
                    'https://www.googleapis.com/auth/drive']
            credentials = ServiceAccountCredentials.from_json_keyfile_dict(
                credentials_data, scope)
            self.gc = gspread.authorize(credentials)
            print("✅ Google Sheets connection established")
            
        except Exception as e:
            print(f"❌ Failed to setup Google Sheets: {e}")
            raise

    def load_distance_matrix(self):
        """Load the distance matrix from Google Sheets"""
        try:
            sheet = self.gc.open_by_key(self.MATRIX_SHEET_ID).worksheet(self.MATRIX_TAB)
            all_values = sheet.get_all_values()
            
            # First row contains dog IDs (skip first cell)
            dog_ids = all_values[0][1:]
            
            self.distance_matrix = {}
            
            # Check if IDs have 'x' suffix
            self.matrix_has_x_suffix = any(str(id).endswith('x') for id in dog_ids[:10] if id)
            
            for i, row in enumerate(all_values[1:]):
                from_dog_id = str(row[0]).strip()
                if not from_dog_id:
                    continue
                
                # Store with original format from matrix
                self.distance_matrix[from_dog_id] = {}
                
                for j, distance_str in enumerate(row[1:]):
                    if j < len(dog_ids):
                        to_dog_id = str(dog_ids[j]).strip()
                        try:
                            distance = float(distance_str) if distance_str else self.EXCLUSION_DISTANCE
                        except (ValueError, TypeError):
                            distance = self.EXCLUSION_DISTANCE
                        
                        if distance < 0 or distance > 100:
                            distance = self.EXCLUSION_DISTANCE
                        
                        self.distance_matrix[from_dog_id][to_dog_id] = distance
            
            print(f"✅ Loaded distance matrix with {len(self.distance_matrix)} dogs")
            if self.matrix_has_x_suffix:
                print("   ℹ️  Note: Matrix IDs have 'x' suffix (e.g., '1x', '2x')")
            
        except Exception as e:
            print(f"❌ Error loading distance matrix: {e}")
            raise

    def load_dog_assignments(self):
        """Load dog assignments - with dynamic capacity based on density"""
        try:
            sheet = self.gc.open_by_key(self.MAP_SHEET_ID).worksheet(self.MAP_TAB)
            all_values = sheet.get_all_values()
            self.assignment_data = all_values
            
            headers = all_values[0] if all_values else []
            self.headers = headers
            
            # Column mappings
            address_idx = 0        # Column A
            dog_name_idx = 1       # Column B
            combined_idx = 7       # Column H
            dog_id_idx = 9         # Column J
            callout_idx = 10       # Column K - Groups needed
            num_dogs_idx = 5       # Column F
            
            # Initialize tracking
            self.dog_assignments = []
            self.driver_capacities = {}
            self.driver_assignment_counts = defaultdict(int)
            
            print(f"📊 Loading assignments (dynamic capacity based on route density)...")
            
            # First, find all unique drivers from Combined assignments
            all_drivers = set()
            
            # Parse dog assignments
            for i, row in enumerate(all_values[1:], start=2):
                if len(row) <= dog_name_idx:
                    continue
                
                dog_name = row[dog_name_idx] if dog_name_idx < len(row) else ""
                if not dog_name or dog_name.strip() == "":
                    continue
                
                # Get combined assignment to find drivers
                combined = row[combined_idx] if combined_idx < len(row) else ""
                if combined and ':' in combined:
                    driver_part = combined.split(':', 1)[0]
                    if driver_part and driver_part.lower() not in ['field', 'parking']:
                        all_drivers.add(driver_part)
                        self.driver_assignment_counts[driver_part] += 1
                
                # Get callout (groups needed)
                callout = row[callout_idx] if callout_idx < len(row) else ""
                
                # Parse groups from callout field
                needed_groups = []
                if callout and callout.strip():
                    callout_str = callout.strip()
                    if callout_str.startswith(':'):
                        callout_str = callout_str[1:]
                    
                    for char in callout_str:
                        if char.isdigit() and char in ['1', '2', '3']:
                            group_num = int(char)
                            if group_num not in needed_groups:
                                needed_groups.append(group_num)
                
                assignment = {
                    'dog_name': dog_name.strip(),
                    'dog_id': row[dog_id_idx].strip() if dog_id_idx < len(row) else "",
                    'combined': combined,
                    'callout': callout,
                    'needed_groups': sorted(needed_groups),
                    'row_number': i,
                    'number_of_dogs': row[num_dogs_idx] if num_dogs_idx < len(row) else "1",
                    'address': row[address_idx] if address_idx < len(row) else ""
                }
                
                self.dog_assignments.append(assignment)
                
                if assignment['dog_name'] and assignment['dog_id']:
                    self.dog_name_to_id[assignment['dog_name']] = assignment['dog_id']
                    self.dog_id_to_name[assignment['dog_id']] = assignment['dog_name']
            
            # Initialize all drivers with dummy capacity (will be calculated dynamically)
            print(f"✅ Found {len(all_drivers)} drivers from assignments")
            for driver in all_drivers:
                self.driver_capacities[driver] = {
                    'group1': 8,  # Default, but will use dynamic calculation
                    'group2': 8,
                    'group3': 8,
                    'total': 24
                }
            
            # All drivers are "active"
            self.active_drivers = all_drivers
            
            print(f"✅ Loaded {len(self.dog_assignments)} dog assignments")
            print(f"✅ Capacity will be calculated dynamically based on route density:")
            print(f"   - Dense routes (avg < {self.DENSE_ROUTE_THRESHOLD}mi): capacity 12")
            print(f"   - Standard routes: capacity 8")
            
            # Count callouts
            callout_count = sum(1 for d in self.dog_assignments 
                               if d.get('needed_groups') and
                               not d.get('combined', '').strip())
            print(f"✅ Found {callout_count} callouts needing assignment")
            
            # Debug: Show some sample callouts
            print("\n📋 Sample callouts:")
            sample_count = 0
            for d in self.dog_assignments:
                if (d.get('needed_groups') and 
                    not d.get('combined', '').strip()):
                    print(f"   - {d['dog_name']}: needs groups {d['needed_groups']} (callout: {d['callout']})")
                    sample_count += 1
                    if sample_count >= 5:
                        break
            
            # Run data integrity check
            self.validate_data_integrity()
            
        except Exception as e:
            print(f"❌ Error loading dog assignments: {e}")
            raise

    def load_dog_coordinates(self):
        """Load dog coordinates from the Map sheet for haversine fallback"""
        print("📍 Loading dog coordinates for distance fallback...")
        
        # Columns D and E are Latitude and Longitude
        lat_idx = 3  # Column D (0-based index)
        lng_idx = 4  # Column E (0-based index)
        dog_id_idx = 9  # Column J
        
        self.dog_coordinates = {}
        
        # Parse coordinates from columns D and E
        coord_count = 0
        missing_coord_count = 0
        
        for row in self.assignment_data[1:]:
            if len(row) > max(lat_idx, lng_idx, dog_id_idx):
                try:
                    dog_id = row[dog_id_idx].strip()
                    lat = float(row[lat_idx]) if row[lat_idx] else None
                    lng = float(row[lng_idx]) if row[lng_idx] else None
                    
                    if dog_id and lat and lng:
                        self.dog_coordinates[dog_id] = (lat, lng)
                        coord_count += 1
                    elif dog_id:
                        missing_coord_count += 1
                except (ValueError, IndexError):
                    continue
        
        print(f"✅ Loaded coordinates for {coord_count} dogs")
        if missing_coord_count > 0:
            print(f"⚠️  {missing_coord_count} dogs missing coordinates in columns D/E")

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points on Earth"""
        # Radius of Earth in miles
        R = 3959.0
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Differences
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Haversine formula
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        # Add a factor to approximate driving distance from straight-line distance
        driving_factor = 1.3
        
        return distance * driving_factor

    def get_distance_with_fallback(self, dog1_id, dog2_id):
        """Get distance with haversine fallback for missing matrix entries"""
        if dog1_id == dog2_id:
            return 0.0
        
        # Try matrix first
        if dog1_id in self.distance_matrix and dog2_id in self.distance_matrix[dog1_id]:
            distance = self.distance_matrix[dog1_id].get(dog2_id, None)
            if distance is not None and 0 <= distance < self.EXCLUSION_DISTANCE:
                return float(distance)
        
        # Fallback to haversine if available
        if hasattr(self, 'dog_coordinates'):
            if dog1_id in self.dog_coordinates and dog2_id in self.dog_coordinates:
                lat1, lon1 = self.dog_coordinates[dog1_id]
                lat2, lon2 = self.dog_coordinates[dog2_id]
                
                haversine_dist = self.haversine_distance(lat1, lon1, lat2, lon2)
                
                # Track fallback usage
                if not hasattr(self, 'haversine_fallback_count'):
                    self.haversine_fallback_count = 0
                    self.haversine_pairs = set()
                
                self.haversine_fallback_count += 1
                self.haversine_pairs.add((dog1_id, dog2_id))
                
                return haversine_dist
        
        # No data available
        return self.EXCLUSION_DISTANCE

    def get_distance(self, dog1_id, dog2_id):
        """Get distance between two dogs from matrix with haversine fallback"""
        return self.get_distance_with_fallback(dog1_id, dog2_id)

    def safe_get_distance(self, dog1_id, dog2_id):
        """Safely get distance between two dogs with fallback"""
        try:
            dist = self.get_distance(dog1_id, dog2_id)
            if dist is None or dist >= self.EXCLUSION_DISTANCE:
                return float('inf')
            return dist
        except (KeyError, TypeError, AttributeError):
            return float('inf')

    def calculate_driver_density(self, driver):
        """Calculate the average distance between dogs for a driver"""
        driver_dogs = []
        for assignment in self.dog_assignments:
            if isinstance(assignment, dict) and assignment.get('combined', '').startswith(f"{driver}:"):
                driver_dogs.append(assignment.get('dog_id'))
        
        if len(driver_dogs) < 2:
            return None  # Can't calculate density with < 2 dogs
        
        # Calculate average distance between all pairs
        distances = []
        for i in range(len(driver_dogs)):
            for j in range(i + 1, len(driver_dogs)):
                dist = self.safe_get_distance(driver_dogs[i], driver_dogs[j])
                if dist < self.EXCLUSION_DISTANCE:
                    distances.append(dist)
        
        if distances:
            return sum(distances) / len(distances)
        return None

    def get_dynamic_capacity(self, driver):
        """Get capacity based on route density - dense routes can handle 12, others 8"""
        # Calculate current route density
        avg_distance = self.calculate_driver_density(driver)
        
        if avg_distance is None:
            # New driver or only 1 dog - start with standard capacity
            return 8
        
        # Dense routes (avg < 0.5 miles between dogs) can handle 12
        if avg_distance < self.DENSE_ROUTE_THRESHOLD:
            return 12
        else:
            return 8

    def check_driver_capacity_for_groups(self, driver, groups, current_dogs=None):
        """Check if driver has capacity - dynamically based on route density"""
        if current_dogs is None:
            current_dogs = self.get_driver_current_dogs(driver)
        
        # Count current usage per group
        usage = {'1': 0, '2': 0, '3': 0}
        for dog in current_dogs:
            dog_groups = self.parse_dog_groups(dog)
            for g in dog_groups:
                group_str = str(g)
                if group_str in usage:
                    usage[group_str] += 1
        
        # Get dynamic capacity based on route density
        max_capacity = self.get_dynamic_capacity(driver)
        
        # Check if adding new dog would exceed capacity
        for g in groups:
            group_str = str(g)
            if usage.get(group_str, 0) >= max_capacity:
                return False
        
        return True

    def parse_dog_groups(self, assignment):
        """Parse groups from assignment - they're in needed_groups already"""
        return assignment.get('needed_groups', [])

    def get_driver_current_dogs(self, driver):
        """Get current dogs assigned to a driver"""
        current_dogs = []
        
        for assignment in self.dog_assignments:
            if isinstance(assignment, dict):
                combined = assignment.get('combined', '')
                if combined.startswith(f"{driver}:"):
                    current_dogs.append(assignment)
        
        return current_dogs

    def find_drivers_with_group_compatibility(self, needed_groups):
        """Find drivers that can handle the specified groups - prioritizes dense routes"""
        compatible_drivers = []
        dense_route_drivers = []
        
        # Check all drivers
        for driver in self.driver_capacities:
            # Skip fake drivers
            if driver.lower() in ['field', 'parking']:
                continue
            
            # Check if driver has room
            current_dogs = self.get_driver_current_dogs(driver)
            if self.check_driver_capacity_for_groups(driver, needed_groups, current_dogs):
                # Check if this is a dense route
                if self.get_dynamic_capacity(driver) == 12:
                    dense_route_drivers.append(driver)
                else:
                    compatible_drivers.append(driver)
        
        # Return dense route drivers first (they can handle more dogs efficiently)
        return dense_route_drivers + compatible_drivers

    def analyze_within_group_distances(self):
        """Analyze distances between dogs within same driver AND same group"""
        print("\n" + "="*80)
        print("🔍 WITHIN-GROUP DISTANCE ANALYSIS")
        print("="*80)
        print("Shows how close dogs are within each driver's groups")
        print("Target: Average < 0.3 miles for efficient routes")
        print("Dense routes (< 0.5 mi avg) can handle 12 dogs per group")
        print("="*80)
        
        max_distance_overall = 0
        max_distance_info = None
        problem_groups = []
        all_group_stats = []
        
        for driver in sorted(self.active_drivers):
            driver_dogs = self.get_driver_current_dogs(driver)
            
            if not driver_dogs:
                continue
            
            # Get current capacity for this driver
            current_capacity = self.get_dynamic_capacity(driver)
            density_status = "DENSE (12)" if current_capacity == 12 else "STANDARD (8)"
            
            # Group dogs by their group numbers
            driver_groups = defaultdict(list)
            for dog in driver_dogs:
                dog_groups = self.parse_dog_groups(dog)
                for g in dog_groups:
                    driver_groups[g].append(dog)
            
            if not driver_groups:
                continue
            
            print(f"\n{'='*60}")
            print(f"DRIVER: {driver} - Route: {density_status}")
            print(f"{'='*60}")
            
            driver_stats = {'driver': driver, 'groups': {}, 'capacity': current_capacity}
            
            for group_num in sorted(driver_groups.keys()):
                dogs = driver_groups[group_num]
                
                if len(dogs) < 2:
                    print(f"\n  Group {group_num}: {len(dogs)}/{current_capacity} dogs - no distances to calculate")
                    continue
                
                # Calculate all pairwise distances
                distances = []
                distance_pairs = []
                
                for i in range(len(dogs)):
                    for j in range(i + 1, len(dogs)):
                        dist = self.safe_get_distance(dogs[i]['dog_id'], dogs[j]['dog_id'])
                        if dist < float('inf'):
                            distances.append(dist)
                            distance_pairs.append({
                                'dog1': dogs[i],
                                'dog2': dogs[j],
                                'distance': dist
                            })
                            
                            # Track maximum distance
                            if dist > max_distance_overall:
                                max_distance_overall = dist
                                max_distance_info = {
                                    'driver': driver,
                                    'group': group_num,
                                    'dog1': dogs[i],
                                    'dog2': dogs[j],
                                    'distance': dist
                                }
                
                if distances:
                    avg_dist = statistics.mean(distances)
                    min_dist = min(distances)
                    max_dist = max(distances)
                    
                    driver_stats['groups'][group_num] = {
                        'dog_count': len(dogs),
                        'avg_distance': avg_dist,
                        'min_distance': min_dist,
                        'max_distance': max_dist,
                        'distances': distances
                    }
                    
                    # Print results
                    print(f"\n  Group {group_num}: {len(dogs)}/{current_capacity} dogs")
                    print(f"    Average distance: {avg_dist:.2f} miles", end="")
                    if avg_dist < 0.3:
                        print(" ✅")
                    elif avg_dist < 0.5:
                        print(" ⚠️")
                    else:
                        print(" 🚨")
                        problem_groups.append({
                            'driver': driver,
                            'group': group_num,
                            'avg_distance': avg_dist,
                            'dog_count': len(dogs)
                        })
                    
                    print(f"    Min distance: {min_dist:.2f} miles")
                    print(f"    Max distance: {max_dist:.2f} miles")
                    
                    # Show dogs in group
                    print(f"    Dogs in group:")
                    for dog in dogs[:5]:  # Show first 5
                        print(f"      - {dog['dog_name']} (ID: {dog['dog_id']})")
                    if len(dogs) > 5:
                        print(f"      ... and {len(dogs) - 5} more")
            
            all_group_stats.append(driver_stats)
        
        # Print summary
        print("\n" + "="*80)
        print("📊 SUMMARY STATISTICS")
        print("="*80)
        
        # Maximum distance
        print(f"\n🚨 MAXIMUM DISTANCE BETWEEN ANY TWO DOGS:")
        if max_distance_info:
            print(f"  Driver: {max_distance_info['driver']}")
            print(f"  Group: {max_distance_info['group']}")
            print(f"  Dogs: {max_distance_info['dog1']['dog_name']} ↔ {max_distance_info['dog2']['dog_name']}")
            print(f"  Distance: {max_distance_info['distance']:.2f} miles")
        
        return all_group_stats, max_distance_info

    def build_initial_assignments_state(self):
        """Build current state of assignments for optimization"""
        current_assignments = []
        
        for assignment in self.dog_assignments:
            if not isinstance(assignment, dict):
                continue
                
            dog_id = assignment.get('dog_id', '')
            dog_name = assignment.get('dog_name', '')
            combined = assignment.get('combined', '')
            
            if combined and ':' in combined:
                driver = combined.split(':', 1)[0]
                groups = self.parse_dog_groups(assignment)
                
                current_assignments.append({
                    'dog_id': dog_id,
                    'dog_name': dog_name,
                    'driver': driver,
                    'combined': combined,
                    'needed_groups': groups,
                    'groups': groups,
                    'callout': assignment.get('callout', '').strip() != ''
                })
        
        return current_assignments

    def verify_capacity_constraints(self, assignments):
        """Verify no capacity constraints are violated - uses dynamic capacity"""
        violations = []
        
        # Count dogs per driver per group
        driver_group_counts = defaultdict(lambda: defaultdict(int))
        
        for assignment in assignments:
            driver = assignment.get('driver', '')
            groups = assignment.get('needed_groups', [])
            
            if driver and driver in self.driver_capacities:
                for group in groups:
                    driver_group_counts[driver][str(group)] += 1
        
        # Check against dynamic capacities
        for driver, group_counts in driver_group_counts.items():
            max_capacity = self.get_dynamic_capacity(driver)
            
            for group_num, count in group_counts.items():
                if count > max_capacity:
                    violations.append({
                        'driver': driver,
                        'group': group_num,
                        'count': count,
                        'max': max_capacity,
                        'excess': count - max_capacity
                    })
        
        return violations

    def validate_data_integrity(self):
        """Validate all data is consistent"""
        issues = []
        
        # Check 1: Find dogs missing from matrix
        matrix_dogs = set(self.distance_matrix.keys())
        assignment_dogs = {a.get('dog_id') for a in self.dog_assignments if a.get('dog_id')}
        
        missing_from_matrix = assignment_dogs - matrix_dogs
        if missing_from_matrix:
            # Check how many have coordinate fallback
            have_coords = sum(1 for dog in missing_from_matrix if dog in self.dog_coordinates)
            
            print(f"\n📍 Dogs not in distance matrix: {len(missing_from_matrix)}")
            print(f"   - {have_coords} have coordinates (will use haversine fallback)")
            print(f"   - {len(missing_from_matrix) - have_coords} have NO distance data")
            
            if len(missing_from_matrix) - have_coords > 0:
                no_data = [d for d in missing_from_matrix if d not in self.dog_coordinates]
                issues.append(f"Dogs with NO distance data: {no_data[:5]}")
        
        # Report on data coverage
        total_dogs = len(assignment_dogs)
        if total_dogs > 0:
            in_matrix = len(assignment_dogs & matrix_dogs)
            have_coords = len([d for d in assignment_dogs if d in self.dog_coordinates])
            
            print(f"\n📊 Distance Data Coverage:")
            print(f"   Total dogs: {total_dogs}")
            print(f"   In distance matrix: {in_matrix} ({in_matrix/total_dogs*100:.1f}%)")
            print(f"   Have coordinates: {have_coords} ({have_coords/total_dogs*100:.1f}%)")
        
        return len(issues) == 0

    def calculate_route_densities(self):
        """Calculate how spread out each driver's route is"""
        print("\n📍 Calculating route densities...")
        
        for driver in self.driver_capacities:
            # Get dogs for this driver
            driver_dogs = []
            for assignment in self.dog_assignments:
                if isinstance(assignment, dict) and assignment.get('combined', '').startswith(f"{driver}:"):
                    driver_dogs.append(assignment.get('dog_id'))
            
            # Need at least 2 dogs to calculate density
            if len(driver_dogs) < 2:
                self.driver_densities[driver] = 'VERY_DENSE'
                continue
            
            # Calculate distances between all pairs
            distances = []
            for i in range(len(driver_dogs)):
                for j in range(i + 1, len(driver_dogs)):
                    dist = self.safe_get_distance(driver_dogs[i], driver_dogs[j])
                    if dist < self.EXCLUSION_DISTANCE:
                        distances.append(dist)
            
            if distances:
                avg_distance = sum(distances) / len(distances)
                
                # Categorize density
                if avg_distance < 0.3:
                    density = 'VERY_DENSE'
                elif avg_distance < 0.5:
                    density = 'DENSE'
                elif avg_distance < 0.8:
                    density = 'MODERATE'
                elif avg_distance < 1.2:
                    density = 'SPREAD_OUT'
                else:
                    density = 'VERY_SPREAD'
                
                self.driver_densities[driver] = density
                
                # Show capacity info for dense routes
                capacity = self.get_dynamic_capacity(driver)
                if capacity == 12:
                    print(f"   {driver}: {density} (avg {avg_distance:.2f}mi) - CAPACITY 12")
                elif density in ['VERY_SPREAD']:
                    print(f"   {driver}: {density} (avg {avg_distance:.2f}mi)")
            else:
                self.driver_densities[driver] = 'MODERATE'

    def optimize_existing_assignments_with_swaps(self, current_assignments):
        """Scan all existing assignments and swap dogs to better drivers"""
        print("\n🔄 OPTIMIZING EXISTING ASSIGNMENTS WITH SWAPS")
        print(f"   Swap threshold: {self.SWAP_THRESHOLD} miles")
        
        swaps_made = []
        swap_candidates = []
        
        # Check each existing assignment for better placement
        for assignment in current_assignments:
            dog_id = assignment['dog_id']
            dog_name = assignment['dog_name']
            current_driver = assignment.get('driver')
            groups = assignment.get('needed_groups', [])
            
            if not current_driver or current_driver not in self.driver_capacities:
                continue
            
            # Find current minimum distance to other dogs in this driver's route
            current_dogs = [a['dog_id'] for a in current_assignments 
                          if a['driver'] == current_driver and a['dog_id'] != dog_id]
            
            current_min_distance = float('inf')
            for other_dog in current_dogs:
                dist = self.safe_get_distance(dog_id, other_dog)
                if dist < current_min_distance:
                    current_min_distance = dist
            
            # Find all drivers who could take this dog
            compatible_drivers = self.find_drivers_with_group_compatibility(groups)
            
            # Check each potential driver
            for potential_driver in compatible_drivers:
                if potential_driver == current_driver:
                    continue
                
                # Skip if driver doesn't have capacity
                if not self.check_driver_capacity_for_groups(potential_driver, groups):
                    continue
                
                # Find minimum distance to dogs in potential driver's route
                potential_dogs = [a['dog_id'] for a in current_assignments 
                               if a['driver'] == potential_driver]
                
                if not potential_dogs:
                    continue
                
                potential_min_distance = float('inf')
                for other_dog in potential_dogs:
                    dist = self.safe_get_distance(dog_id, other_dog)
                    if dist < potential_min_distance:
                        potential_min_distance = dist
                
                # Calculate improvement
                improvement = current_min_distance - potential_min_distance
                
                if improvement >= self.SWAP_THRESHOLD:
                    swap_candidates.append({
                        'dog_id': dog_id,
                        'dog_name': dog_name,
                        'from_driver': current_driver,
                        'to_driver': potential_driver,
                        'groups': groups,
                        'current_distance': current_min_distance,
                        'new_distance': potential_min_distance,
                        'improvement': improvement
                    })
        
        # Sort by improvement (best first) and execute swaps
        swap_candidates.sort(key=lambda x: x['improvement'], reverse=True)
        
        for swap in swap_candidates:
            # Re-check capacity (might have changed due to previous swaps)
            if self.check_driver_capacity_for_groups(swap['to_driver'], swap['groups']):
                swaps_made.append(swap)
                
                # Update the assignment
                for assignment in current_assignments:
                    if assignment['dog_id'] == swap['dog_id']:
                        assignment['driver'] = swap['to_driver']
                        assignment['combined'] = f"{swap['to_driver']}:{''.join(map(str, swap['groups']))}"
                        break
                
                print(f"   ✅ Swapped {swap['dog_name']} from {swap['from_driver']} to {swap['to_driver']} (saves {swap['improvement']:.2f} miles)")
        
        if swaps_made:
            total_improvement = sum(s['improvement'] for s in swaps_made)
            print(f"\n   🎯 Made {len(swaps_made)} swaps, saving {total_improvement:.1f} total miles!")
        
        self.optimization_swaps = swaps_made
        return swaps_made

    def reassign_dogs_closest_first_strategy(self):
        """Assign all callouts to closest driver - with dynamic capacity"""
        print("\n🎯 Starting CLOSEST-FIRST assignment strategy")
        
        # Build current state
        current_assignments = self.build_initial_assignments_state()
        
        # Get callout dogs
        callout_dogs = []
        for d in self.dog_assignments:
            if (d.get('needed_groups') and  # Has groups specified
                not d.get('combined', '').strip()):  # But no assignment
                callout_dogs.append(d)
        
        print(f"📊 Found {len(callout_dogs)} callout dogs to assign")
        
        # Debug: Show first few callouts
        if callout_dogs:
            print("\n📋 Dogs needing assignment:")
            for dog in callout_dogs[:5]:
                print(f"   - {dog['dog_name']} (ID: {dog['dog_id']}) needs groups: {dog['needed_groups']}")
            if len(callout_dogs) > 5:
                print(f"   ... and {len(callout_dogs) - 5} more")
        
        assignments_made = []
        
        # Phase 1: Assign to closest driver
        print("\n📍 Phase 1: Assigning to closest drivers...")
        
        for dog in callout_dogs:
            dog_id = dog['dog_id']
            dog_name = dog['dog_name']
            groups = dog.get('needed_groups', [])
            
            if not groups:
                print(f"   ⚠️ {dog_name} has no groups specified (shouldn't happen)")
                continue
            
            # Find compatible drivers
            compatible_drivers = self.find_drivers_with_group_compatibility(groups)
            
            if not compatible_drivers:
                print(f"   ❌ No driver can handle groups {groups} for {dog_name}")
                continue
            
            # Find closest driver OR first available if no dogs to compare
            best_driver = None
            best_distance = float('inf')
            drivers_with_dogs = []
            drivers_without_dogs = []
            
            for driver in compatible_drivers:
                # Skip fake drivers
                if driver.lower() in ['field', 'parking']:
                    continue
                    
                # Get dogs currently assigned to this driver
                driver_dogs = [a['dog_id'] for a in current_assignments if a['driver'] == driver]
                
                if driver_dogs:
                    # Driver has dogs - calculate minimum distance
                    min_distance = float('inf')
                    for other_dog in driver_dogs:
                        dist = self.safe_get_distance(dog_id, other_dog)
                        if dist < min_distance:
                            min_distance = dist
                    
                    drivers_with_dogs.append((driver, min_distance))
                    
                    if min_distance < best_distance:
                        best_distance = min_distance
                        best_driver = driver
                else:
                    # Driver has no dogs yet - add to backup list
                    drivers_without_dogs.append(driver)
            
            # If no driver with dogs was suitable, assign to first available empty driver
            if not best_driver and drivers_without_dogs:
                best_driver = drivers_without_dogs[0]
                best_distance = 0.0  # No distance to compare for empty driver
                print(f"   📍 Assigning to empty driver: {best_driver}")
            
            if best_driver:
                # Make assignment
                groups_str = ''.join(map(str, sorted(groups)))
                
                # Check current capacity for this driver
                current_capacity = self.get_dynamic_capacity(best_driver)
                capacity_info = " [DENSE: cap 12]" if current_capacity == 12 else ""
                
                assignment = {
                    'dog_id': dog_id,
                    'dog_name': dog_name,
                    'new_assignment': f"{best_driver}:{groups_str}",
                    'driver': best_driver,
                    'distance': best_distance,
                    'quality': 'GOOD' if best_distance < 0.2 else 'BACKUP' if best_distance < 0.5 else 'NEW_DRIVER' if best_distance == 0 else 'EMERGENCY',
                    'assignment_type': 'closest_first'
                }
                assignments_made.append(assignment)
                
                # Update dog's assignment
                dog['combined'] = assignment['new_assignment']
                dog['current_driver'] = best_driver
                
                # Add to current assignments for next distance calculations
                current_assignments.append({
                    'dog_id': dog_id,
                    'dog_name': dog_name,
                    'driver': best_driver,
                    'combined': assignment['new_assignment'],
                    'needed_groups': groups,
                    'groups': groups
                })
                
                if best_distance == 0:
                    print(f"   ✅ {dog_name} → {best_driver} (first dog for this driver) for groups {groups}")
                else:
                    print(f"   ✅ {dog_name} → {best_driver} ({best_distance:.2f}mi) for groups {groups}{capacity_info}")
            else:
                print(f"   ❌ Could not find suitable driver for {dog_name} (groups: {groups})")
        
        # Phase 2: Check and fix capacity violations
        print("\n🔧 Phase 2: Checking capacity violations...")
        violations = self.verify_capacity_constraints(current_assignments)
        
        if violations:
            print(f"   ⚠️ Found {len(violations)} capacity violations to fix")
            for v in violations:
                print(f"      - {v['driver']} group {v['group']}: {v['count']}/{v['max']} (excess: {v['excess']})")
        else:
            print("   ✅ No capacity violations found")
        
        # Phase 3: Optimize through swaps
        drivers_with_multiple_dogs = set()
        for driver in self.driver_capacities:
            driver_dogs = [a for a in current_assignments if a['driver'] == driver]
            if len(driver_dogs) >= 2:
                drivers_with_multiple_dogs.add(driver)
        
        if drivers_with_multiple_dogs:
            print(f"\n🔄 Found {len(drivers_with_multiple_dogs)} drivers with multiple dogs - checking for optimization")
            self.calculate_route_densities()
            swaps = self.optimize_existing_assignments_with_swaps(current_assignments)
            
            # Add swaps to assignments
            for swap in swaps:
                assignments_made.append({
                    'dog_id': swap['dog_id'],
                    'dog_name': swap['dog_name'],
                    'new_assignment': f"{swap['to_driver']}:{''.join(map(str, swap['groups']))}",
                    'driver': swap['to_driver'],
                    'distance': swap['new_distance'],
                    'quality': 'OPTIMIZED',
                    'assignment_type': 'swap_optimization'
                })
        
        self.callouts_assigned = len([a for a in assignments_made if a['assignment_type'] == 'closest_first'])
        self.total_miles_saved = sum(s.get('improvement', 0) for s in self.optimization_swaps) if hasattr(self, 'optimization_swaps') else 0
        
        print(f"\n📊 ASSIGNMENT COMPLETE:")
        print(f"   Callouts assigned: {self.callouts_assigned}")
        print(f"   Dogs swapped: {len(self.optimization_swaps) if hasattr(self, 'optimization_swaps') else 0}")
        print(f"   Miles saved: {self.total_miles_saved:.1f}")
        
        return assignments_made

    def write_results_to_sheets(self, reassignments):
        """Write results back to Google Sheets - to column H"""
        try:
            print("\n📝 Writing results to Google Sheets...")
            
            sheet = self.gc.open_by_key(self.MAP_SHEET_ID).worksheet(self.MAP_TAB)
            
            # Get current data
            all_values = sheet.get_all_values()
            headers = all_values[0] if all_values else []
            
            # Column indices
            dog_id_idx = 9        # Column J
            combined_idx = 7      # Column H - Combined Assignment
            
            # Process reassignments
            batch_updates = []
            rows_updated = 0
            
            for assignment in reassignments:
                dog_id = str(assignment.get('dog_id', '')).strip()
                new_assignment = str(assignment.get('new_assignment', '')).strip()
                
                # Skip UNASSIGNED dogs
                if 'UNASSIGNED:' in new_assignment:
                    print(f"  ⏭️ Skipping unassigned dog {dog_id}")
                    continue
                
                # Final validation
                if not new_assignment or new_assignment == dog_id or ':' not in new_assignment:
                    print(f"  ❌ SKIPPING invalid assignment for {dog_id}")
                    continue
                
                # Find the row
                for i, row in enumerate(all_values[1:], start=2):
                    if len(row) > dog_id_idx and row[dog_id_idx] == dog_id:
                        batch_updates.append({
                            'range': f'{chr(65 + combined_idx)}{i}',  # Column H
                            'values': [[new_assignment]]
                        })
                        rows_updated += 1
                        break
            
            # Apply batch updates
            if batch_updates:
                try:
                    # Rate limit if needed
                    current_time = time.time()
                    if current_time - self.last_sheet_update < self.MIN_UPDATE_INTERVAL:
                        wait_time = self.MIN_UPDATE_INTERVAL - (current_time - self.last_sheet_update)
                        print(f"⏱️ Rate limiting: waiting {wait_time:.1f}s")
                        time.sleep(wait_time)
                    
                    sheet.batch_update(batch_updates)
                    self.last_sheet_update = time.time()
                    print(f"✅ Successfully updated {rows_updated} assignments in column H")
                except Exception as e:
                    print(f"❌ Batch update failed: {e}")
                    raise
            else:
                print("ℹ️ No valid updates to write")
                
        except Exception as e:
            print(f"❌ Error writing to sheets: {e}")
            raise

    def send_slack_notification(self, reassignments):
        """Send Slack notification about reassignments"""
        webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
        if not webhook_url:
            print("⚠️  Slack webhook not configured")
            return
        
        try:
            # Count types of assignments
            callout_count = len([a for a in reassignments if a.get('assignment_type') == 'closest_first'])
            swap_count = len(self.optimization_swaps)
            
            message = {
                "text": f"🐕 Dog Reassignment Complete: {len(reassignments)} total updates",
                "attachments": [{
                    "color": "good",
                    "fields": [
                        {"title": "Callouts Assigned", "value": str(callout_count), "short": True},
                        {"title": "Dogs Swapped", "value": str(swap_count), "short": True},
                        {"title": "Miles Saved", "value": f"{self.total_miles_saved:.1f}", "short": True},
                        {"title": "Timestamp", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                    ]
                }]
            }
            
            response = requests.post(webhook_url, json=message)
            if response.status_code == 200:
                print("📱 Slack notification sent")
            else:
                print(f"⚠️  Slack notification failed: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️  Error sending Slack notification: {e}")

    def report_haversine_usage(self):
        """Report on how many times haversine fallback was used"""
        if hasattr(self, 'haversine_fallback_count') and self.haversine_fallback_count > 0:
            print(f"\n📍 Haversine Fallback Usage:")
            print(f"   Used {self.haversine_fallback_count} times for {len(self.haversine_pairs)} unique dog pairs")
            print(f"   These dogs are not in your distance matrix but have coordinates")
            
            # Show some examples
            examples = list(self.haversine_pairs)[:3]
            if examples:
                print(f"   Examples:")
                for dog1, dog2 in examples:
                    print(f"     - {dog1} ↔ {dog2}")


def main():
    """Main entry point with optional distance analysis"""
    try:
        # Initialize system
        system = DogReassignmentSystem()
        
        # Check if running in GitHub Actions
        if os.environ.get('GITHUB_ACTIONS'):
            # Auto-run optimization in GitHub Actions
            print("\n🤖 Running in GitHub Actions - auto-selecting optimization")
            choice = '1'
        else:
            # Ask user what they want to do
            print("\n" + "="*60)
            print("What would you like to do?")
            print("1. Run optimization and reassignments")
            print("2. Analyze within-group distances only")
            print("3. Both (analyze first, then optimize)")
            print("="*60)
            
            choice = input("Enter choice (1/2/3): ").strip()
        
        if choice in ['2', '3']:
            # Run distance analysis
            system.analyze_within_group_distances()
        
        if choice in ['1', '3']:
            # Run optimization
            reassignments = system.reassign_dogs_closest_first_strategy()
            
            # Write results
            if reassignments:
                system.write_results_to_sheets(reassignments)
                system.send_slack_notification(reassignments)
            else:
                print("\nℹ️ No reassignments needed")
        
        # Report haversine usage
        system.report_haversine_usage()
        
        print("\n✅ Process complete!")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
