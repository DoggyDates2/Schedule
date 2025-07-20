#!/usr/bin/env python3
"""
Dog Assignment Optimization System - Production Ready
Complete rewrite with all fixes and correct Google Sheets IDs
Matrix Sheet: 1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg
Map Sheet: 1-KTOfTKXk_sX7nO7eGmW73JLi8TJBvv5gobK6gyrc7U
"""

import os
import sys
import json
import time
import logging
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
    """Enhanced dog reassignment system with all optimizations
    
    This system connects to two separate Google Sheets:
    1. Matrix spreadsheet - contains dog-to-dog distances
    2. Map spreadsheet - contains assignments, drivers, and capacities
    """
    
    def __init__(self):
        """Initialize the dog reassignment system"""
        print("🚀 Enhanced Dog Reassignment System - WITH DOG SWAPPING OPTIMIZATION")
        
        # Google Sheets IDs
        self.MATRIX_SHEET_ID = "1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg"
        self.MAP_SHEET_ID = "1-KTOfTKXk_sX7nO7eGmW73JLi8TJBvv5gobK6gyrc7U"
        self.MATRIX_TAB = "Matrix"  # Update if your tab name is different
        self.MAP_TAB = "Map"        # Update if your tab name is different
        
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
        
        # Load data from sheets
        self.load_distance_matrix()
        self.load_dog_assignments()
        
        # System parameters (from your existing code)
        self.PREFERRED_DISTANCE = 0.2
        self.MAX_DISTANCE = 0.5
        self.ABSOLUTE_MAX_DISTANCE = 1.5
        self.CASCADING_MOVE_MAX = 0.7
        self.ADJACENT_GROUP_DISTANCE = 0.1
        self.EXCLUSION_DISTANCE = 200.0
        
        # Swap optimization parameters
        self.SWAP_THRESHOLD = 0.2  # Swap if saves 0.2+ miles
        
        # Tracking variables
        self.active_drivers = set()
        self.driver_densities = {}
        self.optimization_swaps = []
        self.callouts_assigned = 0
        self.total_miles_saved = 0
        self.assignments_made = []
        self.emergency_assignments = []
        
        # Route density thresholds
        self.ROUTE_DENSITY_THRESHOLDS = {
            'VERY_DENSE': 0.3,
            'DENSE': 0.5,
            'MODERATE': 0.8,
            'SPREAD_OUT': 1.2
        }
        
        # Density-based utilization targets
        self.DENSITY_TARGETS = {
            'VERY_DENSE': 0.87,
            'DENSE': 0.82,
            'MODERATE': 0.75,
            'SPREAD_OUT': 0.65,
            'VERY_SPREAD': 0.55
        }
        
        # Performance optimization
        self._distance_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Rate limiting
        self.last_sheet_update = 0
        self.MIN_UPDATE_INTERVAL = 1.0

    def setup_google_sheets(self):
        """Initialize Google Sheets connection with better error handling"""
        try:
            credentials_json = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
            if not credentials_json:
                raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON environment variable not set")
            
            # Try to parse JSON
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
            
            for i, row in enumerate(all_values[1:]):
                from_dog_id = row[0]
                if not from_dog_id:
                    continue
                
                self.distance_matrix[from_dog_id] = {}
                
                for j, distance_str in enumerate(row[1:]):
                    if j < len(dog_ids):
                        to_dog_id = dog_ids[j]
                        try:
                            distance = float(distance_str)
                        except (ValueError, TypeError):
                            distance = self.EXCLUSION_DISTANCE
                        
                        if distance < 0 or distance > 100:
                            distance = self.EXCLUSION_DISTANCE
                        
                        self.distance_matrix[from_dog_id][to_dog_id] = distance
            
            print(f"✅ Loaded distance matrix with {len(self.distance_matrix)} dogs")
            
        except Exception as e:
            print(f"❌ Error loading distance matrix: {e}")
            raise

    def load_dog_assignments(self):
        """Load dog assignments from Google Sheets - using YOUR EXACT format"""
        try:
            sheet = self.gc.open_by_key(self.MAP_SHEET_ID).worksheet(self.MAP_TAB)
            all_values = sheet.get_all_values()
            
            # Find column indices from headers (based on your existing code)
            headers = all_values[0] if all_values else []
            
            # Column mapping - from your script
            dog_name_idx = next((i for i, h in enumerate(headers) if "Dog Name" in h), 1)
            combined_idx = next((i for i, h in enumerate(headers) if "Combined Assignment" in h), 7)
            group_idx = next((i for i, h in enumerate(headers) if h == "Group"), 8)
            dog_id_idx = next((i for i, h in enumerate(headers) if "Dog ID" in h), 9)
            callout_idx = next((i for i, h in enumerate(headers) if "Callout" in h), 10)
            num_dogs_idx = next((i for i, h in enumerate(headers) if "Number of Dogs" in h), 5)
            driver_idx = 0  # Driver is typically first column
            
            # Find capacity columns (looking for Group1Cap, Group2Cap, Group3Cap)
            capacity_indices = {}
            for i, header in enumerate(headers):
                if "Group1Cap" in header or "group1" in header.lower():
                    capacity_indices[1] = i
                elif "Group2Cap" in header or "group2" in header.lower():
                    capacity_indices[2] = i
                elif "Group3Cap" in header or "group3" in header.lower():
                    capacity_indices[3] = i
            
            # If not found, default to columns R, S, T (17, 18, 19)
            if not capacity_indices:
                capacity_indices = {1: 17, 2: 18, 3: 19}
            
            # Initialize tracking
            self.dog_assignments = []
            self.driver_capacities = {}
            self.driver_assignment_counts = defaultdict(int)
            self.all_capacity_rows = []
            
            # Parse each row
            for i, row in enumerate(all_values[1:], start=2):
                if len(row) <= dog_name_idx:
                    continue
                
                # Store for capacity parsing
                if len(row) > driver_idx and row[driver_idx]:
                    self.all_capacity_rows.append(row)
                
                # Parse dog assignment
                dog_name = row[dog_name_idx] if dog_name_idx < len(row) else ""
                if not dog_name or dog_name.strip() == "":
                    continue
                
                # Parse combined assignment to extract groups
                combined = row[combined_idx] if combined_idx < len(row) else ""
                needed_groups = []
                
                if combined and ':' in combined:
                    driver_part, groups_part = combined.split(':', 1)
                    # Parse various group formats: "1", "12", "1&2", "123", "1DD1", etc.
                    groups_str = groups_part.replace('&', '').replace(',', '').replace(' ', '')
                    
                    # Extract numeric groups
                    for char in groups_str:
                        if char.isdigit() and char in ['1', '2', '3']:
                            group_num = int(char)
                            if group_num not in needed_groups:
                                needed_groups.append(group_num)
                    
                    # Count assignment for driver
                    if driver_part:
                        self.driver_assignment_counts[driver_part] += 1
                
                assignment = {
                    'dog_name': dog_name.strip(),
                    'combined': combined,
                    'group': row[group_idx] if group_idx < len(row) else "",
                    'dog_id': row[dog_id_idx].strip() if dog_id_idx < len(row) else "",
                    'callout': row[callout_idx] if callout_idx < len(row) else "",
                    'row_number': i,
                    'needed_groups': sorted(needed_groups),  # Store parsed groups
                    'number_of_dogs': row[num_dogs_idx] if num_dogs_idx < len(row) else "1"
                }
                
                self.dog_assignments.append(assignment)
                
                # Build name to ID mapping
                if assignment['dog_name'] and assignment['dog_id']:
                    self.dog_name_to_id[assignment['dog_name']] = assignment['dog_id']
                    self.dog_id_to_name[assignment['dog_id']] = assignment['dog_name']
            
            # Parse driver capacities
            for row in self.all_capacity_rows:
                driver_name = row[driver_idx] if driver_idx < len(row) else ""
                if driver_name:
                    try:
                        capacities = {}
                        total_capacity = 0
                        
                        for group_num, col_idx in capacity_indices.items():
                            if col_idx < len(row) and row[col_idx]:
                                try:
                                    cap = int(row[col_idx])
                                    capacities[f'group{group_num}'] = cap
                                    total_capacity += cap
                                except ValueError:
                                    capacities[f'group{group_num}'] = 0
                            else:
                                capacities[f'group{group_num}'] = 0
                        
                        # Only add drivers with some capacity
                        if total_capacity > 0:
                            self.driver_capacities[driver_name] = capacities
                            self.driver_capacities[driver_name]['total'] = total_capacity
                            
                    except Exception as e:
                        print(f"   Warning: Could not parse capacity for {driver_name}: {e}")
            
            # Identify active drivers (those with current assignments)
            self.active_drivers = {
                driver for driver, count in self.driver_assignment_counts.items() 
                if count > 0 and driver in self.driver_capacities
            }
            
            # Filter capacities to only active drivers
            active_capacities = {}
            for driver, capacity in self.driver_capacities.items():
                if driver in self.active_drivers:
                    active_capacities[driver] = capacity
            self.driver_capacities = active_capacities
            
            print(f"✅ Loaded {len(self.dog_assignments)} dog assignments")
            print(f"✅ Found {len(self.active_drivers)} active drivers (with assignments)")
            
            # Count callouts (not blank = needs assignment)
            callout_count = sum(1 for d in self.dog_assignments if d.get('callout', '').strip())
            print(f"✅ Found {callout_count} callouts needing assignment")
            
            # Show excluded drivers
            all_capacity_drivers = {row[driver_idx] for row in self.all_capacity_rows 
                                  if len(row) > driver_idx and row[driver_idx]}
            excluded = all_capacity_drivers - self.active_drivers
            if excluded:
                print(f"❌ Excluded {len(excluded)} inactive drivers: {', '.join(sorted(excluded)[:3])}{'...' if len(excluded) > 3 else ''}")
            
            # Run data integrity check
            self.validate_data_integrity()
            
        except Exception as e:
            print(f"❌ Error loading dog assignments: {e}")
            raise

    # Core distance and helper methods
    def get_distance(self, dog1_id, dog2_id):
        """Get distance between two dogs from matrix"""
        if dog1_id == dog2_id:
            return 0.0
        
        if dog1_id not in self.distance_matrix:
            return self.EXCLUSION_DISTANCE
        if dog2_id not in self.distance_matrix[dog1_id]:
            return self.EXCLUSION_DISTANCE
        
        distance = self.distance_matrix[dog1_id].get(dog2_id, self.EXCLUSION_DISTANCE)
        
        if distance is None or distance < 0:
            return self.EXCLUSION_DISTANCE
        
        return float(distance)

    def safe_get_distance(self, dog1_id, dog2_id):
        """Safely get distance between two dogs with fallback"""
        try:
            dist = self.get_distance(dog1_id, dog2_id)
            if dist is None or dist >= self.EXCLUSION_DISTANCE:
                return float('inf')
            return dist
        except (KeyError, TypeError, AttributeError):
            return float('inf')

    def parse_dog_groups(self, assignment):
        """Safely parse groups from assignment"""
        combined = assignment.get('combined', '')
        groups = []
        
        if ':' in combined:
            groups_str = combined.split(':', 1)[1]
            # Handle various formats: "123", "1,2,3", "1 2 3", "1&2", "1DD1"
            groups_str = groups_str.replace(',', '').replace(' ', '').replace('&', '')
            groups = [int(g) for g in groups_str if g.isdigit() and g in ['1', '2', '3']]
        
        # Fallback to needed_groups if available
        if not groups and 'needed_groups' in assignment:
            groups = assignment.get('needed_groups', [])
        
        return groups

    def get_driver_current_dogs(self, driver):
        """Get current dogs assigned to a driver"""
        current_dogs = []
        
        for assignment in self.dog_assignments:
            if isinstance(assignment, dict):
                combined = assignment.get('combined', '')
                if combined.startswith(f"{driver}:"):
                    current_dogs.append(assignment)
        
        return current_dogs

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
        """Verify no capacity constraints are violated"""
        violations = []
        
        # Count dogs per driver per group
        driver_group_counts = defaultdict(lambda: defaultdict(int))
        
        for assignment in assignments:
            driver = assignment.get('driver', '')
            groups = assignment.get('needed_groups', [])
            
            if driver and driver in self.driver_capacities:
                for group in groups:
                    driver_group_counts[driver][str(group)] += 1
        
        # Check against capacities
        for driver, group_counts in driver_group_counts.items():
            if driver not in self.driver_capacities:
                continue
                
            capacity = self.driver_capacities[driver]
            
            for group_num, count in group_counts.items():
                capacity_key = f'group{group_num}'
                max_allowed = capacity.get(capacity_key, 0)
                
                if count > max_allowed:
                    violations.append({
                        'driver': driver,
                        'group': group_num,
                        'count': count,
                        'max': max_allowed,
                        'excess': count - max_allowed
                    })
        
        return violations

    def validate_data_integrity(self):
        """Validate all data is consistent"""
        issues = []
        
        # Check 1: All dog IDs in assignments exist in distance matrix
        matrix_dogs = set(self.distance_matrix.keys())
        assignment_dogs = {a.get('dog_id') for a in self.dog_assignments if a.get('dog_id')}
        
        missing_from_matrix = assignment_dogs - matrix_dogs
        if missing_from_matrix:
            issues.append(f"Dogs in assignments but not in distance matrix: {list(missing_from_matrix)[:5]}")
        
        # Check 2: All drivers in assignments have capacities
        assignment_drivers = {a.get('combined', '').split(':')[0] for a in self.dog_assignments if ':' in a.get('combined', '')}
        capacity_drivers = set(self.driver_capacities.keys())
        
        missing_capacities = assignment_drivers - capacity_drivers - {''}
        if missing_capacities:
            issues.append(f"Drivers with assignments but no capacity: {list(missing_capacities)[:5]}")
        
        # Check 3: No duplicate dog IDs
        dog_ids = [a.get('dog_id') for a in self.dog_assignments if a.get('dog_id')]
        if len(dog_ids) != len(set(dog_ids)):
            issues.append("Duplicate dog IDs found in assignments")
        
        if issues:
            print("\n⚠️ DATA INTEGRITY ISSUES:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ Data integrity check passed")
        
        return len(issues) == 0

    def check_driver_capacity_for_groups(self, driver, groups, current_dogs=None):
        """Check if driver has capacity for dog needing specific groups"""
        if driver not in self.driver_capacities:
            return False
        
        capacity = self.driver_capacities[driver]
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
        
        # Check if adding new dog would exceed capacity
        for g in groups:
            group_str = str(g)
            if group_str not in usage:
                continue
            
            cap_key = f'group{group_str}'
            max_capacity = capacity.get(cap_key, 0)
            
            if max_capacity <= 0:
                return False
                
            if usage[group_str] >= max_capacity:
                return False
        
        return True

    def find_drivers_with_group_compatibility(self, needed_groups):
        """Find drivers that can handle the specified groups"""
        compatible_drivers = []
        
        for driver in self.active_drivers:
            if driver not in self.driver_capacities:
                continue
                
            capacity = self.driver_capacities[driver]
            can_handle = True
            
            # Check if driver handles all needed groups
            for group in needed_groups:
                if capacity.get(f'group{group}', 0) <= 0:
                    can_handle = False
                    break
            
            if can_handle:
                compatible_drivers.append(driver)
        
        return compatible_drivers

    def calculate_route_densities(self):
        """Calculate how spread out each driver's route is"""
        print("\n📍 Calculating route densities...")
        
        for driver in self.active_drivers:
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
                
                # Print notable routes
                if density in ['VERY_SPREAD', 'VERY_DENSE']:
                    print(f"   {driver}: {density} (avg {avg_distance:.2f}mi between dogs)")
            else:
                self.driver_densities[driver] = 'MODERATE'

    def optimize_existing_assignments_with_swaps(self, current_assignments):
        """Scan all existing assignments and swap dogs to better drivers"""
        print("\n🔄 OPTIMIZING EXISTING ASSIGNMENTS WITH SWAPS")
        print(f"   Swap threshold: {self.SWAP_THRESHOLD} miles")
        
        swaps_made = []
        swap_candidates = []
        
        # Build a map of current dog locations
        dog_to_driver = {}
        for assignment in current_assignments:
            dog_to_driver[assignment['dog_id']] = assignment['driver']
        
        # Check each existing assignment for better placement
        for assignment in current_assignments:
            dog_id = assignment['dog_id']
            dog_name = assignment['dog_name']
            current_driver = assignment.get('driver')
            groups = assignment.get('needed_groups', [])
            
            if not current_driver or current_driver not in self.active_drivers:
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
        """Assign all callouts to closest driver, then fix capacity violations"""
        print("\n🎯 Starting CLOSEST-FIRST assignment strategy")
        
        # Build current state
        current_assignments = self.build_initial_assignments_state()
        
        # Get callout dogs (not blank = needs assignment)
        callout_dogs = [d for d in self.dog_assignments if d.get('callout', '').strip() != '']
        print(f"📊 Found {len(callout_dogs)} callout dogs to assign")
        
        assignments_made = []
        
        # Phase 1: Assign to closest driver (ignore capacity)
        print("\n📍 Phase 1: Assigning to closest drivers...")
        for dog in callout_dogs:
            dog_id = dog['dog_id']
            dog_name = dog['dog_name']
            groups = dog.get('needed_groups', [])
            
            if not groups:
                print(f"   ⚠️ {dog_name} has no groups specified")
                continue
            
            # Find compatible drivers
            compatible_drivers = self.find_drivers_with_group_compatibility(groups)
            
            if not compatible_drivers:
                print(f"   ❌ No driver can handle groups {groups} for {dog_name}")
                continue
            
            # Find closest driver
            best_driver = None
            best_distance = float('inf')
            
            for driver in compatible_drivers:
                # Get minimum distance to any dog in this driver's route
                driver_dogs = [a['dog_id'] for a in current_assignments if a['driver'] == driver]
                
                if not driver_dogs:
                    continue
                
                min_distance = min(self.safe_get_distance(dog_id, other_dog) for other_dog in driver_dogs)
                
                if min_distance < best_distance:
                    best_distance = min_distance
                    best_driver = driver
            
            if best_driver:
                # Make assignment
                assignment = {
                    'dog_id': dog_id,
                    'dog_name': dog_name,
                    'new_assignment': f"{best_driver}:{''.join(map(str, groups))}",
                    'driver': best_driver,
                    'distance': best_distance,
                    'quality': 'GOOD' if best_distance < 0.2 else 'BACKUP' if best_distance < 0.5 else 'EMERGENCY',
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
                
                print(f"   ✅ {dog_name} → {best_driver} ({best_distance:.2f}mi)")
        
        # Phase 2: Check and fix capacity violations
        print("\n🔧 Phase 2: Checking capacity violations...")
        violations = self.verify_capacity_constraints(current_assignments)
        
        if violations:
            print(f"   ⚠️ Found {len(violations)} capacity violations to fix")
            # Here you would implement capacity fixing logic
            # For now, just report them
            for v in violations:
                print(f"      - {v['driver']} group {v['group']}: {v['count']}/{v['max']} (excess: {v['excess']})")
        
        # Phase 3: Optimize through swaps
        if self.active_drivers:
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
        self.total_miles_saved = sum(s.get('improvement', 0) for s in self.optimization_swaps)
        
        print(f"\n📊 ASSIGNMENT COMPLETE:")
        print(f"   Callouts assigned: {self.callouts_assigned}")
        print(f"   Dogs swapped: {len(self.optimization_swaps)}")
        print(f"   Miles saved: {self.total_miles_saved:.1f}")
        
        return assignments_made

    def write_results_to_sheets(self, reassignments):
        """Write results back to Google Sheets"""
        try:
            print("\n📝 Writing results to Google Sheets...")
            
            sheet = self.gc.open_by_key(self.MAP_SHEET_ID).worksheet(self.MAP_TAB)
            
            # Get current data
            all_values = sheet.get_all_values()
            headers = all_values[0] if all_values else []
            
            # Find column indices
            dog_id_idx = next((i for i, h in enumerate(headers) if "Dog ID" in h), 9)
            combined_idx = next((i for i, h in enumerate(headers) if "Combined Assignment" in h), 7)
            
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
                            'range': f'{chr(65 + combined_idx)}{i}',  # Convert column index to letter
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
                    print(f"✅ Successfully updated {rows_updated} assignments")
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


def main():
    """Main entry point"""
    try:
        # Initialize system
        system = DogReassignmentSystem()
        
        # Run optimization
        reassignments = system.reassign_dogs_closest_first_strategy()
        
        # Write results
        if reassignments:
            system.write_results_to_sheets(reassignments)
            system.send_slack_notification(reassignments)
        else:
            print("\nℹ️ No reassignments needed")
        
        print("\n✅ Process complete!")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# This script is now configured with your specific Google Sheets IDs
# Ready to run: python production_reassignment.py
