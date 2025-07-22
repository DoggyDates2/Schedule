#!/usr/bin/env python3
"""
Dog Reassignment Optimization System
Optimizes dog-to-driver assignments using a 7-phase approach.
Uses "closest individual dog" logic for all reassignments.
Flexible capacity: 8-12 dogs based on route density (max 12)
"""

import os
import sys
import traceback

try:
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Please install required packages:")
    print("  pip install gspread google-auth google-auth-oauthlib google-auth-httplib2")
    sys.exit(1)

from collections import defaultdict
import json
import math

class DogReassignmentSystem:
    def __init__(self):
        # Initialize Google Sheets
        self.setup_google_sheets()
        
        # Data structures
        self.driver_assignments = {}  # {dog_id: driver_name}
        self.dog_groups = {}  # {dog_id: group_number}
        self.time_matrix = {}  # {(dog1, dog2): time_in_minutes}
        self.dog_coordinates = {}  # {dog_id: (lat, lon)}
        self.callout_dogs = set()
        self.all_dogs = set()
        self.active_drivers = set()
        
    def setup_google_sheets(self):
        """Setup Google Sheets connection"""
        try:
            # Check if service account file exists
            if not os.path.exists('service_account_key.json'):
                print("ERROR: service_account_key.json not found!")
                print("Please ensure the service account key file is in the current directory.")
                sys.exit(1)
                
            creds = Credentials.from_service_account_file(
                'service_account_key.json',
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            self.gc = gspread.authorize(creds)
            
            # Production sheet
            self.sheet = self.gc.open_by_key('1m5bCsRQ4avq-p-cGVmHlXTilkRNrS8fNLfRx1F5E6oQ')
            self.drivers_ws = self.sheet.worksheet('Driver View')
            self.time_ws = self.sheet.worksheet('Driving Time')
            
        except Exception as e:
            print(f"ERROR: Failed to setup Google Sheets connection: {e}")
            print(f"Error type: {type(e).__name__}")
            traceback.print_exc()
            sys.exit(1)
        
    def load_data(self):
        """Load all data from Google Sheets"""
        print("Loading data from Google Sheets...")
        
        try:
            # Load driver assignments and groups
            print("Loading driver assignments...")
            drivers_data = self.drivers_ws.get_all_values()
            
            if len(drivers_data) < 4:
                print("ERROR: Driver sheet appears to be empty or malformed!")
                sys.exit(1)
            
            # Skip header rows and load assignments
            for row_idx, row in enumerate(drivers_data[3:], start=4):  # Starting from row 4
                driver = row[0] if row[0] else None
                
                # Load all 3 groups for this driver
                for group in [1, 2, 3]:
                    col_idx = 2 + (group - 1) * 2  # Columns C, E, G
                    if col_idx < len(row) and row[col_idx]:
                        dogs = [d.strip() for d in row[col_idx].split(',') if d.strip()]
                        for dog in dogs:
                            self.driver_assignments[dog] = driver
                            self.dog_groups[dog] = group
                            self.all_dogs.add(dog)
                            if driver:
                                self.active_drivers.add(driver)
            
            # Identify callout dogs (dogs without drivers)
            self.callout_dogs = {dog for dog in self.all_dogs if not self.driver_assignments.get(dog)}
            
            print(f"Loaded {len(self.all_dogs)} dogs")
            print(f"Found {len(self.callout_dogs)} callout dogs")
            print(f"Active drivers: {len(self.active_drivers)}")
            
            if len(self.all_dogs) == 0:
                print("ERROR: No dogs found in the sheet!")
                sys.exit(1)
            
            # Load time matrix
            print("Loading time matrix...")
            self.load_time_matrix()
            
            # Load dog coordinates
            print("Loading dog coordinates...")
            self.load_dog_coordinates()
            
        except Exception as e:
            print(f"ERROR: Failed to load data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
    def load_time_matrix(self):
        """Load driving time matrix"""
        try:
            time_data = self.time_ws.get_all_values()
            
            if not time_data or len(time_data) < 2:
                print("WARNING: Time matrix appears to be empty!")
                return
            
            # First row contains dog names
            dogs_in_matrix = time_data[0][1:]  # Skip first cell
            
            # Load times
            for row_idx, row in enumerate(time_data[1:], start=1):
                if not row:
                    continue
                    
                dog1 = row[0]
                for col_idx, time_str in enumerate(row[1:], start=1):
                    if col_idx - 1 < len(dogs_in_matrix):
                        dog2 = dogs_in_matrix[col_idx - 1]
                        if time_str:
                            try:
                                time_val = float(time_str)
                                self.time_matrix[(dog1, dog2)] = time_val
                                self.time_matrix[(dog2, dog1)] = time_val
                            except (ValueError, TypeError):
                                pass
            
            print(f"Loaded {len(self.time_matrix)} time entries")
            
        except Exception as e:
            print(f"WARNING: Failed to load time matrix: {e}")
            print("Will use haversine fallback for all distances.")
        
    def load_dog_coordinates(self):
        """Load dog coordinates for haversine calculations"""
        # This would load from a coordinates sheet
        # For now, using placeholder
        pass
        
    def get_time(self, dog1, dog2):
        """Get time between two dogs (uses haversine if not in matrix)"""
        if dog1 == dog2:
            return 0
            
        if (dog1, dog2) in self.time_matrix:
            return self.time_matrix[(dog1, dog2)]
        elif (dog2, dog1) in self.time_matrix:
            return self.time_matrix[(dog2, dog1)]
        else:
            # Fallback to haversine
            return self.haversine_time_fallback(dog1, dog2)
            
    def haversine_time_fallback(self, dog1, dog2):
        """Fallback time calculation using haversine formula"""
        self.haversine_count += 1
        
        if dog1 in self.dog_coordinates and dog2 in self.dog_coordinates:
            lat1, lon1 = self.dog_coordinates[dog1]
            lat2, lon2 = self.dog_coordinates[dog2]
            return self.haversine_time(lat1, lon1, lat2, lon2)
        else:
            # If no coordinates, return a default time
            return 15.0  # Default 15 minutes
            
    def haversine_time(self, lat1, lon1, lat2, lon2):
        """Calculate driving time estimate from haversine distance"""
        # Check for same location
        if lat1 == lat2 and lon1 == lon2:
            return 0.0
            
        # Validate coordinates
        if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
            return 15.0
        if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
            return 15.0
            
        # Haversine formula
        R = 3959  # Earth's radius in miles
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Clamp values to prevent math domain error
        a_val = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        a_val = max(0, min(1, a_val))  # Clamp between 0 and 1
        
        c = 2 * math.asin(math.sqrt(a_val))
        distance = R * c
        
        # Convert to time (assume 25 mph average speed in suburban areas)
        time = (distance / 25) * 60  # Convert to minutes
        
        # Add buffer for stops/traffic
        time = time * 1.3
        
        return max(0, time)  # Ensure non-negative
        
    def get_flexible_capacity(self, driver, group):
        """Get flexible capacity based on route density"""
        # Get all dogs for this driver/group
        driver_dogs = [
            d for d, drv in self.driver_assignments.items()
            if drv == driver and self.dog_groups.get(d) == group
        ]
        
        if len(driver_dogs) <= 1:
            return 12  # Max capacity for single or no dogs
        
        # Calculate average distance between dogs
        total_time = 0
        count = 0
        
        for i, dog1 in enumerate(driver_dogs):
            for dog2 in driver_dogs[i+1:]:
                time = self.get_time(dog1, dog2)
                if time is not None:
                    total_time += time
                    count += 1
        
        if count == 0:
            # No valid time data, use default capacity
            return 10
        
        avg_time = total_time / count
        
        # Flexible capacity based on density (max 12)
        if avg_time < 1.5:
            return 12  # Super dense route
        elif avg_time < 2.0:
            return 11  # Dense route
        elif avg_time < 3.0:
            return 10  # Normal route
        else:
            return 8   # Spread out route
            
    def phase1_assign_callouts(self):
        """Phase 1: Assign all callout dogs to available drivers (ignores capacity constraints)"""
        print("\n" + "="*50)
        print("PHASE 1: CALLOUT ASSIGNMENT")
        print("="*50)
        
        phase1_moves = []
        
        # Check if we have any callouts
        if not self.callout_dogs:
            print("No callout dogs to assign.")
            return phase1_moves
        
        # Check if we have any active drivers
        if not self.active_drivers:
            print("❌ No active drivers available for callout assignment!")
            return phase1_moves
        
        for callout_id in self.callout_dogs:
            if callout_id not in self.driver_assignments:
                # Find the closest individual dog (and its driver)
                best_driver = None
                best_time = float('inf')
                closest_dog = None
                
                for driver in self.active_drivers:
                    driver_dogs = [
                        dog_id for dog_id, d in self.driver_assignments.items() 
                        if d == driver
                    ]
                    
                    for dog_id in driver_dogs:
                        time = self.get_time(callout_id, dog_id)
                        if time is not None and time < best_time:
                            best_time = time
                            best_driver = driver
                            closest_dog = dog_id
                
                if best_driver:
                    self.driver_assignments[callout_id] = best_driver
                    move = {
                        'dog_id': callout_id,
                        'from_driver': 'UNASSIGNED',
                        'to_driver': best_driver,
                        'group': self.dog_groups.get(callout_id, 'Unknown'),
                        'reason': f'Callout assignment (closest to {closest_dog})',
                        'time': best_time
                    }
                    phase1_moves.append(move)
                    self.all_moves.append(move)
                    print(f"✅ {callout_id} → {best_driver} (closest at {best_time:.1f} min to {closest_dog})")
        
        print(f"\nPhase 1 complete: {len(phase1_moves)} callouts assigned")
        return phase1_moves
        
    def phase2_consolidate_drivers(self):
        """Phase 2: Consolidate drivers with <12 dogs total"""
        print("\n" + "="*50)
        print("PHASE 2: DRIVER CONSOLIDATION")
        print("="*50)
        
        phase2_moves = []
        
        # Count total dogs per driver
        driver_totals = defaultdict(int)
        for dog, driver in self.driver_assignments.items():
            if driver:
                driver_totals[driver] += 1
        
        # Find drivers with less than 12 dogs
        drivers_to_remove = [
            driver for driver, total in driver_totals.items() 
            if total < 12 and driver in self.active_drivers
        ]
        
        if not drivers_to_remove:
            print("No drivers need consolidation.")
            return phase2_moves
        
        # Sort by fewest dogs first
        drivers_to_remove.sort(key=lambda d: driver_totals[d])
        
        print(f"Consolidating {len(drivers_to_remove)} drivers with <12 dogs:")
        for driver in drivers_to_remove:
            print(f"  - {driver}: {driver_totals[driver]} dogs")
        
        # Reassign dogs from these drivers
        for from_driver in drivers_to_remove:
            dogs_to_reassign = [
                dog for dog, driver in self.driver_assignments.items() 
                if driver == from_driver
            ]
            
            print(f"\nReassigning {len(dogs_to_reassign)} dogs from {from_driver}:")
            
            for dog_id in dogs_to_reassign:
                # Validate dog has a group assignment
                group = self.dog_groups.get(dog_id)
                if not group:
                    print(f"WARNING: {dog_id} has no group assignment! Skipping...")
                    continue
                    
                # Find the closest individual dog (not from drivers being removed)
                best_driver = None
                best_time = float('inf')
                closest_dog = None
                
                for driver in self.active_drivers:
                    if driver in drivers_to_remove:  # Skip drivers being removed
                        continue
                        
                    driver_dogs = [
                        d for d, drv in self.driver_assignments.items() 
                        if drv == driver and self.dog_groups.get(d) == group
                    ]
                    
                    for other_dog in driver_dogs:
                        time = self.get_time(dog_id, other_dog)
                        if time is not None and time < best_time:
                            best_time = time
                            best_driver = driver
                            closest_dog = other_dog
                
                if best_driver:
                    self.driver_assignments[dog_id] = best_driver
                    move = {
                        'dog_id': dog_id,
                        'from_driver': from_driver,
                        'to_driver': best_driver,
                        'group': group,
                        'reason': f'Driver consolidation (closest to {closest_dog})',
                        'time': best_time
                    }
                    phase2_moves.append(move)
                    self.all_moves.append(move)
                    print(f"  ✅ {dog_id} → {best_driver} (closest at {best_time:.1f} min to {closest_dog})")
        
        # Remove drivers from active set
        for driver in drivers_to_remove:
            self.active_drivers.discard(driver)
            print(f"   {driver} removed from active drivers")
        
        print(f"\nPhase 2 complete: {len(phase2_moves)} dogs moved, {len(drivers_to_remove)} drivers removed")
        return phase2_moves
        
    def phase3_cluster_nearby(self):
        """Phase 3: Cluster dogs that are <1 minute apart (marks protected clusters)"""
        print("\n" + "="*50)
        print("PHASE 3: CLUSTER NEARBY DOGS")
        print("="*50)
        
        phase3_moves = []
        protected_clusters = []  # Track clusters to protect
        
        # Process each driver
        for driver in self.active_drivers:
            driver_dogs = [
                dog for dog, d in self.driver_assignments.items() 
                if d == driver
            ]
            
            # Process each group
            for group in [1, 2, 3]:
                group_dogs = [
                    dog for dog in driver_dogs 
                    if self.dog_groups.get(dog) == group
                ]
                
                if len(group_dogs) < 2:
                    continue
                
                # Find pairs of dogs that are very close (<1 minute)
                close_pairs = []
                for i, dog1 in enumerate(group_dogs):
                    for dog2 in group_dogs[i+1:]:
                        time = self.get_time(dog1, dog2)
                        if time is not None and time < 1.0:
                            close_pairs.append((dog1, dog2, time))
                
                if not close_pairs:
                    continue
                
                # Sort by smallest time gap
                close_pairs.sort(key=lambda x: x[2])
                
                # Try to cluster very close dogs
                for dog1, dog2, time_gap in close_pairs:
                    driver1 = self.driver_assignments.get(dog1)
                    driver2 = self.driver_assignments.get(dog2)
                    
                    if driver1 != driver2:
                        # Move dog2 to driver1 if possible
                        capacity = self.get_flexible_capacity(driver1, group)
                        current_count = len([
                            d for d, drv in self.driver_assignments.items()
                            if drv == driver1 and self.dog_groups.get(d) == group
                        ])
                        
                        if current_count < capacity:
                            self.driver_assignments[dog2] = driver1
                            move = {
                                'dog_id': dog2,
                                'from_driver': driver2,
                                'to_driver': driver1,
                                'group': group,
                                'reason': f'Clustering (<{time_gap:.1f} min apart)',
                                'time': time_gap
                            }
                            phase3_moves.append(move)
                            self.all_moves.append(move)
                            print(f"✅ Clustered {dog2} with {dog1} "
                                  f"({time_gap:.1f} min gap) → {driver1}")
                    
                    # Regardless of move, mark these as protected cluster
                    if time_gap < 1.0:
                        # Form a new cluster with all dogs involved
                        new_cluster = set()
                        cluster_id = len(protected_clusters) + 1
                        
                        # Add all dogs from both sides
                        new_cluster.add(dog1)
                        new_cluster.add(dog2)
                        
                        # If either dog is already in a cluster, merge the clusters
                        clusters_to_merge = []
                        for existing_cluster in protected_clusters:
                            if dog1 in existing_cluster or dog2 in existing_cluster:
                                clusters_to_merge.append(existing_cluster)
                        
                        # Merge all related clusters
                        for cluster in clusters_to_merge:
                            new_cluster.update(cluster)
                            protected_clusters.remove(cluster)
                        
                        # Add all dogs in the new cluster to protected set
                        for dog in new_cluster:
                            self.protected_clusters.add(dog)
                        
                        protected_clusters.append(new_cluster)
        
        print(f"\nProtected {len(protected_clusters)} clusters "
              f"({len(self.protected_clusters)} total dogs)")
        print(f"Phase 3 complete: {len(phase3_moves)} dogs clustered")
        return phase3_moves
        
    def phase4_remove_outliers(self):
        """Phase 4: Remove outliers from all groups (with protected cluster check)"""
        print("\n" + "="*50)
        print("PHASE 4: REMOVE OUTLIERS (ALL GROUPS)")
        print("="*50)
        
        outlier_moves = []
        outlier_threshold = 5.0  # Dogs more than 5 minutes from nearest neighbor
        
        # Process each driver
        for driver in self.active_drivers:
            driver_dogs = [
                dog for dog, d in self.driver_assignments.items() 
                if d == driver
            ]
            
            # Process each group
            for group in [1, 2, 3]:
                driver_dogs = [
                    dog for dog, d in self.driver_assignments.items() 
                    if d == driver and self.dog_groups.get(dog) == group
                ]
                
                if len(driver_dogs) <= 2:
                    continue
                
                # Calculate average time for this group
                total_time = 0
                count = 0
                for i, dog1 in enumerate(driver_dogs):
                    for dog2 in driver_dogs[i+1:]:
                        time = self.get_time(dog1, dog2)
                        if time is not None:
                            total_time += time
                            count += 1
                
                if count == 0:
                    continue
                    
                avg_time = total_time / count
                
                # Find outliers
                outliers = []
                for dog_id in driver_dogs:
                    # Skip if dog is in a protected cluster
                    if hasattr(self, 'protected_clusters') and dog_id in self.protected_clusters:
                        continue
                    
                    # Calculate minimum distance to nearest neighbor in same group
                    min_time = float('inf')
                    for other_dog in driver_dogs:
                        if other_dog != dog_id:
                            time = self.get_time(dog_id, other_dog)
                            if time is not None and time < min_time:
                                min_time = time
                    
                    # Check if outlier (2x average OR >5 min to nearest)
                    if min_time > 2 * avg_time or min_time > outlier_threshold:
                        outliers.append({
                            'dog_id': dog_id,
                            'min_time': min_time,
                            'avg_time': avg_time
                        })
                
                if outliers:
                    print(f"\n{driver} Group {group}: Found {len(outliers)} outliers "
                          f"(avg time: {avg_time:.1f} min)")
                    
                    # Sort by worst outliers first
                    outliers.sort(key=lambda x: x['min_time'], reverse=True)
                    
                    for outlier in outliers:
                        dog_id = outlier['dog_id']
                        
                        # Find best alternative placement
                        best_alt = self.find_best_alternative_placement(dog_id, driver)
                        
                        if best_alt:
                            old_driver = self.driver_assignments.get(dog_id)
                            self.driver_assignments[dog_id] = best_alt['driver']
                            
                            move = {
                                'dog_id': dog_id,
                                'from_driver': old_driver,
                                'to_driver': best_alt['driver'],
                                'group': group,
                                'reason': f'Outlier (>{outlier_threshold:.0f}min to nearest, closest to {best_alt["closest_dog"]})',
                                'time': best_alt['time']
                            }
                            outlier_moves.append(move)
                            self.all_moves.append(move)
                            
                            print(f"   ✅ {dog_id} → {best_alt['driver']} "
                                  f"(closest at {best_alt['time']:.1f} min to {best_alt['closest_dog']})")
                        else:
                            print(f"   ❌ {dog_id}: No better placement found")
        
        print(f"\nPhase 4 complete: {len(outlier_moves)} outliers moved")
        return outlier_moves
        
    def phase5_consolidate_small_groups(self):
        """Phase 5: Consolidate groups with <4 dogs in Group 1 or 3"""
        print("\n" + "="*50)
        print("PHASE 5: CONSOLIDATE SMALL GROUPS")
        print("="*50)
        
        phase5_moves = []
        
        # Find drivers with small Group 1 or 3
        for driver in self.active_drivers:
            # Check if driver has all 3 groups
            has_group1 = any(
                self.driver_assignments.get(d) == driver and self.dog_groups.get(d) == 1 
                for d in self.all_dogs
            )
            has_group2 = any(
                self.driver_assignments.get(d) == driver and self.dog_groups.get(d) == 2 
                for d in self.all_dogs
            )
            has_group3 = any(
                self.driver_assignments.get(d) == driver and self.dog_groups.get(d) == 3 
                for d in self.all_dogs
            )
            
            if not (has_group1 and has_group2 and has_group3):
                continue  # Skip if driver doesn't have all 3 groups
            
            # Check Group 1
            group1_dogs = [
                d for d, drv in self.driver_assignments.items() 
                if drv == driver and self.dog_groups.get(d) == 1
            ]
            
            if 0 < len(group1_dogs) < 4:
                print(f"\n{driver} Group 1 has only {len(group1_dogs)} dogs - consolidating:")
                
                for dog_id in group1_dogs:
                    # Find the closest individual dog (not from same driver)
                    best_driver = None
                    best_time = float('inf')
                    closest_dog = None
                    
                    for other_driver in self.active_drivers:
                        if other_driver == driver:
                            continue
                            
                        driver_dogs = [
                            d for d, drv in self.driver_assignments.items() 
                            if drv == other_driver and self.dog_groups.get(d) == 1
                        ]
                        
                        for other_dog in driver_dogs:
                            time = self.get_time(dog_id, other_dog)
                            if time is not None and time < best_time:
                                best_time = time
                                best_driver = other_driver
                                closest_dog = other_dog
                    
                    if best_driver and best_time <= 5.0:  # Respect distance limit
                        self.driver_assignments[dog_id] = best_driver
                        move = {
                            'dog_id': dog_id,
                            'from_driver': driver,
                            'to_driver': best_driver,
                            'group': 1,
                            'reason': f'Small group consolidation (closest to {closest_dog})',
                            'time': best_time
                        }
                        phase5_moves.append(move)
                        self.all_moves.append(move)
                        print(f"  ✅ {dog_id} → {best_driver} (closest at {best_time:.1f} min to {closest_dog})")
            
            # Check Group 3
            group3_dogs = [
                d for d, drv in self.driver_assignments.items() 
                if drv == driver and self.dog_groups.get(d) == 3
            ]
            
            if 0 < len(group3_dogs) < 4:
                print(f"\n{driver} Group 3 has only {len(group3_dogs)} dogs - consolidating:")
                
                for dog_id in group3_dogs:
                    # Find the closest individual dog (not from same driver)
                    best_driver = None
                    best_time = float('inf')
                    closest_dog = None
                    
                    for other_driver in self.active_drivers:
                        if other_driver == driver:
                            continue
                            
                        driver_dogs = [
                            d for d, drv in self.driver_assignments.items() 
                            if drv == other_driver and self.dog_groups.get(d) == 3
                        ]
                        
                        for other_dog in driver_dogs:
                            time = self.get_time(dog_id, other_dog)
                            if time is not None and time < best_time:
                                best_time = time
                                best_driver = other_driver
                                closest_dog = other_dog
                    
                    if best_driver and best_time <= 5.0:  # Respect distance limit
                        self.driver_assignments[dog_id] = best_driver
                        move = {
                            'dog_id': dog_id,
                            'from_driver': driver,
                            'to_driver': best_driver,
                            'group': 3,
                            'reason': f'Small group consolidation (closest to {closest_dog})',
                            'time': best_time
                        }
                        phase5_moves.append(move)
                        self.all_moves.append(move)
                        print(f"  ✅ {dog_id} → {best_driver} (closest at {best_time:.1f} min to {closest_dog})")
        
        print(f"\nPhase 5 complete: {len(phase5_moves)} dogs moved from small groups")
        return phase5_moves
        
    def phase6_balance_capacity(self):
        """Phase 6: Balance capacity using nearest neighbor approach"""
        print("\n" + "="*50)
        print("PHASE 6: BALANCE CAPACITY (Nearest Neighbor)")
        print("="*50)
        
        capacity_moves = []
        max_iterations = 50
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            made_moves = False
            
            # Find over-capacity groups
            over_capacity_groups = []
            
            for driver in self.active_drivers:
                for group in [1, 2, 3]:
                    driver_dogs = [
                        d for d, drv in self.driver_assignments.items()
                        if drv == driver and self.dog_groups.get(d) == group
                    ]
                    
                    # Skip if no dogs in this group
                    if not driver_dogs:
                        continue
                        
                    current_count = len(driver_dogs)
                    capacity = self.get_flexible_capacity(driver, group)
                    
                    if current_count > capacity:
                        over_capacity_groups.append({
                            'driver': driver,
                            'group': group,
                            'count': current_count,
                            'capacity': capacity,
                            'excess': current_count - capacity,
                            'dogs': driver_dogs
                        })
            
            if not over_capacity_groups:
                print(f"All groups within capacity after {iteration-1} iterations!")
                break
            
            # Sort by most over capacity
            over_capacity_groups.sort(key=lambda x: x['excess'], reverse=True)
            
            print(f"\nIteration {iteration}: {len(over_capacity_groups)} groups over capacity")
            
            # Process most over-capacity group
            worst_group = over_capacity_groups[0]
            driver = worst_group['driver']
            group = worst_group['group']
            excess = worst_group['excess']
            
            print(f"\n{driver} Group {group}: {worst_group['count']}/{worst_group['capacity']} "
                  f"(need to move {excess} dogs)")
            
            # Analyze connectivity of all dogs in the group
            dogs_with_connectivity = []
            
            for dog_id in worst_group['dogs']:
                # Skip protected clusters
                if hasattr(self, 'protected_clusters') and dog_id in self.protected_clusters:
                    continue
                    
                # Calculate connectivity metrics
                min_distance = float('inf')
                close_neighbors = 0  # < 1 minute
                very_close_neighbors = 0  # < 0.5 minutes
                
                for other_dog in worst_group['dogs']:
                    if other_dog != dog_id:
                        time = self.get_time(dog_id, other_dog)
                        if time is not None:
                            if time < min_distance:
                                min_distance = time
                            if time < 1.0:
                                close_neighbors += 1
                            if time < 0.5:
                                very_close_neighbors += 1
                
                # Find best alternative placement
                best_alt = self.find_best_alternative_placement(dog_id, driver, check_improvement=False)
                
                if best_alt:
                    # Calculate connectivity score (higher = worse connectivity = move first)
                    connectivity_score = (
                        min_distance * 10 +              # Heavily weight isolation
                        (10 / (close_neighbors + 1)) +   # Penalize few close neighbors
                        (5 / (very_close_neighbors + 1)) + # Extra penalty for no very close neighbors
                        (10 / (best_alt['time'] + 1))    # Consider quality of alternative
                    )
                    
                    dogs_with_connectivity.append({
                        'dog_id': dog_id,
                        'min_distance': min_distance,
                        'close_neighbors': close_neighbors,
                        'very_close_neighbors': very_close_neighbors,
                        'score': connectivity_score,
                        'best_alt': best_alt
                    })
            
            # Check if we have any dogs that can be moved
            if not dogs_with_connectivity:
                print(f"   ❌ ERROR: No dogs could be moved! Group will remain over capacity.")
                
                # Forced move: Find least connected dogs even without alternatives
                forced_dogs = []
                for dog_id in worst_group['dogs']:
                    if hasattr(self, 'protected_clusters') and dog_id in self.protected_clusters:
                        continue
                        
                    min_distance = float('inf')
                    for other_dog in worst_group['dogs']:
                        if other_dog != dog_id:
                            time = self.get_time(dog_id, other_dog)
                            if time is not None and time < min_distance:
                                min_distance = time
                    
                    forced_dogs.append({
                        'dog_id': dog_id,
                        'min_distance': min_distance
                    })
                
                # Sort by worst connectivity
                forced_dogs.sort(key=lambda x: x['min_distance'], reverse=True)
                
                # Force move the required number
                moved_count = 0
                for dog_data in forced_dogs[:excess]:
                    old_driver = self.driver_assignments[dog_data['dog_id']]
                    
                    # Forced move: Find driver with least dogs in this group
                    print(f"   ⚠️  All other drivers are at capacity for Group {group}")
                    least_dogs = float('inf')
                    target_driver = None
                    target_dog_count = 0
                    
                    for driver in self.active_drivers:
                        if driver == old_driver:
                            continue
                        driver_group_dogs = [
                            d for d, drvr in self.driver_assignments.items()
                            if drvr == driver and self.dog_groups.get(d) == group
                        ]
                        if len(driver_group_dogs) < least_dogs:
                            least_dogs = len(driver_group_dogs)
                            target_driver = driver
                            target_dog_count = len(driver_group_dogs)
                    
                    if target_driver:
                        # Find closest dog in target driver's group
                        target_dogs = [
                            d for d, drvr in self.driver_assignments.items()
                            if drvr == target_driver and self.dog_groups.get(d) == group
                        ]
                        
                        best_time = float('inf')
                        closest_dog = None
                        for other_dog in target_dogs:
                            time = self.get_time(dog_data['dog_id'], other_dog)
                            if time is not None and time < best_time:
                                best_time = time
                                closest_dog = other_dog
                        
                        # Force the move
                        self.driver_assignments[dog_data['dog_id']] = target_driver
                        
                        move = {
                            'dog_id': dog_data['dog_id'],
                            'from_driver': old_driver,
                            'to_driver': target_driver,
                            'group': group,
                            'reason': f'FORCED - Over capacity (closest to {closest_dog})',
                            'time': best_time
                        }
                        capacity_moves.append(move)
                        self.all_moves.append(move)
                        moved_count += 1
                        
                        print(f"   ⚠️  FORCED MOVE: {dog_data['dog_id']} → {target_driver} "
                              f"(has {target_dog_count} dogs, closest at {best_time:.1f} min to {closest_dog})")
                        print(f"   ⚠️  This may create cascade effect in next iteration!")
                
                made_moves = moved_count > 0
            else:
                # Sort by connectivity score (worst connected first)
                dogs_with_connectivity.sort(key=lambda x: x['score'], reverse=True)
                
                print(f"\nDogs ranked by connectivity (worst connected = move first):")
                for i, dog_data in enumerate(dogs_with_connectivity[:10]):  # Show top 10
                    print(f"   {dog_data['dog_id']}: "
                          f"nearest neighbor {dog_data['min_distance']:.1f} min, "
                          f"{dog_data['close_neighbors']} neighbors <1min")
                
                # Move dogs with worst connectivity
                moved_count = 0
                for dog_data in dogs_with_connectivity:
                    if moved_count >= excess:
                        break
                        
                    best_alt = dog_data['best_alt']
                    if best_alt:
                        # Move the dog
                        old_driver = self.driver_assignments[dog_data['dog_id']]
                        self.driver_assignments[dog_data['dog_id']] = best_alt['driver']
                        
                        move = {
                            'dog_id': dog_data['dog_id'],
                            'from_driver': old_driver,
                            'to_driver': best_alt['driver'],
                            'group': group,
                            'reason': f'Over capacity (closest to {best_alt["closest_dog"]})',
                            'time': best_alt['time']
                        }
                        capacity_moves.append(move)
                        self.all_moves.append(move)
                        moved_count += 1
                        
                        print(f"   ✅ {dog_data['dog_id']} → {best_alt['driver']} "
                              f"(closest at {best_alt['time']:.1f} min to {best_alt['closest_dog']})")
                
                made_moves = moved_count > 0
            
            if not made_moves:
                print("No moves possible - stopping")
                break
        
        print(f"\nPhase 6 complete: {len(capacity_moves)} dogs moved for capacity")
        return capacity_moves
        
    def phase7_balance_workloads(self):
        """Phase 7: Balance driver workloads (total dogs across all groups)"""
        print("\n" + "="*50)
        print("PHASE 7: BALANCE DRIVER WORKLOADS")
        print("="*50)
        
        workload_moves = []
        max_iterations = 10
        
        for iteration in range(max_iterations):
            # Calculate total dogs per driver
            driver_totals = defaultdict(int)
            for dog, driver in self.driver_assignments.items():
                if driver:
                    driver_totals[driver] += 1
            
            # Find most and least loaded
            if not driver_totals:
                break
                
            most_loaded = max(driver_totals.items(), key=lambda x: x[1])
            least_loaded = min(driver_totals.items(), key=lambda x: x[1])
            
            # Check if balanced enough (within 5 dogs)
            if most_loaded[1] - least_loaded[1] <= 5:
                print(f"Workloads balanced! Range: {least_loaded[1]}-{most_loaded[1]} dogs")
                break
            
            print(f"\nIteration {iteration + 1}:")
            print(f"Most loaded: {most_loaded[0]} ({most_loaded[1]} dogs)")
            print(f"Least loaded: {least_loaded[0]} ({least_loaded[1]} dogs)")
            
            # Try to move a dog from most to least loaded
            moved = False
            
            for group in [1, 2, 3]:
                if moved:
                    break
                    
                # Check capacity for least loaded driver
                capacity = self.get_flexible_capacity(least_loaded[0], group)
                current_count = len([
                    d for d, driver in self.driver_assignments.items()
                    if driver == least_loaded[0] and self.dog_groups.get(d) == group
                ])
                
                if current_count >= capacity:
                    continue
                
                # Find best dog to move from most loaded driver
                # Get current dogs for this driver/group
                current_dogs = [
                    d for d, driver in self.driver_assignments.items()
                    if driver == most_loaded[0] and self.dog_groups.get(d) == group
                ]
                
                best_dog = None
                best_alt = None
                
                for dog_id in current_dogs:
                    # Temporarily remove dog to find alternative
                    alt = self.find_best_alternative_placement(dog_id, most_loaded[0], check_improvement=False)
                    if alt and alt['time'] <= 5.0:  # Respect distance limit
                        if best_alt is None or alt['time'] < best_alt['time']:
                            best_dog = dog_id
                            best_alt = alt
                
                if best_dog and best_alt:
                    # Make the move
                    self.driver_assignments[best_dog] = best_alt['driver']
                    
                    move = {
                        'dog_id': best_dog,
                        'from_driver': most_loaded[0],
                        'to_driver': best_alt['driver'],
                        'group': group,
                        'reason': f'Workload balancing (closest to {best_alt["closest_dog"]})',
                        'time': best_alt['time']
                    }
                    workload_moves.append(move)
                    self.all_moves.append(move)
                    
                    print(f"Moving {best_dog} from {most_loaded[0]} to {least_loaded[0]} "
                          f"(closest at {best_alt['time']:.1f} min to {best_alt['closest_dog']})")
                    moved = True
                    break
            
            if not moved:
                print("No valid moves found - stopping")
                break
        
        print(f"\nPhase 7 complete: {len(workload_moves)} dogs moved for workload balance")
        return workload_moves
        
    def find_best_alternative_placement(self, dog_id, current_driver, check_improvement=True):
        """Find the best alternative driver for a dog using closest individual dog approach"""
        best_alternative = None
        current_group = self.dog_groups.get(dog_id)
        
        # If we don't know the current time, we can't check improvement
        current_time = None
        if check_improvement and current_driver:
            current_driver_dogs = [
                d for d, driver in self.driver_assignments.items() 
                if driver == current_driver and 
                self.dog_groups.get(d) == current_group and 
                d != dog_id
            ]
            
            if current_driver_dogs:
                # Find closest dog in current placement
                min_time = float('inf')
                for other_dog in current_driver_dogs:
                    time = self.get_time(dog_id, other_dog)
                    if time is not None and time < min_time:
                        min_time = time
                current_time = min_time if min_time != float('inf') else None
        
        # Find the closest individual dog (not from current driver)
        best_time = float('inf')
        best_driver = None
        closest_dog = None
        
        for driver in self.active_drivers:
            if driver == current_driver:
                continue
                
            driver_dogs = [
                d for d, drvr in self.driver_assignments.items() 
                if drvr == driver and self.dog_groups.get(d) == current_group
            ]
            
            if not driver_dogs:
                continue
            
            # Check capacity using flexible limits
            current_count = len(driver_dogs)
            capacity = self.get_flexible_capacity(driver, current_group)
            
            if current_count >= capacity:
                continue
            
            # Find closest dog from this driver
            for other_dog in driver_dogs:
                time = self.get_time(dog_id, other_dog)
                if time is not None and time < best_time:
                    # Check distance limit - never add more than 5 minutes
                    if time <= 5.0:
                        best_time = time
                        best_driver = driver
                        closest_dog = other_dog
        
        if best_driver:
            # Check if this is actually an improvement
            if check_improvement and current_time is not None:
                improvement = current_time - best_time
                if improvement <= 2.0:  # Require at least 2 minute improvement
                    return None
            
            best_alternative = {
                'driver': best_driver,
                'time': best_time,
                'improvement': current_time - best_time if current_time else 0,
                'closest_dog': closest_dog
            }
        
        return best_alternative
        
    def optimize_routes(self):
        """Run all 7 optimization phases"""
        print("\n" + "="*50)
        print("STARTING 7-PHASE OPTIMIZATION")
        print("="*50)
        
        # Validate data integrity
        dogs_to_remove = []
        for dog_id in self.all_dogs:
            if dog_id not in self.dog_groups:
                print(f"WARNING: {dog_id} has no group assignment! Removing from optimization...")
                dogs_to_remove.append(dog_id)
            elif dog_id not in self.driver_assignments:
                print(f"WARNING: {dog_id} has no driver assignment! Removing from optimization...")
                dogs_to_remove.append(dog_id)
        
        for dog_id in dogs_to_remove:
            self.all_dogs.remove(dog_id)
        
        # Initialize tracking
        self.all_moves = []
        self.haversine_count = 0
        
        # Initialize protected clusters set
        self.protected_clusters = set()
        
        # Run phases
        phase1_moves = self.phase1_assign_callouts()
        phase2_moves = self.phase2_consolidate_drivers()
        phase3_moves = self.phase3_cluster_nearby()
        phase4_moves = self.phase4_remove_outliers()
        phase5_moves = self.phase5_consolidate_small_groups()
        phase6_moves = self.phase6_balance_capacity()
        phase7_moves = self.phase7_balance_workloads()
        
        # Summary
        print("\n" + "="*50)
        print("OPTIMIZATION COMPLETE")
        print("="*50)
        print(f"Total moves: {len(self.all_moves)}")
        print(f"Phase 1 (Callouts): {len(phase1_moves)} moves")
        print(f"Phase 2 (Consolidation): {len(phase2_moves)} moves")
        print(f"Phase 3 (Clustering): {len(phase3_moves)} moves")
        print(f"Phase 4 (Outliers): {len(phase4_moves)} moves")
        print(f"Phase 5 (Small Groups): {len(phase5_moves)} moves")
        print(f"Phase 6 (Capacity): {len(phase6_moves)} moves")
        print(f"Phase 7 (Workload): {len(phase7_moves)} moves")
        print(f"\nHaversine fallback used: {self.haversine_count} times")
        
        return self.all_moves
        
    def update_google_sheets(self):
        """Update Google Sheets with optimized assignments"""
        print("\n" + "="*50)
        print("UPDATING GOOGLE SHEETS")
        print("="*50)
        
        # Prepare data for update
        driver_data = defaultdict(lambda: {'1': [], '2': [], '3': []})
        
        for dog_id, driver in self.driver_assignments.items():
            if driver:
                group = str(self.dog_groups.get(dog_id, ''))
                if group in ['1', '2', '3']:
                    driver_data[driver][group].append(dog_id)
        
        # Build update rows
        update_values = []
        
        # Get current sheet data to preserve structure
        current_data = self.drivers_ws.get_all_values()
        
        # Update each driver's row
        driver_rows = {}
        for row_idx, row in enumerate(current_data[3:], start=4):
            if row[0]:  # Has driver name
                driver_rows[row[0]] = row_idx
        
        # Update values
        for driver, groups in driver_data.items():
            if driver in driver_rows:
                row_idx = driver_rows[driver]
                
                # Update cells for each group
                # Group 1: Column C (index 2)
                cell_range = f'C{row_idx}'
                value = ', '.join(sorted(groups['1']))
                self.drivers_ws.update(cell_range, value)
                
                # Group 2: Column E (index 4)
                cell_range = f'E{row_idx}'
                value = ', '.join(sorted(groups['2']))
                self.drivers_ws.update(cell_range, value)
                
                # Group 3: Column G (index 6)
                cell_range = f'G{row_idx}'
                value = ', '.join(sorted(groups['3']))
                self.drivers_ws.update(cell_range, value)
                
                print(f"Updated {driver}: G1={len(groups['1'])}, G2={len(groups['2'])}, G3={len(groups['3'])}")
        
        print("\nGoogle Sheets updated successfully!")
        
    def analyze_results(self):
        """Analyze and report on optimization results"""
        print("\n" + "="*50)
        print("RESULTS ANALYSIS")
        print("="*50)
        
        # Active drivers and workload
        driver_totals = defaultdict(int)
        driver_groups = defaultdict(lambda: {'1': 0, '2': 0, '3': 0})
        
        for dog, driver in self.driver_assignments.items():
            if driver:
                driver_totals[driver] += 1
                group = self.dog_groups.get(dog)
                if group:
                    driver_groups[driver][str(group)] += 1
        
        print(f"\nActive drivers: {len(driver_totals)}")
        print("\nDriver workloads:")
        
        sorted_drivers = sorted(driver_totals.items(), key=lambda x: x[1], reverse=True)
        for driver, total in sorted_drivers:
            groups = driver_groups[driver]
            print(f"  {driver}: {total} dogs (G1:{groups['1']}, G2:{groups['2']}, G3:{groups['3']})")
        
        # Check for capacity issues
        print("\nCapacity check:")
        over_capacity_count = 0
        
        for driver in driver_totals:
            for group in ['1', '2', '3']:
                count = driver_groups[driver][group]
                if count > 0:
                    capacity = self.get_flexible_capacity(driver, int(group))
                    if count > capacity:
                        print(f"  ⚠️  {driver} Group {group}: {count}/{capacity} (OVER CAPACITY)")
                        over_capacity_count += 1
        
        if over_capacity_count == 0:
            print("  ✅ All groups within capacity!")
        else:
            print(f"  ❌ {over_capacity_count} groups still over capacity")
            
        # Analyze route quality
        print("\nRoute quality analysis:")
        total_groups = sum(1 for d in driver_groups.values() 
                          for g in ['1', '2', '3'] 
                          if d[g] > 0)
        
        dense_routes = 0
        sparse_routes = 0
        
        for driver in driver_totals:
            for group in [1, 2, 3]:
                dogs = [
                    d for d, drv in self.driver_assignments.items()
                    if drv == driver and self.dog_groups.get(d) == group
                ]
                
                if len(dogs) > 1:
                    # Calculate average time
                    total_time = 0
                    count = 0
                    
                    for i, dog1 in enumerate(dogs):
                        for dog2 in dogs[i+1:]:
                            time = self.get_time(dog1, dog2)
                            if time is not None:
                                total_time += time
                                count += 1
                    
                    if count > 0:
                        avg_time = total_time / count
                        if avg_time < 2.0:
                            dense_routes += 1
                        elif avg_time > 5.0:
                            sparse_routes += 1
        
        print(f"  Dense routes (<2 min avg): {dense_routes}")
        print(f"  Sparse routes (>5 min avg): {sparse_routes}")
        print(f"  Protected clusters maintained: {len(self.protected_clusters)} dogs")
        
def main():
    try:
        print("Starting Dog Reassignment Optimization System...")
        system = DogReassignmentSystem()
        
        # Load data
        system.load_data()
        
        # Run optimization
        moves = system.optimize_routes()
        
        # Update Google Sheets
        if moves:
            system.update_google_sheets()
        
        # Analyze results
        system.analyze_results()
        
    except KeyboardInterrupt:
        print("\n\nOptimization cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: An unexpected error occurred: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        sys.exit(1)
    
if __name__ == "__main__":
    main()
