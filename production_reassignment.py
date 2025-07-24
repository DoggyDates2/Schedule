#!/usr/bin/env python3
"""
Dog Walking Route Optimization System - Version 3.0
Holistic Geographic Clustering with Post-Assignment Group Balancing
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
from typing import Dict, List, Tuple, Optional, Set

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
        print("Version 3.0 - Holistic Geographic Clustering")
        print("=" * 60)
        
        # Configuration
        self.CLUSTER_THRESHOLD = 1.5  # minutes for tight clustering
        self.NEIGHBOR_THRESHOLD = 2.0  # minutes to consider "nearby"
        self.MIN_DOGS_FOR_DRIVER = 7
        self.OUTLIER_THRESHOLD = 3.5  # minutes
        
        # Capacity settings
        self.CAPACITY_STANDARD = 8
        self.CAPACITY_DENSE = 12
        self.DENSE_ROUTE_THRESHOLD = 2.0  # avg minutes between dogs
        
        # Initialize data structures
        self.dog_assignments = []
        self.distance_matrix = {}
        self.dog_coordinates = {}
        self.active_drivers = set()
        self.driver_clusters = defaultdict(list)  # driver -> list of dogs
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
            
            # Open spreadsheets
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
            matrix_data = self.matrix_sheet.get_all_values()
            
            if not matrix_data:
                print("❌ No data found in matrix sheet")
                return
            
            header_dog_ids = matrix_data[0][1:]
            
            for row_idx, row in enumerate(matrix_data[1:], 1):
                if not row:
                    continue
                    
                from_dog_id = row[0]
                if not from_dog_id:
                    continue
                
                if from_dog_id not in self.distance_matrix:
                    self.distance_matrix[from_dog_id] = {}
                
                for col_idx, distance_str in enumerate(row[1:], 0):
                    if col_idx < len(header_dog_ids):
                        to_dog_id = header_dog_ids[col_idx]
                        
                        if to_dog_id and distance_str:
                            try:
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
    
    def get_time_between_dogs(self, dog_id1: str, dog_id2: str) -> float:
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
    
    def load_dog_assignments(self):
        """Load current dog assignments from Google Sheets"""
        print("\n📋 Loading dog assignments...")
        
        try:
            all_data = self.map_sheet.get_all_values()
            
            if len(all_data) < 2:
                print("❌ No dog data found")
                return
            
            headers = all_data[0]
            col_indices = {}
            
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
                        
                        # Track active drivers and build initial clusters
                        if dog_data['combined'] and ':' in dog_data['combined']:
                            driver = dog_data['combined'].split(':')[0]
                            if driver and driver not in ['Field', 'Parking']:
                                self.active_drivers.add(driver)
                                self.driver_clusters[driver].append(dog_data)
            
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
    
    def calculate_cluster_metrics(self, dogs: List[dict]) -> dict:
        """Calculate metrics for a cluster of dogs"""
        if len(dogs) < 2:
            return {
                'avg_distance': 0,
                'max_distance': 0,
                'density': 0,
                'cohesion': 1.0
            }
        
        distances = []
        for i in range(len(dogs)):
            for j in range(i + 1, len(dogs)):
                dog1_id = dogs[i].get('dog_id', '')
                dog2_id = dogs[j].get('dog_id', '')
                if dog1_id and dog2_id:
                    dist = self.get_time_between_dogs(dog1_id, dog2_id)
                    if dist < float('inf'):
                        distances.append(dist)
        
        if not distances:
            return {
                'avg_distance': float('inf'),
                'max_distance': float('inf'),
                'density': float('inf'),
                'cohesion': 0
            }
        
        avg_dist = sum(distances) / len(distances)
        max_dist = max(distances)
        
        # Cohesion score: how tight the cluster is (0-1, higher is better)
        cohesion = 1.0 / (1.0 + avg_dist)
        
        return {
            'avg_distance': avg_dist,
            'max_distance': max_dist,
            'density': avg_dist,
            'cohesion': cohesion
        }
    
    def find_nearest_cluster(self, dog: dict, exclude_drivers: Set[str] = None) -> Tuple[str, float]:
        """Find the nearest cluster for a dog"""
        dog_id = dog.get('dog_id', '')
        if not dog_id:
            return None, float('inf')
        
        best_driver = None
        best_distance = float('inf')
        
        for driver, cluster_dogs in self.driver_clusters.items():
            if exclude_drivers and driver in exclude_drivers:
                continue
            
            if not cluster_dogs:
                continue
            
            # Find minimum distance to any dog in cluster
            min_dist = float('inf')
            for cluster_dog in cluster_dogs:
                cluster_dog_id = cluster_dog.get('dog_id', '')
                if cluster_dog_id and cluster_dog_id != dog_id:
                    dist = self.get_time_between_dogs(dog_id, cluster_dog_id)
                    min_dist = min(min_dist, dist)
            
            if min_dist < best_distance:
                best_distance = min_dist
                best_driver = driver
        
        return best_driver, best_distance
    
    def optimize_routes(self):
        """Run the complete optimization with holistic clustering approach"""
        print("\n🚀 Starting holistic route optimization...")
        print("=" * 60)
        print("Strategy: Create tight geographic clusters while preserving time windows (G1, G2, G3)")
        print("=" * 60)
        
        total_moves = 0
        
        # PHASE 1: Pure geographic clustering (preserve time windows)
        print("\n📍 PHASE 1: Geographic clustering (preserving time windows)")
        phase1_moves = self.phase1_geographic_clustering()
        total_moves += phase1_moves
        
        # PHASE 2: Assign all unassigned dogs to nearest clusters
        print("\n📍 PHASE 2: Assigning unassigned dogs with correct time windows")
        phase2_moves = self.phase2_assign_unassigned()
        total_moves += phase2_moves
        
        # PHASE 3: Tighten clusters by removing outliers
        print("\n📍 PHASE 3: Tightening clusters (preserving time windows)")
        phase3_moves = self.phase3_tighten_clusters()
        total_moves += phase3_moves
        
        # PHASE 4: Consolidate small drivers
        print("\n📍 PHASE 4: Consolidating small drivers (preserving time windows)")
        phase4_moves = self.phase4_consolidate_small_drivers()
        total_moves += phase4_moves
        
        # PHASE 5: Ensure correct group assignments
        print("\n📍 PHASE 5: Ensuring correct time window assignments")
        phase5_moves = self.phase5_assign_groups()
        total_moves += phase5_moves
        
        # PHASE 6: Balance groups with cascading
        print("\n📍 PHASE 6: Balancing time window groups with cascading")
        phase6_moves = self.phase6_balance_groups()
        total_moves += phase6_moves
        
        # Summary
        print("\n" + "=" * 60)
        print("🏁 OPTIMIZATION COMPLETE!")
        print("=" * 60)
        print(f"Phase 1 (Geographic clustering): {phase1_moves} moves")
        print(f"Phase 2 (Assign unassigned): {phase2_moves} assignments")
        print(f"Phase 3 (Tighten clusters): {phase3_moves} moves")
        print(f"Phase 4 (Consolidate drivers): {phase4_moves} moves")
        print(f"Phase 5 (Verify time windows): {phase5_moves} corrections")
        print(f"Phase 6 (Balance groups): {phase6_moves} moves")
        print(f"\n📊 Total optimizations: {total_moves}")
        
        return total_moves
    
    def phase1_geographic_clustering(self):
        """Phase 1: Create tight geographic clusters considering ALL dogs together"""
        print("\n🔗 Creating tight geographic clusters (considering all groups together)...")
        
        moves_made = 0
        
        # First, clear all driver clusters and rebuild from current assignments
        self.driver_clusters.clear()
        
        for dog in self.dog_assignments:
            if dog.get('combined') and ':' in dog['combined']:
                driver = dog['combined'].split(':')[0]
                if driver in self.active_drivers:
                    self.driver_clusters[driver].append(dog)
        
        # Iterate until no more improvements
        iteration = 0
        max_iterations = 10
        
        while iteration < max_iterations:
            iteration += 1
            iteration_moves = 0
            
            print(f"\n   Iteration {iteration}...")
            
            # For each driver, check if any dogs from other drivers are closer
            # Consider ALL dogs regardless of group - we'll preserve groups when moving
            for driver in list(self.active_drivers):
                cluster_dogs = self.driver_clusters[driver].copy()
                
                if not cluster_dogs:
                    continue
                
                # Calculate cluster center (dog with minimum avg distance to others)
                center_dog = self._find_cluster_center(cluster_dogs)
                
                if not center_dog:
                    continue
                
                center_id = center_dog.get('dog_id', '')
                
                # Check all dogs from other drivers
                dogs_to_steal = []
                
                for other_driver in self.active_drivers:
                    if other_driver == driver:
                        continue
                    
                    for dog in self.driver_clusters[other_driver]:
                        dog_id = dog.get('dog_id', '')
                        if not dog_id:
                            continue
                        
                        # Find nearest dog in current cluster
                        min_dist_to_cluster = float('inf')
                        for cluster_dog in cluster_dogs:
                            cluster_dog_id = cluster_dog.get('dog_id', '')
                            if cluster_dog_id:
                                dist = self.get_time_between_dogs(dog_id, cluster_dog_id)
                                min_dist_to_cluster = min(min_dist_to_cluster, dist)
                        
                        # Check if this dog is very close to our cluster
                        if min_dist_to_cluster <= self.CLUSTER_THRESHOLD:
                            # Find its current cluster distance
                            current_cluster_dist = self._get_min_distance_to_cluster(
                                dog, self.driver_clusters[other_driver]
                            )
                            
                            # Only steal if significantly closer to us
                            if min_dist_to_cluster < current_cluster_dist * 0.7:
                                dogs_to_steal.append({
                                    'dog': dog,
                                    'from_driver': other_driver,
                                    'distance': min_dist_to_cluster,
                                    'improvement': current_cluster_dist - min_dist_to_cluster
                                })
                
                # Sort by improvement and take best ones
                dogs_to_steal.sort(key=lambda x: x['improvement'], reverse=True)
                
                for steal_info in dogs_to_steal[:3]:  # Limit moves per iteration
                    dog = steal_info['dog']
                    from_driver = steal_info['from_driver']
                    
                    # Don't leave a driver with too few dogs
                    if len(self.driver_clusters[from_driver]) <= self.MIN_DOGS_FOR_DRIVER:
                        continue
                    
                    # Move the dog
                    self.driver_clusters[from_driver].remove(dog)
                    self.driver_clusters[driver].append(dog)
                    
                    # Update assignment - PRESERVE THE GROUP!
                    old_combined = dog.get('combined', '')
                    if ':' in old_combined:
                        group_part = old_combined.split(':', 1)[1]
                        dog['combined'] = f"{driver}:{group_part}"  # Keep same group
                    else:
                        # If no group, try to get from callout
                        callout = dog.get('callout', '')
                        groups = self.parse_dog_groups_from_callout(callout)
                        if groups:
                            dog['combined'] = f"{driver}:{groups[0]}"
                        else:
                            dog['combined'] = f"{driver}:1"  # Default
                    
                    print(f"      ✅ {dog['dog_name']}: {from_driver} → {driver} "
                          f"(improved by {steal_info['improvement']:.1f} min, "
                          f"keeping group {dog['combined'].split(':')[1]})")
                    
                    iteration_moves += 1
                    moves_made += 1
            
            print(f"   Made {iteration_moves} moves this iteration")
            
            if iteration_moves == 0:
                break
        
        # Print cluster metrics
        print("\n📊 Cluster metrics after geographic optimization:")
        for driver in self.active_drivers:
            if self.driver_clusters[driver]:
                metrics = self.calculate_cluster_metrics(self.driver_clusters[driver])
                
                # Count by group
                group_counts = {1: 0, 2: 0, 3: 0}
                for dog in self.driver_clusters[driver]:
                    combined = dog.get('combined', '')
                    if ':' in combined:
                        group_str = combined.split(':', 1)[1]
                        for g in [1, 2, 3]:
                            if str(g) in group_str:
                                group_counts[g] += 1
                                break
                
                print(f"   {driver}: {len(self.driver_clusters[driver])} dogs "
                      f"(G1:{group_counts[1]}, G2:{group_counts[2]}, G3:{group_counts[3]}), "
                      f"avg distance: {metrics['avg_distance']:.1f} min, "
                      f"cohesion: {metrics['cohesion']:.2f}")
        
        return moves_made
    
    def phase2_assign_unassigned(self):
        """Phase 2: Assign all unassigned dogs to their nearest cluster"""
        print("\n🎯 Assigning unassigned dogs to nearest clusters...")
        
        unassigned = []
        for dog in self.dog_assignments:
            if not dog.get('combined') and dog.get('callout'):
                unassigned.append(dog)
        
        print(f"   Found {len(unassigned)} unassigned dogs")
        
        assignments_made = 0
        
        for dog in unassigned:
            dog_name = dog.get('dog_name', 'Unknown')
            callout = dog.get('callout', '').strip()
            
            # Parse required groups from callout
            required_groups = self.parse_dog_groups_from_callout(callout)
            
            if not required_groups:
                print(f"   ⚠️ {dog_name} has no group in callout: '{callout}'")
                continue
            
            # Find nearest cluster
            best_driver, min_distance = self.find_nearest_cluster(dog)
            
            if best_driver and min_distance < float('inf'):
                # Assign to this driver with correct group
                group_str = ''.join(map(str, required_groups))
                dog['combined'] = f"{best_driver}:{group_str}"
                self.driver_clusters[best_driver].append(dog)
                
                print(f"   ✅ {dog_name} → {best_driver} "
                      f"(distance: {min_distance:.1f} min, groups: {group_str})")
                
                assignments_made += 1
            else:
                print(f"   ❌ Could not find cluster for {dog_name}")
        
        return assignments_made
    
    def phase3_tighten_clusters(self):
        """Phase 3: Remove outliers to create super tight clusters"""
        print("\n🎯 Tightening clusters by removing outliers...")
        
        moves_made = 0
        
        for driver in list(self.active_drivers):
            cluster = self.driver_clusters[driver]
            
            if len(cluster) < 3:
                continue
            
            print(f"\n   Analyzing {driver} cluster ({len(cluster)} dogs)...")
            
            # Find outliers
            outliers = []
            
            for dog in cluster:
                dog_id = dog.get('dog_id', '')
                if not dog_id:
                    continue
                
                # Calculate average distance to other dogs in cluster
                distances = []
                for other_dog in cluster:
                    other_id = other_dog.get('dog_id', '')
                    if other_id and other_id != dog_id:
                        dist = self.get_time_between_dogs(dog_id, other_id)
                        if dist < float('inf'):
                            distances.append(dist)
                
                if distances:
                    avg_dist = sum(distances) / len(distances)
                    min_dist = min(distances)
                    
                    # Check if outlier
                    if avg_dist > self.OUTLIER_THRESHOLD or min_dist > self.NEIGHBOR_THRESHOLD:
                        outliers.append({
                            'dog': dog,
                            'avg_distance': avg_dist,
                            'min_distance': min_dist
                        })
            
            # Sort outliers by how far they are
            outliers.sort(key=lambda x: x['avg_distance'], reverse=True)
            
            # Move outliers to better clusters
            for outlier_info in outliers:
                dog = outlier_info['dog']
                dog_name = dog.get('dog_name', 'Unknown')
                
                # Find better cluster
                best_driver, best_distance = self.find_nearest_cluster(
                    dog, exclude_drivers={driver}
                )
                
                if best_driver and best_distance < outlier_info['min_distance'] * 0.8:
                    # Move to better cluster
                    self.driver_clusters[driver].remove(dog)
                    self.driver_clusters[best_driver].append(dog)
                    
                    # Update assignment - PRESERVE GROUP!
                    old_combined = dog.get('combined', '')
                    if ':' in old_combined:
                        group_part = old_combined.split(':', 1)[1]
                        dog['combined'] = f"{best_driver}:{group_part}"
                    else:
                        # Get group from callout
                        callout = dog.get('callout', '')
                        groups = self.parse_dog_groups_from_callout(callout)
                        if groups:
                            group_str = ''.join(map(str, groups))
                            dog['combined'] = f"{best_driver}:{group_str}"
                        else:
                            dog['combined'] = f"{best_driver}:1"
                    
                    print(f"      ✅ Moved outlier {dog_name} → {best_driver} "
                          f"(improved from {outlier_info['min_distance']:.1f} to {best_distance:.1f} min, "
                          f"keeping group {dog['combined'].split(':')[1]})")
                    
                    moves_made += 1
        
        return moves_made
    
    def phase4_consolidate_small_drivers(self):
        """Phase 4: Consolidate drivers with too few dogs"""
        print("\n🔄 Consolidating small drivers...")
        
        moves_made = 0
        
        # Find drivers with too few dogs
        small_drivers = []
        for driver, dogs in self.driver_clusters.items():
            if 0 < len(dogs) < self.MIN_DOGS_FOR_DRIVER:
                small_drivers.append((driver, len(dogs)))
        
        small_drivers.sort(key=lambda x: x[1])  # Sort by count
        
        print(f"   Found {len(small_drivers)} drivers with < {self.MIN_DOGS_FOR_DRIVER} dogs")
        
        for driver, count in small_drivers:
            print(f"\n   Redistributing {driver}'s {count} dogs...")
            
            dogs_to_move = self.driver_clusters[driver].copy()
            
            for dog in dogs_to_move:
                # Find best alternative cluster
                best_driver, best_distance = self.find_nearest_cluster(
                    dog, exclude_drivers={driver}
                )
                
                if best_driver:
                    # Move dog
                    self.driver_clusters[driver].remove(dog)
                    self.driver_clusters[best_driver].append(dog)
                    
                    # Update assignment - PRESERVE GROUP!
                    old_combined = dog.get('combined', '')
                    if ':' in old_combined:
                        group_part = old_combined.split(':', 1)[1]
                        dog['combined'] = f"{best_driver}:{group_part}"
                    else:
                        # Get group from callout
                        callout = dog.get('callout', '')
                        groups = self.parse_dog_groups_from_callout(callout)
                        if groups:
                            group_str = ''.join(map(str, groups))
                            dog['combined'] = f"{best_driver}:{group_str}"
                        else:
                            dog['combined'] = f"{best_driver}:1"
                    
                    print(f"      ✅ {dog['dog_name']} → {best_driver} "
                          f"(distance: {best_distance:.1f} min, "
                          f"keeping group {dog['combined'].split(':')[1]})")
                    
                    moves_made += 1
            
            # Remove driver from active set
            self.active_drivers.discard(driver)
            del self.driver_clusters[driver]
        
        return moves_made
    
    def phase5_assign_groups(self):
        """Phase 5: Ensure all dogs have their correct group assignments"""
        print("\n🏷️ Ensuring correct group assignments based on original callouts...")
        
        corrections_made = 0
        
        # Groups are TIME WINDOWS that cannot be changed!
        # We just need to ensure the combined field has the correct group
        
        for driver, dogs in self.driver_clusters.items():
            print(f"\n   Checking {driver} ({len(dogs)} dogs)...")
            
            for dog in dogs:
                callout = dog.get('callout', '').strip()
                current_combined = dog.get('combined', '')
                
                # Parse the original group assignment from callout
                original_groups = self.parse_dog_groups_from_callout(callout)
                
                if original_groups:
                    # Dog MUST be in their original group(s)
                    group_str = ''.join(map(str, original_groups))
                    correct_combined = f"{driver}:{group_str}"
                    
                    if current_combined != correct_combined:
                        dog['combined'] = correct_combined
                        corrections_made += 1
                        print(f"      ✅ Corrected {dog['dog_name']}: {current_combined} → {correct_combined}")
                elif ':' not in current_combined:
                    # No group in callout, check if there was a group in original combined
                    # If not, this might be an error - flag it
                    print(f"      ⚠️ {dog['dog_name']} has no group assignment!")
        
        print(f"\n   ✅ Made {corrections_made} corrections to preserve time windows")
        return corrections_made
    
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
    
    def phase6_balance_groups(self):
        """Phase 6: Balance groups using cascading moves within geographic areas"""
        print("\n⚖️ Balancing time window groups with cascading...")
        
        moves_made = 0
        max_iterations = 50
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Find over-capacity groups
            over_capacity = []
            
            for driver, dogs in self.driver_clusters.items():
                # Calculate capacity
                metrics = self.calculate_cluster_metrics(dogs)
                capacity = self.CAPACITY_DENSE if metrics['avg_distance'] < self.DENSE_ROUTE_THRESHOLD else self.CAPACITY_STANDARD
                
                # Count dogs in each group
                group_counts = {1: 0, 2: 0, 3: 0}
                group_dogs = {1: [], 2: [], 3: []}
                
                for dog in dogs:
                    combined = dog.get('combined', '')
                    if ':' in combined:
                        group_str = combined.split(':', 1)[1]
                        for g in [1, 2, 3]:
                            if str(g) in group_str:
                                group_counts[g] += 1
                                group_dogs[g].append(dog)
                                break
                
                # Check each group
                for group_num, count in group_counts.items():
                    if count > capacity:
                        over_capacity.append({
                            'driver': driver,
                            'group': group_num,
                            'count': count,
                            'capacity': capacity,
                            'over_by': count - capacity,
                            'dogs': group_dogs[group_num]
                        })
            
            if not over_capacity:
                print(f"\n✅ All groups balanced after {moves_made} moves!")
                break
            
            # Sort by how much over capacity
            over_capacity.sort(key=lambda x: x['over_by'], reverse=True)
            
            # Process the most over-capacity group
            worst = over_capacity[0]
            print(f"\n   Iteration {iteration}: {worst['driver']} Group {worst['group']} "
                  f"is over by {worst['over_by']} dogs ({worst['count']}/{worst['capacity']})")
            
            # Find a dog to move (preferably one with nearby alternatives in SAME GROUP)
            best_move = None
            
            for dog in worst['dogs']:
                dog_id = dog.get('dog_id', '')
                if not dog_id:
                    continue
                
                # Find nearby dogs in SAME GROUP but different drivers
                for other_driver, other_dogs in self.driver_clusters.items():
                    if other_driver == worst['driver']:
                        continue
                    
                    # Check capacity of other driver
                    other_metrics = self.calculate_cluster_metrics(other_dogs)
                    other_capacity = self.CAPACITY_DENSE if other_metrics['avg_distance'] < self.DENSE_ROUTE_THRESHOLD else self.CAPACITY_STANDARD
                    
                    # Count dogs in this SPECIFIC GROUP for other driver
                    other_group_count = 0
                    nearby_dogs_in_group = []
                    
                    for other_dog in other_dogs:
                        other_combined = other_dog.get('combined', '')
                        if ':' in other_combined:
                            other_group_str = other_combined.split(':', 1)[1]
                            # Check if this dog is in the same group
                            if str(worst['group']) in other_group_str:
                                other_group_count += 1
                                other_id = other_dog.get('dog_id', '')
                                if other_id:
                                    dist = self.get_time_between_dogs(dog_id, other_id)
                                    if dist <= self.NEIGHBOR_THRESHOLD:
                                        nearby_dogs_in_group.append({
                                            'dog': other_dog,
                                            'distance': dist
                                        })
                    
                    # Only consider if there's capacity AND nearby dogs in same group
                    if other_group_count < other_capacity and nearby_dogs_in_group:
                        min_dist = min(d['distance'] for d in nearby_dogs_in_group)
                        
                        if not best_move or min_dist < best_move['distance']:
                            best_move = {
                                'dog': dog,
                                'to_driver': other_driver,
                                'distance': min_dist,
                                'group': worst['group'],
                                'nearby_count': len(nearby_dogs_in_group)
                            }
            
            if best_move:
                # Execute the move
                dog = best_move['dog']
                from_driver = worst['driver']
                to_driver = best_move['to_driver']
                
                # Move dog
                self.driver_clusters[from_driver].remove(dog)
                self.driver_clusters[to_driver].append(dog)
                
                # Update assignment (group stays the same!)
                dog['combined'] = f"{to_driver}:{best_move['group']}"
                
                print(f"      ✅ Moved {dog['dog_name']}: {from_driver} → {to_driver} "
                      f"(Group {best_move['group']}, distance: {best_move['distance']:.1f} min, "
                      f"{best_move['nearby_count']} nearby dogs in same group)")
                
                moves_made += 1
            else:
                print(f"      ❌ No good moves found for this group (need nearby dogs in same time window)")
                # Skip to next over-capacity group
                continue
        
        if iteration >= max_iterations:
            print(f"\n⚠️ Reached max iterations ({max_iterations})")
        
        return moves_made
    
    def _find_cluster_center(self, dogs: List[dict]) -> dict:
        """Find the dog that's most central to the cluster"""
        if not dogs:
            return None
        
        if len(dogs) == 1:
            return dogs[0]
        
        min_avg_distance = float('inf')
        center_dog = None
        
        for dog in dogs:
            dog_id = dog.get('dog_id', '')
            if not dog_id:
                continue
            
            total_distance = 0
            count = 0
            
            for other_dog in dogs:
                other_id = other_dog.get('dog_id', '')
                if other_id and other_id != dog_id:
                    dist = self.get_time_between_dogs(dog_id, other_id)
                    if dist < float('inf'):
                        total_distance += dist
                        count += 1
            
            if count > 0:
                avg_distance = total_distance / count
                if avg_distance < min_avg_distance:
                    min_avg_distance = avg_distance
                    center_dog = dog
        
        return center_dog
    
    def _get_min_distance_to_cluster(self, dog: dict, cluster: List[dict]) -> float:
        """Get minimum distance from a dog to any dog in a cluster"""
        dog_id = dog.get('dog_id', '')
        if not dog_id:
            return float('inf')
        
        min_dist = float('inf')
        for cluster_dog in cluster:
            cluster_dog_id = cluster_dog.get('dog_id', '')
            if cluster_dog_id and cluster_dog_id != dog_id:
                dist = self.get_time_between_dogs(dog_id, cluster_dog_id)
                min_dist = min(min_dist, dist)
        
        return min_dist
    
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
                    col_letter = chr(ord('A') + combined_col_idx - 1)
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
            print("   Attempting individual updates...")
            updated = 0
            for assignment in self.dog_assignments:
                if isinstance(assignment, dict) and 'row_index' in assignment:
                    try:
                        row_idx = assignment['row_index']
                        combined_value = assignment.get('combined', '')
                        self.map_sheet.update_cell(row_idx, 8, combined_value)
                        updated += 1
                    except:
                        pass
            print(f"   ✅ Updated {updated} assignments individually")
    
    def print_final_summary(self):
        """Print a final summary of the optimization"""
        print("\n" + "=" * 60)
        print("📊 FINAL CLUSTER SUMMARY")
        print("=" * 60)
        
        total_dogs = 0
        
        for driver in sorted(self.active_drivers):
            dogs = self.driver_clusters[driver]
            if not dogs:
                continue
            
            total_dogs += len(dogs)
            metrics = self.calculate_cluster_metrics(dogs)
            
            # Count by group
            group_counts = {1: 0, 2: 0, 3: 0}
            for dog in dogs:
                combined = dog.get('combined', '')
                if ':' in combined:
                    group_str = combined.split(':', 1)[1]
                    for g in [1, 2, 3]:
                        if str(g) in group_str:
                            group_counts[g] += 1
                            break
            
            print(f"\n{driver}:")
            print(f"   Total dogs: {len(dogs)}")
            print(f"   Groups: G1={group_counts[1]}, G2={group_counts[2]}, G3={group_counts[3]}")
            print(f"   Avg distance: {metrics['avg_distance']:.1f} min")
            print(f"   Max distance: {metrics['max_distance']:.1f} min")
            print(f"   Cohesion score: {metrics['cohesion']:.2f}")
            
            # Check capacity
            capacity = self.CAPACITY_DENSE if metrics['avg_distance'] < self.DENSE_ROUTE_THRESHOLD else self.CAPACITY_STANDARD
            for g, count in group_counts.items():
                if count > capacity:
                    print(f"   ⚠️ Group {g} is over capacity ({count}/{capacity})")
        
        print(f"\n📊 Total dogs assigned: {total_dogs}")
        
        # Count unassigned
        unassigned = sum(1 for d in self.dog_assignments 
                        if not d.get('combined') and d.get('callout'))
        if unassigned > 0:
            print(f"⚠️ Unassigned dogs remaining: {unassigned}")
    
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
            self.print_final_summary()
            
            print("\n✅ OPTIMIZATION COMPLETE!")
            
            if self.haversine_fallback_count > 0:
                print(f"\n📍 Note: Used haversine fallback {self.haversine_fallback_count} times")
                print("   Consider updating distance matrix for better accuracy")
            
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
