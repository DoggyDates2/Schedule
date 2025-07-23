#!/usr/bin/env python3
"""
Chain-Based Dog Clustering System

A fresh approach that builds optimal pickup/dropoff chains from scratch.
Instead of fighting existing assignments, we create ideal routes considering:
- Group 1 dropoff → Group 2 pickup transitions
- Group 2 dropoff → Group 3 pickup transitions

DATA SETUP:
1. In your Map sheet, mark every dog's groups in the Callout column (K):
   - :1 (just group 1)
   - :23 (groups 2 and 3)  
   - :123 (all three groups)
2. Leave the Combined column (H) completely blank
3. Run this script - it will fill Combined with optimized assignments:
   - Driver_01:123
   - Driver_02:1
   - etc.

CAPACITY RULES:
- Each group (G1, G2, G3) has its own capacity limit
- Dense routes (avg < 2 min between dogs): 12 dogs per group
- Standard routes: 8 dogs per group  
- A driver might have: G1=12, G2=12, G3=12 (36 total) for dense routes
- Or: G1=8, G2=8, G3=8 (24 total) for standard routes

MINIMUM GROUP SIZE:
- Each group must have at least 4 dogs to be worthwhile
- Groups with < 4 dogs get consolidated to other drivers
- This prevents paying drivers for tiny pickups/dropoffs

MUCH simpler than the 7-phase approach!
"""

import os
import sys
import json
import time
import math
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set

try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    import requests
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics.pairwise import pairwise_distances
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("💡 Install with: pip install gspread oauth2client requests scikit-learn numpy")
    sys.exit(1)


@dataclass
class Dog:
    """Represents a dog with its requirements"""
    dog_id: str
    name: str
    lat: float
    lng: float
    groups_needed: List[int]  # [1], [2], [3], [1,2], [2,3], or [1,2,3]
    address: str = ""
    
    def needs_group(self, group_num: int) -> bool:
        return group_num in self.groups_needed


@dataclass
class RouteChain:
    """Represents a complete route chain for a driver"""
    driver_id: str
    group1_dogs: List[Dog]
    group2_dogs: List[Dog]
    group3_dogs: List[Dog]
    
    def __post_init__(self):
        """Ensure lists are never None"""
        if self.group1_dogs is None:
            self.group1_dogs = []
        if self.group2_dogs is None:
            self.group2_dogs = []
        if self.group3_dogs is None:
            self.group3_dogs = []
    
    @property
    def total_dogs(self) -> int:
        # Count unique dogs (some may appear in multiple groups)
        all_dogs = set()
        for dog in self.group1_dogs + self.group2_dogs + self.group3_dogs:
            if dog:  # Safety check
                all_dogs.add(dog.dog_id)
        return len(all_dogs)
    
    @property
    def has_all_groups(self) -> bool:
        return len(self.group1_dogs) > 0 and len(self.group2_dogs) > 0 and len(self.group3_dogs) > 0


class ChainBasedClusteringSystem:
    def __init__(self):
        # Parameters (in minutes)
        self.DENSE_THRESHOLD = 2.0  # Routes < 2 min avg get 12 dogs
        self.STANDARD_CAPACITY = 8
        self.DENSE_CAPACITY = 12
        self.MIN_GROUP_SIZE = 4     # Minimum dogs to make a group worthwhile
        self.MAX_ROUTE_SPAN = 10.0  # Maximum minutes across a route
        self.TRANSITION_WEIGHT = 0.5  # How much G1→G2 and G2→G3 transitions matter
        self.MAX_CONSOLIDATION_ITERATIONS = 10  # Prevent infinite loops
        
        print("🚀 Chain-Based Dog Clustering System")
        print("   Building optimal routes from scratch!")
        print("   Considering G1→G2 and G2→G3 transitions")
        print(f"   Minimum {self.MIN_GROUP_SIZE} dogs per group (or group gets consolidated)")
        print(f"   Capacity: Dense routes={self.DENSE_CAPACITY}, Standard={self.STANDARD_CAPACITY}")
        
        # Google Sheets IDs
        self.MAP_SHEET_ID = "1-KTOfTKXk_sX7nO7eGmW73JLi8TJBvv5gobK6gyrc7U"
        self.DISTANCE_MATRIX_SHEET_ID = "1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg"
        
        # Data storage
        self.dogs: List[Dog] = []
        self.distance_matrix: Dict[str, Dict[str, float]] = {}
        self.routes: List[RouteChain] = []
        
        # Setup
        self.setup_google_sheets()
        self.load_data()
    
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
    
    def load_data(self):
        """Load dogs and distance matrix from Google Sheets"""
        try:
            self.load_distance_matrix()
            self.load_dogs()
            
            if not self.dogs:
                print("❌ No dogs loaded - please check your data")
                return
                
            print(f"\n📊 Loaded {len(self.dogs)} dogs")
            
            # Count dogs by group needs
            group_counts = defaultdict(int)
            for dog in self.dogs:
                group_counts[tuple(sorted(dog.groups_needed))] += 1
            
            print("\n📊 Dogs by group needs:")
            for groups, count in sorted(group_counts.items()):
                groups_str = ','.join(map(str, groups))
                print(f"   Groups [{groups_str}]: {count} dogs")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
    
    def load_distance_matrix(self):
        """Load the distance/time matrix"""
        try:
            spreadsheet = self.gc.open_by_key(self.DISTANCE_MATRIX_SHEET_ID)
            
            # Try different sheet names
            sheet = None
            for sheet_name in ["Matrix", "Time Matrix", "Distance Matrix"]:
                try:
                    sheet = spreadsheet.worksheet(sheet_name)
                    print(f"✅ Found sheet: '{sheet_name}'")
                    break
                except gspread.WorksheetNotFound:
                    continue
            
            if not sheet:
                print("❌ Could not find distance matrix sheet")
                return
            
            all_values = sheet.get_all_values()
            
            if not all_values or len(all_values) < 2:
                print("❌ Distance matrix is empty or too small")
                return
            
            # Parse matrix - handle potential empty cells
            header_dogs = []
            for h in all_values[0][1:]:
                if h and h.strip():
                    header_dogs.append(h.strip())
            
            if not header_dogs:
                print("❌ No column headers found in distance matrix")
                return
            
            distances_loaded = 0
            for row_idx, row in enumerate(all_values[1:], 1):
                if not row or len(row) < 2:
                    continue
                
                from_dog = row[0].strip() if row[0] else ""
                if not from_dog:
                    continue
                
                if from_dog not in self.distance_matrix:
                    self.distance_matrix[from_dog] = {}
                
                for col_idx, value in enumerate(row[1:]):
                    if col_idx >= len(header_dogs):
                        break
                        
                    to_dog = header_dogs[col_idx]
                    
                    if value and value.strip():
                        try:
                            time_min = float(value.strip())
                            if time_min >= 0:  # Validate non-negative
                                self.distance_matrix[from_dog][to_dog] = time_min
                                distances_loaded += 1
                        except ValueError:
                            # Skip invalid values
                            continue
            
            print(f"✅ Loaded {distances_loaded} distance values")
            print(f"✅ Matrix has {len(self.distance_matrix)} dogs with time data")
            
            if self.distance_matrix and len(self.distance_matrix) > 0:
                sample_dog = list(self.distance_matrix.keys())[0]
                sample_count = len(self.distance_matrix[sample_dog])
                print(f"✅ Sample: {sample_dog} has distances to {sample_count} other dogs")
                
        except Exception as e:
            print(f"❌ Error loading distance matrix: {e}")
            import traceback
            traceback.print_exc()
            self.distance_matrix = {}
    
    def load_dogs(self):
        """Load dogs from the Map sheet"""
        try:
            sheet = self.gc.open_by_key(self.MAP_SHEET_ID).worksheet("Map")
            all_values = sheet.get_all_values()
            
            if not all_values:
                print("❌ No dog data found")
                return
            
            # Column indices (matching existing sheet structure - 0-indexed)
            dog_name_idx = 1    # Column B - Dog name
            address_idx = 2     # Column C - Address  
            lat_idx = 3         # Column D - Latitude
            lng_idx = 4         # Column E - Longitude
            combined_idx = 7    # Column H - Combined (will be blank, we'll fill this)
            dog_id_idx = 9      # Column J - Dog ID
            callout_idx = 10    # Column K - Callout (groups needed, e.g. ":123")
            
            skipped_count = 0
            
            for row_idx, row in enumerate(all_values[1:], 2):  # Skip header, start at row 2
                if len(row) <= max(dog_name_idx, lat_idx, lng_idx, dog_id_idx, callout_idx):
                    continue
                
                try:
                    dog_name = row[dog_name_idx].strip() if row[dog_name_idx] else ""
                    dog_id = row[dog_id_idx].strip() if row[dog_id_idx] else ""
                    
                    # Parse coordinates
                    lat = None
                    lng = None
                    if row[lat_idx].strip() and row[lng_idx].strip():
                        try:
                            lat = float(row[lat_idx].strip())
                            lng = float(row[lng_idx].strip())
                            
                            # Validate coordinates
                            if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                                lat, lng = None, None
                        except ValueError:
                            pass
                    
                    callout = row[callout_idx].strip() if row[callout_idx] else ""
                    address = row[address_idx].strip() if len(row) > address_idx else ""
                    
                    # Skip if missing required fields
                    if not dog_name or not dog_id or lat is None or lng is None:
                        if dog_name:  # Only report if there's a name but missing other data
                            print(f"⚠️  Skipping {dog_name}: missing {'ID' if not dog_id else 'coordinates'}")
                        skipped_count += 1
                        continue
                    
                    # Parse groups needed from callout (e.g., ":1", ":23", ":123")
                    groups_needed = []
                    if callout:
                        # Remove leading colon if present
                        callout_clean = callout.lstrip(':')
                        # Parse each character that's a valid group number
                        for char in callout_clean:
                            if char in ['1', '2', '3']:
                                group_num = int(char)
                                if group_num not in groups_needed:
                                    groups_needed.append(group_num)
                    
                    if not groups_needed:
                        print(f"⚠️  Skipping {dog_name}: no valid groups in callout '{callout}'")
                        skipped_count += 1
                        continue  # Skip dogs with no group assignments
                    
                    # Create dog object
                    dog = Dog(
                        dog_id=dog_id,
                        name=dog_name,
                        lat=lat,
                        lng=lng,
                        groups_needed=sorted(groups_needed),
                        address=address
                    )
                    self.dogs.append(dog)
                    
                except (ValueError, IndexError) as e:
                    skipped_count += 1
                    continue
            
            print(f"✅ Loaded {len(self.dogs)} dogs from Map sheet")
            if skipped_count > 0:
                print(f"⚠️  Skipped {skipped_count} rows (missing data or invalid format)")
            
            # Since combined column is blank, all dogs are unassigned - perfect!
            print("✅ All dogs marked as unassigned (combined column blank)")
            
        except Exception as e:
            print(f"❌ Error loading dogs: {e}")
            import traceback
            traceback.print_exc()
    
    def get_distance(self, dog1: Dog, dog2: Dog) -> float:
        """Get distance between two dogs (with haversine fallback)"""
        if not dog1 or not dog2:
            return float('inf')
            
        if dog1.dog_id == dog2.dog_id:
            return 0.0
        
        # Try multiple ID formats to handle 'x' suffix variations
        id1_variants = [dog1.dog_id]
        id2_variants = [dog2.dog_id]
        
        # Add variants without 'x' if present
        if dog1.dog_id.endswith('x'):
            id1_variants.append(dog1.dog_id[:-1])
        else:
            id1_variants.append(f"{dog1.dog_id}x")
            
        if dog2.dog_id.endswith('x'):
            id2_variants.append(dog2.dog_id[:-1])
        else:
            id2_variants.append(f"{dog2.dog_id}x")
        
        # Try matrix first with all ID variants
        for id1 in id1_variants:
            for id2 in id2_variants:
                if id1 in self.distance_matrix and id2 in self.distance_matrix[id1]:
                    distance = self.distance_matrix[id1][id2]
                    if 0 <= distance < float('inf'):  # Validate distance
                        return distance
                if id2 in self.distance_matrix and id1 in self.distance_matrix[id2]:
                    distance = self.distance_matrix[id2][id1]
                    if 0 <= distance < float('inf'):  # Validate distance
                        return distance
        
        # Fallback to haversine
        return self.haversine_distance(dog1, dog2)
    
    def haversine_distance(self, dog1: Dog, dog2: Dog) -> float:
        """Calculate distance using haversine formula"""
        # Check for same location
        if dog1.lat == dog2.lat and dog1.lng == dog2.lng:
            return 0.0
        
        try:
            # Convert to radians
            lat1, lon1 = math.radians(dog1.lat), math.radians(dog1.lng)
            lat2, lon2 = math.radians(dog2.lat), math.radians(dog2.lng)
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            
            # Clamp to valid range to avoid math domain errors
            a = max(0.0, min(1.0, a))
            
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            
            # Distance in miles, then convert to minutes
            distance_miles = 3959 * c
            driving_minutes = distance_miles * 1.3 * 2.5  # 1.3 for roads, 2.5 min/mile
            
            return max(0.0, driving_minutes)
        except Exception as e:
            print(f"⚠️  Error calculating distance: {e}")
            return float('inf')
    
    def build_transition_aware_chains(self):
        """Build route chains considering G1→G2 and G2→G3 transitions"""
        print("\n🔗 Building transition-aware chains...")
        
        # Separate dogs by groups they need
        dogs_by_group = {
            1: [d for d in self.dogs if d.needs_group(1)],
            2: [d for d in self.dogs if d.needs_group(2)],
            3: [d for d in self.dogs if d.needs_group(3)]
        }
        
        print(f"📊 Dogs per group: G1={len(dogs_by_group[1])}, "
              f"G2={len(dogs_by_group[2])}, G3={len(dogs_by_group[3])}")
        
        # Check if we have enough dogs
        if not any(dogs_by_group.values()):
            print("❌ No dogs found in any group!")
            return []
        
        # Strategy: Start with Group 2 (middle of chain) and build outward
        used_dogs = set()
        chains = []
        
        # If no G2 dogs, try different strategy
        if not dogs_by_group[2]:
            print("⚠️  No Group 2 dogs - using alternative clustering strategy")
            return self.build_chains_without_g2(dogs_by_group)
        
        # Build clusters starting from Group 2
        g2_clusters = self.cluster_dogs_with_capacity(dogs_by_group[2], eps_minutes=2.0)
        
        print(f"\n📊 Created {len(g2_clusters)} Group 2 clusters")
        
        # For each G2 cluster, find matching G1 and G3 dogs
        max_chains = 100  # Sanity check - prevent infinite chains
        
        for cluster_idx, g2_cluster in enumerate(g2_clusters):
            if cluster_idx >= max_chains:
                print(f"⚠️  Reached maximum chains limit ({max_chains})")
                break
                
            if not g2_cluster:
                continue
            
            print(f"\n🔍 Processing G2 cluster {cluster_idx + 1} ({len(g2_cluster)} dogs)")
            
            # Determine capacity based on cluster density
            # Calculate average distance within G2 cluster to determine if it's dense
            if len(g2_cluster) > 1:
                cluster_distances = []
                for i in range(len(g2_cluster)):
                    for j in range(i + 1, len(g2_cluster)):
                        dist = self.get_distance(g2_cluster[i], g2_cluster[j])
                        if dist < float('inf'):
                            cluster_distances.append(dist)
                
                if cluster_distances:
                    avg_cluster_distance = sum(cluster_distances) / len(cluster_distances)
                    is_dense = avg_cluster_distance < self.DENSE_THRESHOLD
                    capacity = self.DENSE_CAPACITY if is_dense else self.STANDARD_CAPACITY
                else:
                    capacity = self.STANDARD_CAPACITY
            else:
                capacity = self.STANDARD_CAPACITY
            
            print(f"   Capacity: {capacity} dogs per group ({'DENSE' if capacity == self.DENSE_CAPACITY else 'STANDARD'} route)")
            
            # Ensure G2 cluster itself doesn't exceed capacity
            if len(g2_cluster) > capacity:
                print(f"   ⚠️  G2 cluster has {len(g2_cluster)} dogs, exceeds capacity of {capacity}")
                # Take only up to capacity
                g2_cluster = g2_cluster[:capacity]
            
            # Find center of G2 cluster
            if not g2_cluster:  # Safety check after potential truncation
                continue
                
            g2_center_lat = sum(d.lat for d in g2_cluster) / len(g2_cluster)
            g2_center_lng = sum(d.lng for d in g2_cluster) / len(g2_cluster)
            
            # For G2 dogs that also need G1 or G3, ensure they appear in those groups too
            g1_from_g2 = [d for d in g2_cluster if d.needs_group(1) and d.dog_id not in used_dogs]
            g3_from_g2 = [d for d in g2_cluster if d.needs_group(3) and d.dog_id not in used_dogs]
            
            # Find additional G1 dogs needed (respecting capacity)
            available_g1 = [d for d in dogs_by_group[1] if d.dog_id not in used_dogs and d not in g2_cluster]
            g1_slots_available = max(0, capacity - len(g1_from_g2))
            target_g1_count = min(g1_slots_available, len(available_g1))
            additional_g1 = self.find_nearby_dogs(
                available_g1, g2_center_lat, g2_center_lng, 
                target_count=target_g1_count
            )
            g1_candidates = g1_from_g2 + additional_g1
            
            # Ensure G1 doesn't exceed capacity
            if len(g1_candidates) > capacity:
                g1_candidates = g1_candidates[:capacity]
            
            # Find additional G3 dogs needed (respecting capacity)
            available_g3 = [d for d in dogs_by_group[3] if d.dog_id not in used_dogs and d not in g2_cluster]
            g3_slots_available = max(0, capacity - len(g3_from_g2))
            target_g3_count = min(g3_slots_available, len(available_g3))
            additional_g3 = self.find_nearby_dogs(
                available_g3, g2_center_lat, g2_center_lng,
                target_count=target_g3_count
            )
            g3_candidates = g3_from_g2 + additional_g3
            
            # Ensure G3 doesn't exceed capacity
            if len(g3_candidates) > capacity:
                g3_candidates = g3_candidates[:capacity]
            
            # Only create chain if it has at least one group with dogs
            if g1_candidates or g2_cluster or g3_candidates:
                chain = RouteChain(
                    driver_id=f"Driver_{cluster_idx + 1:02d}",
                    group1_dogs=g1_candidates,
                    group2_dogs=g2_cluster,
                    group3_dogs=g3_candidates
                )
                
                # Mark dogs as used (only mark each unique dog once)
                unique_dogs = set()
                for dog in chain.group1_dogs + chain.group2_dogs + chain.group3_dogs:
                    unique_dogs.add(dog.dog_id)
                used_dogs.update(unique_dogs)
                
                chains.append(chain)
                
                # Calculate transition efficiency
                if chain.group1_dogs and chain.group2_dogs:
                    g1_to_g2_avg = self.calculate_transition_distance(chain.group1_dogs, chain.group2_dogs)
                else:
                    g1_to_g2_avg = 0.0
                    
                if chain.group2_dogs and chain.group3_dogs:
                    g2_to_g3_avg = self.calculate_transition_distance(chain.group2_dogs, chain.group3_dogs)
                else:
                    g2_to_g3_avg = 0.0
                
                print(f"   ✅ Created chain: G1={len(chain.group1_dogs)}, "
                      f"G2={len(chain.group2_dogs)}, G3={len(chain.group3_dogs)}")
                if g1_to_g2_avg > 0 or g2_to_g3_avg > 0:
                    print(f"   📍 Transition distances: G1→G2={g1_to_g2_avg:.1f}min, "
                          f"G2→G3={g2_to_g3_avg:.1f}min")
        
        # Handle remaining dogs
        remaining_dogs = [d for d in self.dogs if d.dog_id not in used_dogs]
        if remaining_dogs:
            print(f"\n⚠️  {len(remaining_dogs)} dogs not yet assigned - handling stragglers...")
            self.handle_remaining_dogs(chains, remaining_dogs, used_dogs)
        
        return chains
    
    def build_chains_without_g2(self, dogs_by_group: Dict[int, List[Dog]]) -> List[RouteChain]:
        """Alternative strategy when there are no Group 2 dogs"""
        chains = []
        used_dogs = set()
        
        # Try to pair G1 and G3 dogs geographically
        if dogs_by_group[1] and dogs_by_group[3]:
            g1_clusters = self.cluster_dogs_with_capacity(dogs_by_group[1], eps_minutes=2.0)
            
            for idx, g1_cluster in enumerate(g1_clusters):
                # Find center of G1 cluster
                center_lat = sum(d.lat for d in g1_cluster) / len(g1_cluster)
                center_lng = sum(d.lng for d in g1_cluster) / len(g1_cluster)
                
                # Determine capacity based on G1 cluster density
                capacity = self.STANDARD_CAPACITY
                if len(g1_cluster) > 1:
                    distances = []
                    for i in range(min(5, len(g1_cluster))):
                        for j in range(i + 1, min(5, len(g1_cluster))):
                            dist = self.get_distance(g1_cluster[i], g1_cluster[j])
                            if dist < float('inf'):
                                distances.append(dist)
                    
                    if distances:
                        avg_distance = sum(distances) / len(distances)
                        capacity = self.DENSE_CAPACITY if avg_distance < self.DENSE_THRESHOLD else self.STANDARD_CAPACITY
                
                # Find nearby G3 dogs (respecting capacity)
                available_g3 = [d for d in dogs_by_group[3] if d.dog_id not in used_dogs]
                target_count = min(len(g1_cluster), capacity, len(available_g3))
                g3_dogs = self.find_nearby_dogs(available_g3, center_lat, center_lng, target_count)
                
                chain = RouteChain(
                    driver_id=f"Driver_{idx + 1:02d}",
                    group1_dogs=g1_cluster,
                    group2_dogs=[],
                    group3_dogs=g3_dogs
                )
                
                for dog in chain.group1_dogs + chain.group3_dogs:
                    used_dogs.add(dog.dog_id)
                
                chains.append(chain)
        
        # Handle any G2 or G3 only dogs
        remaining_g2 = [d for d in dogs_by_group.get(2, []) if d.dog_id not in used_dogs]
        remaining_g3 = [d for d in dogs_by_group.get(3, []) if d.dog_id not in used_dogs]
        
        if remaining_g2 or remaining_g3:
            # Cluster remaining dogs
            if remaining_g2:
                g2_clusters = self.cluster_dogs_with_capacity(remaining_g2, eps_minutes=2.0)
                for idx, cluster in enumerate(g2_clusters):
                    chain = RouteChain(
                        driver_id=f"Driver_{len(chains) + idx + 1:02d}",
                        group1_dogs=[],
                        group2_dogs=cluster,
                        group3_dogs=[]
                    )
                    chains.append(chain)
            
            if remaining_g3:
                g3_clusters = self.cluster_dogs_with_capacity(remaining_g3, eps_minutes=2.0)
                for idx, cluster in enumerate(g3_clusters):
                    chain = RouteChain(
                        driver_id=f"Driver_{len(chains) + idx + 1:02d}",
                        group1_dogs=[],
                        group2_dogs=[],
                        group3_dogs=cluster
                    )
                    chains.append(chain)
        
        return chains
    
    def cluster_dogs_with_capacity(self, dogs: List[Dog], eps_minutes: float = 2.0) -> List[List[Dog]]:
        """Cluster dogs using DBSCAN with capacity limits"""
        if not dogs:
            return []
        
        # First, get initial clusters without capacity limits
        initial_clusters = self.cluster_dogs_dbscan(dogs, eps_minutes)
        
        # Now split clusters that exceed capacity
        final_clusters = []
        
        for cluster in initial_clusters:
            if not cluster:  # Skip empty clusters
                continue
                
            if len(cluster) <= self.STANDARD_CAPACITY:
                # Small enough for standard capacity
                final_clusters.append(cluster)
            else:
                # Check if this is a dense cluster
                capacity = self.STANDARD_CAPACITY  # Default
                
                if len(cluster) > 1:
                    distances = []
                    # Sample distances to avoid O(n²) for large clusters
                    sample_size = min(20, len(cluster))
                    for i in range(sample_size):
                        for j in range(i + 1, sample_size):
                            try:
                                dist = self.get_distance(cluster[i], cluster[j])
                                if dist < float('inf'):
                                    distances.append(dist)
                            except Exception:
                                continue
                    
                    if distances:
                        avg_distance = sum(distances) / len(distances)
                        capacity = self.DENSE_CAPACITY if avg_distance < self.DENSE_THRESHOLD else self.STANDARD_CAPACITY
                
                # Split cluster if it exceeds capacity
                if len(cluster) <= capacity:
                    final_clusters.append(cluster)
                else:
                    # Split into capacity-sized chunks, keeping nearby dogs together
                    # Sort dogs by location to keep geographic proximity
                    try:
                        # Simple geographic sort by latitude then longitude
                        sorted_cluster = sorted(cluster, key=lambda d: (d.lat, d.lng))
                    except Exception:
                        sorted_cluster = cluster
                    
                    for i in range(0, len(sorted_cluster), capacity):
                        final_clusters.append(sorted_cluster[i:i + capacity])
        
        return final_clusters
    
    def cluster_dogs_dbscan(self, dogs: List[Dog], eps_minutes: float = 2.0) -> List[List[Dog]]:
        """Cluster dogs using DBSCAN based on driving time"""
        if not dogs:
            return []
        
        if len(dogs) == 1:
            # Single dog becomes its own cluster
            return [dogs]
        
        if len(dogs) == 2:
            # Two dogs - check if they should cluster together
            dist = self.get_distance(dogs[0], dogs[1])
            if dist <= eps_minutes:
                return [dogs]  # One cluster with both
            else:
                return [[dogs[0]], [dogs[1]]]  # Two separate clusters
        
        # For 3+ dogs, use DBSCAN
        n = len(dogs)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i][j] = self.get_distance(dogs[i], dogs[j])
        
        # Run DBSCAN with min_samples=2 for small datasets
        min_samples = min(3, max(2, n // 4))  # Adaptive min_samples
        clustering = DBSCAN(eps=eps_minutes, min_samples=min_samples, metric='precomputed')
        labels = clustering.fit_predict(distances)
        
        # Group dogs by cluster
        clusters = defaultdict(list)
        outliers = []
        
        for dog, label in zip(dogs, labels):
            if label != -1:  # -1 means noise/outlier
                clusters[label].append(dog)
            else:
                outliers.append(dog)
        
        # Handle outliers by assigning to nearest cluster or creating new ones
        for outlier in outliers:
            best_cluster = None
            best_distance = float('inf')
            
            for label, cluster_dogs in clusters.items():
                if not cluster_dogs:  # Skip empty clusters
                    continue
                    
                # Calculate average distance to dogs in this cluster
                total_dist = 0
                count = 0
                for dog in cluster_dogs[:5]:  # Sample for efficiency
                    dist = self.get_distance(outlier, dog)
                    if dist < float('inf'):
                        total_dist += dist
                        count += 1
                
                if count > 0:
                    avg_distance = total_dist / count
                    if avg_distance < best_distance and avg_distance <= 5.0:
                        best_distance = avg_distance
                        best_cluster = label
            
            if best_cluster is not None:
                clusters[best_cluster].append(outlier)
            else:
                # Create new cluster for distant outlier
                new_label = max(clusters.keys()) + 1 if clusters else 0
                clusters[new_label] = [outlier]
        
        # Convert to list and filter out empty clusters
        return [dogs for dogs in clusters.values() if dogs]
    
    def find_nearby_dogs(self, candidates: List[Dog], center_lat: float, 
                        center_lng: float, target_count: int) -> List[Dog]:
        """Find the nearest dogs to a geographic center"""
        if not candidates:
            return []
        
        # Limit target count to available candidates
        target_count = min(target_count, len(candidates))
        if target_count <= 0:
            return []
        
        # Calculate distances from center using haversine directly
        dogs_with_distance = []
        for dog in candidates:
            # Direct haversine calculation to avoid matrix lookup issues
            try:
                lat1, lon1 = math.radians(center_lat), math.radians(center_lng)
                lat2, lon2 = math.radians(dog.lat), math.radians(dog.lng)
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                a = max(0.0, min(1.0, a))
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                
                distance_miles = 3959 * c
                driving_minutes = distance_miles * 1.3 * 2.5
                
                dogs_with_distance.append((dog, driving_minutes))
            except Exception:
                # If calculation fails, use a high distance
                dogs_with_distance.append((dog, 999.0))
        
        # Sort by distance and take target_count
        dogs_with_distance.sort(key=lambda x: x[1])
        selected = [dog for dog, _ in dogs_with_distance[:target_count]]
        
        return selected
    
    def calculate_transition_distance(self, from_dogs: List[Dog], to_dogs: List[Dog]) -> float:
        """Calculate average transition distance between groups"""
        if not from_dogs or not to_dogs:
            return 0.0
        
        total_distance = 0
        count = 0
        
        # For each from_dog, find distance to nearest to_dog
        for from_dog in from_dogs:
            min_dist = float('inf')
            for to_dog in to_dogs:
                try:
                    dist = self.get_distance(from_dog, to_dog)
                    if dist < min_dist and dist < float('inf'):
                        min_dist = dist
                except Exception as e:
                    print(f"⚠️  Error calculating distance: {e}")
                    continue
            
            if min_dist < float('inf'):
                total_distance += min_dist
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def handle_remaining_dogs(self, chains: List[RouteChain], remaining_dogs: List[Dog], used_dogs: Set[str]):
        """Handle dogs that weren't assigned in the initial clustering"""
        if not remaining_dogs:
            return
            
        print("\n🔧 Handling remaining dogs...")
        
        dogs_handled = 0
        
        for dog in remaining_dogs:
            if not dog or not hasattr(dog, 'dog_id'):
                continue
                
            # Skip if already used (safety check)
            if dog.dog_id in used_dogs:
                continue
                
            best_chain = None
            best_score = float('inf')
            
            # Find best existing chain for this dog
            for chain in chains:
                if not chain:
                    continue
                    
                # Calculate route density to determine capacity
                all_chain_dogs = []
                for d in (chain.group1_dogs or []) + (chain.group2_dogs or []) + (chain.group3_dogs or []):
                    if d and d.dog_id not in [existing.dog_id for existing in all_chain_dogs]:
                        all_chain_dogs.append(d)
                
                if len(all_chain_dogs) > 1:
                    distances = []
                    sample_size = min(5, len(all_chain_dogs))
                    for i in range(sample_size):
                        for j in range(i + 1, sample_size):
                            dist = self.get_distance(all_chain_dogs[i], all_chain_dogs[j])
                            if 0 <= dist < float('inf'):
                                distances.append(dist)
                    
                    if distances:
                        avg_distance = sum(distances) / len(distances)
                        capacity = self.DENSE_CAPACITY if avg_distance < self.DENSE_THRESHOLD else self.STANDARD_CAPACITY
                    else:
                        capacity = self.STANDARD_CAPACITY
                else:
                    capacity = self.STANDARD_CAPACITY
                
                # Check if chain has room for this dog in the groups it needs
                can_add = True
                for group_num in dog.groups_needed:
                    if group_num == 1:
                        current_count = len(chain.group1_dogs) if chain.group1_dogs else 0
                        if current_count >= capacity:
                            can_add = False
                            break
                    elif group_num == 2:
                        current_count = len(chain.group2_dogs) if chain.group2_dogs else 0
                        if current_count >= capacity:
                            can_add = False
                            break
                    elif group_num == 3:
                        current_count = len(chain.group3_dogs) if chain.group3_dogs else 0
                        if current_count >= capacity:
                            can_add = False
                            break
                
                if not can_add:
                    continue
                
                # Calculate average distance to relevant groups
                total_distance = 0
                count = 0
                
                if dog.needs_group(1) and chain.group1_dogs:
                    for other_dog in chain.group1_dogs[:5]:  # Check first 5 for efficiency
                        dist = self.get_distance(dog, other_dog)
                        if 0 <= dist < float('inf'):
                            total_distance += dist
                            count += 1
                
                if dog.needs_group(2) and chain.group2_dogs:
                    for other_dog in chain.group2_dogs[:5]:
                        dist = self.get_distance(dog, other_dog)
                        if 0 <= dist < float('inf'):
                            total_distance += dist
                            count += 1
                
                if dog.needs_group(3) and chain.group3_dogs:
                    for other_dog in chain.group3_dogs[:5]:
                        dist = self.get_distance(dog, other_dog)
                        if 0 <= dist < float('inf'):
                            total_distance += dist
                            count += 1
                
                if count > 0:
                    avg_distance = total_distance / count
                    if avg_distance < best_score:
                        best_score = avg_distance
                        best_chain = chain
            
            # Add to best chain or create new one
            dog_name = dog.name if hasattr(dog, 'name') else dog.dog_id
            
            if best_chain and best_score < 10.0:  # Only if reasonably close
                if dog.needs_group(1):
                    best_chain.group1_dogs.append(dog)
                if dog.needs_group(2):
                    best_chain.group2_dogs.append(dog)
                if dog.needs_group(3):
                    best_chain.group3_dogs.append(dog)
                
                used_dogs.add(dog.dog_id)
                dogs_handled += 1
                print(f"   ✅ Added {dog_name} to {best_chain.driver_id} (distance: {best_score:.1f}min)")
            else:
                # Create new chain if no suitable existing chain
                new_chain_idx = len(chains) + 1
                new_chain = RouteChain(
                    driver_id=f"Driver_{new_chain_idx:02d}",
                    group1_dogs=[dog] if dog.needs_group(1) else [],
                    group2_dogs=[dog] if dog.needs_group(2) else [],
                    group3_dogs=[dog] if dog.needs_group(3) else []
                )
                chains.append(new_chain)
                used_dogs.add(dog.dog_id)
                dogs_handled += 1
                print(f"   ✅ Created new chain {new_chain.driver_id} for {dog_name}")
        
        print(f"   Handled {dogs_handled} remaining dogs")
    
    def consolidate_small_groups(self, chains: List[RouteChain]) -> List[RouteChain]:
        """Consolidate groups with less than minimum size (4 dogs)"""
        print(f"\n🔄 Consolidating groups with < {self.MIN_GROUP_SIZE} dogs...")
        
        if not chains:
            print("   No chains to consolidate")
            return chains
        
        changes_made = True
        consolidation_count = 0
        iterations = 0
        
        while changes_made and iterations < self.MAX_CONSOLIDATION_ITERATIONS:
            changes_made = False
            iterations += 1
            
            for chain_idx, chain in enumerate(chains):
                if not chain:  # Skip empty chains
                    continue
                
                # Count dogs in each group
                g1_count = len(chain.group1_dogs) if chain.group1_dogs else 0
                g2_count = len(chain.group2_dogs) if chain.group2_dogs else 0
                g3_count = len(chain.group3_dogs) if chain.group3_dogs else 0
                
                # Count how many groups this driver has
                groups_count = sum([g1_count > 0, g2_count > 0, g3_count > 0])
                
                # Don't consolidate if it would leave driver with no groups
                if groups_count <= 1:
                    continue
                
                # Special check: Figure out which groups would remain after consolidation
                remaining_groups = []
                small_groups = []
                
                if g1_count >= self.MIN_GROUP_SIZE:
                    remaining_groups.append(1)
                elif 0 < g1_count < self.MIN_GROUP_SIZE:
                    small_groups.append((1, chain.group1_dogs[:], "Group 1"))  # Copy list
                    
                if g2_count >= self.MIN_GROUP_SIZE:
                    remaining_groups.append(2)
                elif 0 < g2_count < self.MIN_GROUP_SIZE:
                    small_groups.append((2, chain.group2_dogs[:], "Group 2"))  # Copy list
                    
                if g3_count >= self.MIN_GROUP_SIZE:
                    remaining_groups.append(3)
                elif 0 < g3_count < self.MIN_GROUP_SIZE:
                    small_groups.append((3, chain.group3_dogs[:], "Group 3"))  # Copy list
                
                # Don't consolidate if it would leave only Group 2
                if remaining_groups == [2]:
                    print(f"   ⚠️  Cannot consolidate {chain.driver_id} - would leave only Group 2")
                    continue
                
                # Process small groups
                for group_num, group_dogs, group_name in small_groups:
                    if not group_dogs:  # Safety check
                        continue
                        
                    print(f"\n   ⚠️  {chain.driver_id} {group_name} has only {len(group_dogs)} dogs")
                    
                    # Find best alternative placement for each dog
                    dogs_to_move = group_dogs.copy()
                    successfully_moved = []
                    
                    for dog in dogs_to_move:
                        # Ensure we have a valid dog name for error handling
                        if dog and hasattr(dog, 'name'):
                            dog_name = dog.name
                        else:
                            dog_name = f"Dog_{dog.dog_id if dog and hasattr(dog, 'dog_id') else 'Unknown'}"
                        
                        best_chain = None
                        best_distance = float('inf')
                        
                        # Look for another chain with this group that has capacity
                        for other_idx, other_chain in enumerate(chains):
                            if other_idx == chain_idx or not other_chain:
                                continue
                            
                            # Get the appropriate group from other chain
                            if group_num == 1:
                                other_group_dogs = other_chain.group1_dogs if other_chain.group1_dogs else []
                            elif group_num == 2:
                                other_group_dogs = other_chain.group2_dogs if other_chain.group2_dogs else []
                            else:
                                other_group_dogs = other_chain.group3_dogs if other_chain.group3_dogs else []
                            
                            # Skip if other chain doesn't have this group
                            if len(other_group_dogs) == 0:
                                continue
                            
                            # Check capacity
                            other_chain_density = self.calculate_chain_density(other_chain)
                            capacity = self.DENSE_CAPACITY if other_chain_density < self.DENSE_THRESHOLD else self.STANDARD_CAPACITY
                            
                            if len(other_group_dogs) >= capacity:
                                continue
                            
                            # Calculate average distance to dogs in this group
                            total_dist = 0
                            count = 0
                            for other_dog in other_group_dogs[:5]:  # Sample for efficiency
                                dist = self.get_distance(dog, other_dog)
                                if dist < float('inf'):
                                    total_dist += dist
                                    count += 1
                            
                            if count > 0:
                                avg_dist = total_dist / count
                                if avg_dist < best_distance:
                                    best_distance = avg_dist
                                    best_chain = other_chain
                        
                        # Move dog to best chain if found
                        if best_chain and best_distance < 10.0:
                            if group_num == 1:
                                best_chain.group1_dogs.append(dog)
                            elif group_num == 2:
                                best_chain.group2_dogs.append(dog)
                            else:
                                best_chain.group3_dogs.append(dog)
                            
                            successfully_moved.append(dog)
                            print(f"      ✅ Moved {dog_name} to {best_chain.driver_id} ({best_distance:.1f}min)")
                        else:
                            print(f"      ❌ Could not find good placement for {dog_name}")
                    
                    # Remove successfully moved dogs from original chain
                    if successfully_moved:
                        if group_num == 1:
                            chain.group1_dogs = [d for d in chain.group1_dogs if d not in successfully_moved]
                        elif group_num == 2:
                            chain.group2_dogs = [d for d in chain.group2_dogs if d not in successfully_moved]
                        else:
                            chain.group3_dogs = [d for d in chain.group3_dogs if d not in successfully_moved]
                        
                        consolidation_count += len(successfully_moved)
                        changes_made = True
                        break  # Re-evaluate this chain
        
        # Remove empty chains
        chains = [c for c in chains if c and (c.group1_dogs or c.group2_dogs or c.group3_dogs)]
        
        # Renumber chains
        for idx, chain in enumerate(chains):
            chain.driver_id = f"Driver_{idx + 1:02d}"
        
        if iterations >= self.MAX_CONSOLIDATION_ITERATIONS:
            print(f"\n⚠️  Reached maximum consolidation iterations ({self.MAX_CONSOLIDATION_ITERATIONS})")
        
        print(f"\n✅ Consolidation complete: {consolidation_count} dogs moved")
        return chains
    
    def calculate_chain_density(self, chain: RouteChain) -> float:
        """Calculate average distance within a chain to determine if it's dense"""
        if not chain:
            return self.DENSE_THRESHOLD + 1  # Default to standard capacity
            
        all_dogs = []
        seen_ids = set()
        
        # Safely collect all unique dogs
        for dog in (chain.group1_dogs or []) + (chain.group2_dogs or []) + (chain.group3_dogs or []):
            if dog and dog.dog_id not in seen_ids:
                all_dogs.append(dog)
                seen_ids.add(dog.dog_id)
        
        if len(all_dogs) < 2:
            return self.DENSE_THRESHOLD + 1  # Default to standard capacity
        
        # Sample distances to avoid O(n²) for large groups
        distances = []
        sample_size = min(10, len(all_dogs))
        
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                try:
                    dist = self.get_distance(all_dogs[i], all_dogs[j])
                    if dist < float('inf'):
                        distances.append(dist)
                except Exception:
                    continue
        
        if distances:
            return sum(distances) / len(distances)
        else:
            return self.DENSE_THRESHOLD + 1  # Default to standard capacity
    
    def optimize_chains(self, chains: List[RouteChain]) -> List[RouteChain]:
        """Optimize chains for balance and efficiency"""
        print("\n⚖️  Optimizing chains for balance...")
        
        if not chains:
            print("   No chains to optimize")
            return chains
        
        # Calculate current statistics
        total_dogs = 0
        for chain in chains:
            if chain:
                total_dogs += chain.total_dogs
        
        if len(chains) == 0:
            avg_dogs_per_driver = 0
        else:
            avg_dogs_per_driver = total_dogs / len(chains)
        
        print(f"📊 Average dogs per driver: {avg_dogs_per_driver:.1f}")
        
        # TODO: Could implement more sophisticated balancing here
        # For now, chains are already well-balanced from clustering
        
        return chains
    
    def generate_assignments(self) -> Dict[str, str]:
        """Generate final assignments in format needed for Google Sheets"""
        assignments = {}
        
        if not self.routes:
            print("⚠️  No routes to generate assignments from")
            return assignments
        
        for chain in self.routes:
            if not chain:  # Skip None/empty chains
                continue
                
            # Track which dogs we've already assigned for this driver
            assigned_for_driver = set()
            
            # Collect all unique dogs in this chain
            all_chain_dogs = {}  # dog_id -> dog object
            
            for dog in (chain.group1_dogs or []):
                if dog and hasattr(dog, 'dog_id'):
                    all_chain_dogs[dog.dog_id] = dog
                    
            for dog in (chain.group2_dogs or []):
                if dog and hasattr(dog, 'dog_id'):
                    all_chain_dogs[dog.dog_id] = dog
                    
            for dog in (chain.group3_dogs or []):
                if dog and hasattr(dog, 'dog_id'):
                    all_chain_dogs[dog.dog_id] = dog
            
            # Create assignment for each unique dog
            for dog_id, dog in all_chain_dogs.items():
                if not dog_id or dog_id in assigned_for_driver:
                    continue
                    
                # Build the group string from dog's original needs
                group_string = ''.join(str(g) for g in sorted(dog.groups_needed))
                
                # Format: Driver_01:123 or Driver_02:1 etc
                assignment = f"{chain.driver_id}:{group_string}"
                assignments[dog_id] = assignment
                assigned_for_driver.add(dog_id)
        
        print(f"📋 Generated {len(assignments)} assignments")
        return assignments
    
    def analyze_results(self):
        """Analyze and display the optimization results"""
        print("\n📊 OPTIMIZATION RESULTS")
        print("=" * 60)
        
        if not self.routes:
            print("❌ No routes created!")
            return
        
        # Overall statistics
        total_dogs_assigned = sum(chain.total_dogs for chain in self.routes)
        total_drivers = len(self.routes)
        
        print(f"✅ Total drivers needed: {total_drivers}")
        print(f"✅ Total dogs assigned: {total_dogs_assigned}/{len(self.dogs)}")
        
        if total_drivers > 0:
            avg_dogs = total_dogs_assigned / total_drivers
            print(f"✅ Average dogs per driver: {avg_dogs:.1f}")
        
        # Check for unassigned dogs
        assigned_dog_ids = set()
        for chain in self.routes:
            for dog in chain.group1_dogs + chain.group2_dogs + chain.group3_dogs:
                assigned_dog_ids.add(dog.dog_id)
        
        unassigned = [d for d in self.dogs if d.dog_id not in assigned_dog_ids]
        if unassigned:
            print(f"\n⚠️  {len(unassigned)} dogs not assigned:")
            for dog in unassigned[:10]:  # Show first 10
                print(f"   - {dog.name} (needs groups {dog.groups_needed})")
        
        # Detailed breakdown
        print("\n📋 Driver Assignments:")
        for chain in sorted(self.routes, key=lambda x: x.driver_id):
            print(f"\n{chain.driver_id}:")
            print(f"  Group 1: {len(chain.group1_dogs)} dogs", end="")
            if 0 < len(chain.group1_dogs) < self.MIN_GROUP_SIZE:
                print(" ⚠️  BELOW MINIMUM", end="")
            print()
            
            print(f"  Group 2: {len(chain.group2_dogs)} dogs", end="")
            if 0 < len(chain.group2_dogs) < self.MIN_GROUP_SIZE:
                print(" ⚠️  BELOW MINIMUM", end="")
            print()
            
            print(f"  Group 3: {len(chain.group3_dogs)} dogs", end="")
            if 0 < len(chain.group3_dogs) < self.MIN_GROUP_SIZE:
                print(" ⚠️  BELOW MINIMUM", end="")
            print()
            
            print(f"  Total unique dogs: {chain.total_dogs}")
            
            # Calculate route statistics
            all_dogs_list = []
            seen_ids = set()
            for dog in chain.group1_dogs + chain.group2_dogs + chain.group3_dogs:
                if dog.dog_id not in seen_ids:
                    all_dogs_list.append(dog)
                    seen_ids.add(dog.dog_id)
            
            if len(all_dogs_list) > 1:
                distances = []
                for i in range(len(all_dogs_list)):
                    for j in range(i + 1, len(all_dogs_list)):
                        dist = self.get_distance(all_dogs_list[i], all_dogs_list[j])
                        if dist < float('inf'):
                            distances.append(dist)
                
                if distances:
                    avg_dist = sum(distances) / len(distances)
                    max_dist = max(distances)
                    min_dist = min(distances)
                    density = "DENSE" if avg_dist < self.DENSE_THRESHOLD else "STANDARD"
                    capacity = self.DENSE_CAPACITY if density == "DENSE" else self.STANDARD_CAPACITY
                    
                    print(f"  Route stats: avg={avg_dist:.1f}min, min={min_dist:.1f}min, max={max_dist:.1f}min")
                    print(f"  Route type: {density} (capacity: {capacity})")
                    
                    # Check if over capacity
                    for group_name, group_dogs, group_num in [
                        ("Group 1", chain.group1_dogs, 1),
                        ("Group 2", chain.group2_dogs, 2),
                        ("Group 3", chain.group3_dogs, 3)
                    ]:
                        if len(group_dogs) > capacity:
                            print(f"  ⚠️  {group_name} OVER CAPACITY: {len(group_dogs)}/{capacity}")
            
            # Transition efficiency
            if chain.group1_dogs and chain.group2_dogs:
                g1_to_g2 = self.calculate_transition_distance(chain.group1_dogs, chain.group2_dogs)
                if g1_to_g2 > 0:
                    print(f"  G1→G2 transition: {g1_to_g2:.1f}min avg")
            
            if chain.group2_dogs and chain.group3_dogs:
                g2_to_g3 = self.calculate_transition_distance(chain.group2_dogs, chain.group3_dogs)
                if g2_to_g3 > 0:
                    print(f"  G2→G3 transition: {g2_to_g3:.1f}min avg")
        
        # Summary statistics
        print(f"\n📊 SUMMARY:")
        print(f"  Drivers needed: {total_drivers}")
        print(f"  Dogs assigned: {total_dogs_assigned}/{len(self.dogs)}")
        
        if total_drivers > 0:
            avg_per_driver = total_dogs_assigned / total_drivers
            print(f"  Average per driver: {avg_per_driver:.1f} dogs")
        
        if unassigned:
            print(f"  ⚠️  Unassigned dogs: {len(unassigned)}")
        else:
            print(f"  ✅ All dogs assigned!")
        
        # Check for remaining small groups
        small_groups_remaining = 0
        for chain in self.routes:
            if chain:  # Safety check
                if 0 < len(chain.group1_dogs or []) < self.MIN_GROUP_SIZE:
                    small_groups_remaining += 1
                if 0 < len(chain.group2_dogs or []) < self.MIN_GROUP_SIZE:
                    small_groups_remaining += 1
                if 0 < len(chain.group3_dogs or []) < self.MIN_GROUP_SIZE:
                    small_groups_remaining += 1
        
        if small_groups_remaining > 0:
            print(f"  ⚠️  {small_groups_remaining} groups still below minimum size")
            print(f"     (Could not consolidate - may need manual adjustment)")
    
    def write_to_sheets(self, assignments: Dict[str, str]):
        """Write assignments back to Google Sheets
        
        Fills in the Combined column (H) with optimized assignments like:
        - Driver_01:123 (for a dog that needs all 3 groups)
        - Driver_02:1 (for a dog that only needs group 1)
        - Driver_03:23 (for a dog that needs groups 2 and 3)
        """
        try:
            print("\n💾 Writing results to Google Sheets...")
            print("   (Filling in blank Combined column with optimized assignments)")
            
            sheet = self.gc.open_by_key(self.MAP_SHEET_ID).worksheet("Map")
            
            # Read all values to get row indices
            all_values = sheet.get_all_values()
            
            updates = []
            combined_col = 'H'  # Column H for combined assignments
            dog_id_col_idx = 9  # Column J for dog ID (0-indexed)
            
            # Build updates
            updated_count = 0
            for row_idx, row in enumerate(all_values[1:], 2):  # Start at row 2
                if len(row) > dog_id_col_idx:
                    dog_id = row[dog_id_col_idx].strip()
                    if dog_id in assignments:
                        updates.append({
                            'range': f'{combined_col}{row_idx}',
                            'values': [[assignments[dog_id]]]
                        })
                        updated_count += 1
            
            # Batch update
            if updates:
                print(f"   Updating {len(updates)} assignments...")
                for i in range(0, len(updates), 25):
                    batch = updates[i:i+25]
                    for update in batch:
                        sheet.update(update['values'], update['range'])
                        time.sleep(0.5)
                    print(f"   📝 Updated batch {i//25 + 1}/{(len(updates)-1)//25 + 1}")
                
                print(f"✅ Successfully wrote {len(updates)} assignments to Combined column")
            else:
                print("⚠️  No assignments to write - something may be wrong")
                
        except Exception as e:
            print(f"❌ Error writing to sheets: {e}")
            import traceback
            traceback.print_exc()
    
    def run_optimization(self):
        """Main optimization process"""
        try:
            print("\n🚀 Starting chain-based optimization...")
            
            if not self.dogs:
                print("❌ No dogs to optimize!")
                return
            
            # Build chains
            self.routes = self.build_transition_aware_chains()
            
            if not self.routes:
                print("❌ Failed to build any routes!")
                return
            
            # Consolidate small groups
            self.routes = self.consolidate_small_groups(self.routes)
            
            # Optimize for balance
            self.routes = self.optimize_chains(self.routes)
            
            # Analyze results
            self.analyze_results()
            
            # Generate assignments
            assignments = self.generate_assignments()
            
            if not assignments:
                print("⚠️  No assignments generated - something went wrong")
                return
            
            # Validate assignments cover all dogs
            assigned_dogs = set(assignments.keys())
            all_dog_ids = set(dog.dog_id for dog in self.dogs if dog and hasattr(dog, 'dog_id'))
            unassigned = all_dog_ids - assigned_dogs
            
            if unassigned:
                print(f"⚠️  WARNING: {len(unassigned)} dogs have no assignment!")
                for dog_id in list(unassigned)[:10]:  # Show first 10
                    print(f"     - {dog_id}")
            
            # Write to sheets
            if assignments:
                self.write_to_sheets(assignments)
            
            print("\n✅ Optimization complete!")
            
        except Exception as e:
            print(f"\n❌ Error during optimization: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    try:
        system = ChainBasedClusteringSystem()
        
        # Check if we have data
        if not system.dogs:
            print("❌ No dogs loaded. Please check your data source.")
            return
        
        print(f"\n🐕 Ready to optimize {len(system.dogs)} dogs")
        
        # Check if running in GitHub Actions
        is_github_actions = os.environ.get('GITHUB_ACTIONS') == 'true'
        
        if is_github_actions:
            print("🤖 Running in GitHub Actions - proceeding with optimization")
            system.run_optimization()
        else:
            # Interactive mode
            print("\nOptions:")
            print("1. Run optimization")
            print("2. Analyze current data only")
            print("3. Exit")
            
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == '1':
                system.run_optimization()
            elif choice == '2':
                # Just analyze without optimizing
                if system.routes:
                    system.analyze_results()
                else:
                    print("No routes to analyze. Run optimization first.")
            elif choice == '3':
                print("👋 Goodbye!")
            else:
                print("❌ Invalid choice")
        
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
