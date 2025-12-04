#!/usr/bin/env python3
"""
CITYFLOW COMPLETE SETUP FOR PRESSLIGHT
Generates all required configuration files automatically
"""

import json
import os
import sys

def generate_arterial_roadnet():
    """Generate 2-intersection arterial road network"""
    
    roadnet = {
        "intersections": [
            {
                "id": "intersection_1",
                "point": {"x": 0.0, "y": 0.0},
                "width": 15.0,
                "roads": ["road_0_1", "road_1_2", "road_3_1", "road_1_4"],
                "roadLinks": [
                    {
                        "type": "go_straight",
                        "startRoad": "road_0_1",
                        "endRoad": "road_1_2",
                        "direction": 0,
                        "laneLinks": [
                            {"startLaneIndex": 0, "endLaneIndex": 0},
                            {"startLaneIndex": 1, "endLaneIndex": 1},
                            {"startLaneIndex": 2, "endLaneIndex": 2}
                        ]
                    },
                    {
                        "type": "go_straight",
                        "startRoad": "road_3_1",
                        "endRoad": "road_1_4",
                        "direction": 2,
                        "laneLinks": [
                            {"startLaneIndex": 0, "endLaneIndex": 0},
                            {"startLaneIndex": 1, "endLaneIndex": 1}
                        ]
                    }
                ],
                "trafficLight": {
                    "roadLinkIndices": [[0], [1]],
                    "lightphases": [
                        {"time": 30, "availableRoadLinks": [0]},
                        {"time": 30, "availableRoadLinks": [1]}
                    ]
                }
            },
            {
                "id": "intersection_2",
                "point": {"x": 300.0, "y": 0.0},
                "width": 15.0,
                "roads": ["road_1_2", "road_2_5", "road_6_2", "road_2_7"],
                "roadLinks": [
                    {
                        "type": "go_straight",
                        "startRoad": "road_1_2",
                        "endRoad": "road_2_5",
                        "direction": 0,
                        "laneLinks": [
                            {"startLaneIndex": 0, "endLaneIndex": 0},
                            {"startLaneIndex": 1, "endLaneIndex": 1},
                            {"startLaneIndex": 2, "endLaneIndex": 2}
                        ]
                    },
                    {
                        "type": "go_straight",
                        "startRoad": "road_6_2",
                        "endRoad": "road_2_7",
                        "direction": 2,
                        "laneLinks": [
                            {"startLaneIndex": 0, "endLaneIndex": 0},
                            {"startLaneIndex": 1, "endLaneIndex": 1}
                        ]
                    }
                ],
                "trafficLight": {
                    "roadLinkIndices": [[0], [1]],
                    "lightphases": [
                        {"time": 30, "availableRoadLinks": [0]},
                        {"time": 30, "availableRoadLinks": [1]}
                    ]
                }
            }
        ],
        "roads": []
    }
    
    # Define roads
    roads_config = [
        {"id": "road_0_1", "start": "intersection_0", "end": "intersection_1", "lanes": 3, "length": 300, "speed": 16.67},
        {"id": "road_1_2", "start": "intersection_1", "end": "intersection_2", "lanes": 3, "length": 300, "speed": 16.67},
        {"id": "road_2_5", "start": "intersection_2", "end": "intersection_5", "lanes": 3, "length": 300, "speed": 16.67},
        {"id": "road_3_1", "start": "intersection_3", "end": "intersection_1", "lanes": 2, "length": 300, "speed": 11.11},
        {"id": "road_1_4", "start": "intersection_1", "end": "intersection_4", "lanes": 2, "length": 300, "speed": 11.11},
        {"id": "road_6_2", "start": "intersection_6", "end": "intersection_2", "lanes": 2, "length": 300, "speed": 11.11},
        {"id": "road_2_7", "start": "intersection_2", "end": "intersection_7", "lanes": 2, "length": 300, "speed": 11.11}
    ]
    
    for road in roads_config:
        road_obj = {
            "id": road["id"],
            "startIntersection": road["start"],
            "endIntersection": road["end"],
            "lanes": [{"width": 3.0, "maxSpeed": road["speed"]} for _ in range(road["lanes"])]
        }
        roadnet["roads"].append(road_obj)
    
    return roadnet

def generate_arterial_flow():
    """Generate traffic flow"""
    
    vehicle_template = {
        "length": 5.0,
        "width": 2.0,
        "maxPosAcc": 2.0,
        "maxNegAcc": 4.5,
        "usualPosAcc": 2.0,
        "usualNegAcc": 4.5,
        "minGap": 2.5,
        "maxSpeed": 16.67,
        "headwayTime": 2.0
    }
    
    flow = [
        {
            "vehicle": vehicle_template,
            "route": ["road_0_1", "road_1_2", "road_2_5"],
            "interval": 3.0,
            "startTime": 0,
            "endTime": 3600
        },
        {
            "vehicle": vehicle_template,
            "route": ["road_3_1", "road_1_4"],
            "interval": 9.0,
            "startTime": 0,
            "endTime": 3600
        },
        {
            "vehicle": vehicle_template,
            "route": ["road_6_2", "road_2_7"],
            "interval": 9.0,
            "startTime": 0,
            "endTime": 3600
        }
    ]
    
    return flow

def generate_config():
    """Generate simulation configuration"""
    
    config = {
        "interval": 1.0,
        "seed": 42,
        "dir": "./cityflow_data/",
        "roadnetFile": "roadnet_2x1.json",
        "flowFile": "flow_arterial.json",
        "rlTrafficLight": True,
        "saveReplay": False,
        "laneChange": False
    }
    
    return config

def main():
    print("\n" + "="*70)
    print("CITYFLOW SETUP FOR PRESSLIGHT")
    print("="*70)
    
    os.makedirs("cityflow_data", exist_ok=True)
    
    print("\n🔧 Generating configuration files...")
    
    roadnet = generate_arterial_roadnet()
    with open("cityflow_data/roadnet_2x1.json", "w") as f:
        json.dump(roadnet, f, indent=2)
    print("  ✓ Created roadnet_2x1.json")
    
    flow = generate_arterial_flow()
    with open("cityflow_data/flow_arterial.json", "w") as f:
        json.dump(flow, f, indent=2)
    print("  ✓ Created flow_arterial.json")
    
    config = generate_config()
    with open("cityflow_data/config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("  ✓ Created config.json")
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("  1. Install CityFlow: pip install cityflow torch numpy")
    print("  2. Run training: python presslight_complete.py --train")

if __name__ == "__main__":
    main()