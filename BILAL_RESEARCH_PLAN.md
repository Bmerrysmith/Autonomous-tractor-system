# Bilal Research Plan: Lane Detection, Localization, and LiDAR-Camera Integration

**Project:** Autonomous Agricultural Tractor
**Role:** Bilal Dogutas — Lane Detection & Localization
**Branch:** `bilal/lane-detection`

## 1. Research Goal

Build the lane detection and localization subsystem for an autonomous paddy-field tractor. This subsystem must provide stable crop-row geometry, global pose, and a clean interface for the object detection module so the full system can navigate rows, identify weeds, and act on them safely.

The focus is not only implementation. The work should also read like a research contribution: clear problem framing, literature-backed algorithm choices, measurable outputs, and an integration story that connects perception, localization, and mission control.

## 2. Bilal's Scope

My work covers two tightly connected research threads:

1. LiDAR-based crop row detection for under-canopy navigation.
2. EKF-based GNSS/IMU sensor fusion for global localization.

The main deliverable is a row-aware navigation layer that publishes:

- left row boundary
- right row boundary
- row centerline
- tractor pose estimate
- confidence or quality metrics for downstream modules

These outputs are meant to support both lane following and the ROI bridge used by the weed detection pipeline.

## 3. Literature Review Targets

### A. LiDAR Crop Row Detection

The row-detection paper should justify a geometric approach for field environments where RGB-only methods are fragile under lighting change, dust, shadows, and occlusion.

Key ideas to capture:

- extracting row boundaries from 2D LiDAR scans
- clustering or line fitting on crop-row returns
- row centerline estimation for steering control
- handling irregular spacing, missing returns, and noisy vegetation edges

### B. EKF GNSS/IMU Localization

The localization paper should support a practical EKF design for outdoor agricultural robots.

Key ideas to capture:

- fusing GNSS position with IMU orientation and angular rate
- smoothing noisy GPS while maintaining drift resistance
- supporting global pose for mission planning and return-to-row behavior
- defining a state vector and measurement update process suitable for ROS integration

### C. Optional Bridge Paper: LiDAR-IMU Row Following

If the team wants a more system-level contribution, a third paper can support row following under partial GNSS loss. This helps justify graceful degradation in real farm conditions.

## 4. Proposed Research Contribution

The original contribution for my section should be presented as a sensor-fusion navigation layer with direct system utility.

Proposed contribution statement:

> A LiDAR-guided crop-row localization and GNSS/IMU fusion module for autonomous paddy-field navigation, producing robust row geometry and global pose estimates that also constrain downstream weed detection through a region-of-interest bridge.

This keeps the section grounded in research while still tying it to the larger tractor system.

## 5. Software Architecture for My Portion

### Node 1: `lidar_row_detection_node`

Responsibilities:

- ingest 2D LiDAR scans
- detect crop row edges or clusters
- estimate left/right row boundaries
- compute row centerline
- publish lane geometry for steering and for Benny's ROI subscriber

Suggested outputs:

- left boundary line or polygon
- right boundary line or polygon
- centerline points or polynomial fit
- confidence score

### Node 2: `ekf_localization_node`

Responsibilities:

- fuse GNSS and IMU measurements
- maintain tractor pose in a global frame
- smooth short-term GNSS noise
- provide heading and position for planner and lane-following logic

Suggested outputs:

- x, y, yaw
- velocity estimate if available
- covariance or confidence metrics

### Interface Contract

The lane detection node should not be a black box. It should publish data in a format that the rest of the system can consume consistently.

Recommended contract:

- row boundaries as ROS geometry messages or a custom lane message
- centerline as a path or marker sequence
- pose as standard localization output
- timestamped data synchronized to the camera and planner cycle

## 6. How My Work Supports the Full System

My subsystem has two downstream impacts:

1. Navigation: the mission controller can keep the tractor aligned in rows and recover from drift.
2. Detection: row boundary data can constrain Benny's weed detection pipeline to the valid inter-row region.

This is the cleanest way to show that lane detection is not just a support feature. It is a system-wide enabler.

## 7. Research Questions

The write-up should answer the following questions:

1. How reliably can 2D LiDAR recover crop-row geometry in a paddy field?
2. How much does GNSS/IMU EKF fusion improve pose stability over raw GPS?
3. Can row geometry be expressed in a way that improves both lane following and weed-detection ROI filtering?
4. What failure modes appear under dense foliage, weak GNSS, or uneven field structure?

## 8. Suggested Methodology

### Step 1: Baseline row detection

- implement LiDAR scan processing
- cluster returns and fit row boundaries
- evaluate centerline smoothness and row tracking stability

### Step 2: Localization fusion

- design EKF state and update model
- integrate GNSS position and IMU orientation
- compare raw vs fused pose consistency

### Step 3: Integration testing

- publish row geometry in ROS
- verify planner compatibility
- verify Benny's ROI subscriber can consume the row output

### Step 4: Metrics

Suggested metrics:

- row boundary precision
- centerline stability
- pose drift reduction
- localization jitter
- downstream ROI reduction in false positives

## 9. Paper Section Draft Outline

### Section IV: LiDAR-only Crop Row Navigation

- paddy-field navigation problem
- why LiDAR geometry is reliable in farm rows
- row extraction and centerline generation
- limitations and failure cases

### Section V: EKF-based GNSS-IMU Localization

- outdoor localization challenge
- EKF sensor fusion model
- state estimation pipeline
- performance under noisy GNSS

### Section VIII: System Integration

- how lane detection feeds mission control
- how row data constrains weed detection
- ROS topic contract and module dependencies
- overall tractor workflow

## 10. Current Status

Completed or in progress:

- assigned lane detection and localization role
- identified LiDAR row detection as the core navigation contribution
- identified EKF GNSS/IMU fusion as the global pose contribution
- established the row boundary data bridge to the detection subsystem

Still needed:

- select and read the final path-planning paper for the mission controller
- finalize ROS message formats
- build and test the LiDAR row detection pipeline
- implement and validate the EKF fusion node

## 11. Path Planning Research Gap

The remaining literature review gap is path planning for the weed-intercept maneuver.

Best candidate algorithms:

- DWA for fast, lightweight, short detours
- TEB for kinematic constraints and ROS compatibility
- RRT* for obstacle-heavy routing

Recommended choice for this tractor use case:

- DWA if the goal is simple, real-time detours in structured rows
- TEB if the mission controller needs stronger vehicle-constraint handling

This paper choice should be coordinated with Anthony because it affects the mission controller state machine.

## 12. Final Deliverable

My final contribution should be a research-backed lane detection and localization package that can be described in the paper as a robust navigation subsystem for autonomous paddy-field operation.

The final system should show that the tractor can:

- detect rows
- stay centered
- recover pose with EKF fusion
- publish usable geometry for weed interception planning
- support the larger autonomy pipeline
