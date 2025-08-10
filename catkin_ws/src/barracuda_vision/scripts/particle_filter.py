#!/usr/bin/env python3

import numpy as np
import math
from filterpy.monte_carlo import systematic_resample
from enum import Enum

class TrackState(Enum):
    """States for track management"""
    CONFIRMED = "confirmed"  # Have recent hit
    HOLD = "hold"           # Missed ≤ k frames, confidence decaying
    LOST = "lost"           # Confidence dropped below threshold or covariance too big

class ParticleFilter3D:
    """Optimized particle filter for 3D object tracking with hold-and-decay state machine
    
    Optimizations:
    - Vectorized operations using NumPy arrays instead of individual particle objects
    - Adaptive particle count based on tracking state and confidence
    - Reduced computational overhead through efficient array operations
    """
    def __init__(self, min_particles=20, max_particles=50, process_noise=0.1, measurement_noise=0.5, 
                 max_hold_frames=10, survival_factor=0.97, max_covariance_threshold=2.0, 
                 min_confidence_threshold=0.1):
        # Adaptive particle count parameters
        self.min_particles = min_particles  # Minimum particles (for HOLD state)
        self.max_particles = max_particles  # Maximum particles (for CONFIRMED state)
        self.current_num_particles = max_particles  # Start with max for good initialization
        
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.measurement_noise_sq = measurement_noise ** 2  # Pre-compute for efficiency
        self.initialized = False
        
        # State machine parameters
        self.state = TrackState.LOST
        self.frames_since_update = 0
        self.max_hold_frames = max_hold_frames  # Still used for transition to HOLD
        self.survival_factor = survival_factor
        self.max_covariance_threshold = max_covariance_threshold
        self.min_confidence_threshold = min_confidence_threshold  # Confidence threshold for LOST
        
        # Vectorized particle storage: [N x 3] positions and [N] weights
        self.positions = np.zeros((self.current_num_particles, 3))
        self.weights = np.ones(self.current_num_particles) / self.current_num_particles
        
        # Pre-allocate arrays for efficiency
        self._temp_distances = np.zeros(self.current_num_particles)
        self._temp_weights = np.zeros(self.current_num_particles)
    
    def _resize_arrays_if_needed(self, new_size):
        """Resize particle arrays if needed for adaptive particle count"""
        if new_size != self.current_num_particles:
            old_size = self.current_num_particles
            self.current_num_particles = new_size
            
            if new_size > old_size:
                # Expanding: add new particles by duplicating existing ones with noise
                new_positions = np.zeros((new_size, 3))
                new_weights = np.zeros(new_size)
                
                # Copy existing particles
                new_positions[:old_size] = self.positions
                new_weights[:old_size] = self.weights
                
                # Add new particles with small noise around existing ones
                if old_size > 0:
                    # Duplicate particles with noise
                    indices = np.random.choice(old_size, new_size - old_size)
                    new_positions[old_size:] = self.positions[indices] + np.random.normal(0, 0.1, (new_size - old_size, 3))
                    new_weights[old_size:] = 1.0 / new_size
                
                self.positions = new_positions
                self.weights = new_weights
            else:
                # Shrinking: keep top weighted particles
                if old_size > 0:
                    indices = np.argsort(self.weights)[-new_size:]  # Keep highest weighted
                    self.positions = self.positions[indices]
                    self.weights = self.weights[indices]
                else:
                    self.positions = self.positions[:new_size]
                    self.weights = self.weights[:new_size]
            
            # Normalize weights
            weight_sum = np.sum(self.weights)
            if weight_sum > 0:
                self.weights /= weight_sum
            else:
                self.weights.fill(1.0 / new_size)
            
            # Resize temporary arrays
            self._temp_distances = np.zeros(self.current_num_particles)
            self._temp_weights = np.zeros(self.current_num_particles)
    
    def _adapt_particle_count(self):
        """Adapt particle count based on tracking state"""
        if self.state == TrackState.CONFIRMED:
            target_count = self.max_particles
        elif self.state == TrackState.HOLD:
            # Reduce particles for efficiency during hold
            target_count = self.min_particles
        else:
            target_count = self.max_particles  # For initialization
        
        self._resize_arrays_if_needed(target_count)
    
    def initialize(self, x, y, z, std_dev=1.0):
        """Initialize particles around the first measurement using vectorized operations"""
        # Start with max particles for good initialization
        self._resize_arrays_if_needed(self.max_particles)
        
        # Vectorized initialization
        self.positions[:, 0] = np.random.normal(x, std_dev, self.current_num_particles)
        self.positions[:, 1] = np.random.normal(y, std_dev, self.current_num_particles)
        self.positions[:, 2] = np.random.normal(z, std_dev, self.current_num_particles)
        self.weights.fill(1.0 / self.current_num_particles)
        
        self.initialized = True
        self.state = TrackState.CONFIRMED
        self.frames_since_update = 0
    
    def predict(self, dt=0.1):
        """Predict step - vectorized behavior depends on track state"""
        # Adapt particle count based on current state
        self._adapt_particle_count()
        
        if self.state == TrackState.CONFIRMED:
            # Normal predict: add process noise using vectorized operations
            noise_std = self.process_noise * dt
            self.positions += np.random.normal(0, noise_std, self.positions.shape)
            
            # Auto-resample after prediction for CONFIRMED tracks
            self.resample()
            
        elif self.state == TrackState.HOLD:
            # Skip predict: position stays frozen, but decay weights
            self.weights *= self.survival_factor
            
            # Normalize weights after decay
            weight_sum = np.sum(self.weights)
            if weight_sum > 0:
                self.weights /= weight_sum
            else:
                self.weights.fill(1.0 / self.current_num_particles)
        # LOST state: do nothing (track will be deleted)
    
    def update(self, measurement):
        """Update step - vectorized weight calculation based on measurement likelihood"""
        # If not initialized, initialize with this measurement
        if not self.initialized:
            self.initialize(measurement[0], measurement[1], measurement[2])
            return
        
        # Reset frames since update and set state to CONFIRMED
        self.frames_since_update = 0
        self.state = TrackState.CONFIRMED
        
        # Vectorized distance calculation (avoid sqrt for efficiency)
        measurement_array = np.array(measurement)
        diff = self.positions - measurement_array
        squared_distances = np.sum(diff * diff, axis=1)
        
        # Vectorized Gaussian likelihood calculation
        # Using squared distances to avoid expensive sqrt operations
        self.weights = np.exp(-0.5 * squared_distances / self.measurement_noise_sq)
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.weights.fill(1.0 / self.current_num_particles)
    
    def resample(self):
        """Efficient resampling with adaptive strategy"""
        # Calculate effective sample size
        weight_sum_sq = np.sum(self.weights ** 2)
        eff_sample_size = 1.0 / weight_sum_sq if weight_sum_sq > 0 else 0
        
        # Only resample if effective sample size is too low
        resample_threshold = self.current_num_particles * 0.5
        if eff_sample_size < resample_threshold:
            # Use systematic resampling for efficiency
            indices = systematic_resample(self.weights)
            
            # Vectorized resampling
            self.positions = self.positions[indices]
            self.weights.fill(1.0 / self.current_num_particles)
    
    def get_estimate(self):
        """Get weighted average estimate using vectorized operations"""
        if not self.initialized:
            return None
        
        # Vectorized weighted average
        estimate = np.sum(self.positions * self.weights.reshape(-1, 1), axis=0)
        return estimate.tolist()
    
    def get_covariance(self):
        """Get covariance matrix using vectorized operations"""
        if not self.initialized:
            return None
        
        # Get weighted mean
        mean = np.sum(self.positions * self.weights.reshape(-1, 1), axis=0)
        
        # Vectorized covariance calculation
        diff = self.positions - mean
        # Weighted outer product sum
        cov = np.zeros((3, 3))
        for i in range(self.current_num_particles):
            cov += self.weights[i] * np.outer(diff[i], diff[i])
        
        return cov
    
    def handle_missed_detection(self):
        """Handle a frame where no measurement was received"""
        if not self.initialized:
            return
        
        self.frames_since_update += 1
        
        # Update state based on confidence decay, not just frame count
        current_confidence = self.get_confidence()
        
        if self.state == TrackState.CONFIRMED:
            # Transition to HOLD state when we start missing detections
            self.state = TrackState.HOLD
        elif self.state == TrackState.HOLD:
            # Check if confidence has decayed too much to continue holding
            if current_confidence < self.min_confidence_threshold:
                self.state = TrackState.LOST
            
        # Also check if covariance is too large (track has diverged)
        cov = self.get_covariance()
        if cov is not None:
            # Use trace of covariance matrix as a measure of uncertainty
            uncertainty = np.trace(cov)
            if uncertainty > self.max_covariance_threshold:
                self.state = TrackState.LOST
    
    def is_lost(self):
        """Check if track should be deleted"""
        return self.state == TrackState.LOST
    
    def get_state(self):
        """Get current track state"""
        return self.state
    
    def get_frames_since_update(self):
        """Get number of frames since last measurement"""
        return self.frames_since_update
    
    def get_confidence(self):
        """Get confidence measure based on state and frames since update"""
        if self.state == TrackState.CONFIRMED:
            return 1.0
        elif self.state == TrackState.HOLD:
            # Decay confidence based on frames since update
            decay_factor = self.survival_factor ** self.frames_since_update
            return decay_factor
        else:  # LOST
            return 0.0
    
    def get_num_particles(self):
        """Get current number of particles"""
        return self.current_num_particles
    
    def get_effective_sample_size(self):
        """Get effective sample size for monitoring filter health"""
        weight_sum_sq = np.sum(self.weights ** 2)
        return 1.0 / weight_sum_sq if weight_sum_sq > 0 else 0
