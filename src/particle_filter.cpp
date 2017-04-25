/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <iostream>
#include <float.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  //Set the number of particles.
  num_particles = NUMBER_OF_PARTICLES;
  particles.resize(num_particles);
  weights.resize(num_particles);

  normal_distribution<double> N_x_init(x, std[0]);
  normal_distribution<double> N_y_init(y, std[1]);
  normal_distribution<double> N_theta_init(theta, std[2]);

  // Initialize all particles to first position, all weights to 1.
  // Add random Gaussian noise to each particle.
  for (int i = 0; i < num_particles; i++) {
    particles[i].init(i, N_x_init(gen), N_y_init(gen), N_theta_init(gen), 1);
    weights[i] = 1;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	//Add measurements to each particle and add random Gaussian noise.

  normal_distribution<double> N_x_init(0, std_pos[0]);
  normal_distribution<double> N_y_init(0, std_pos[1]);
  normal_distribution<double> N_theta_init(0, std_pos[2]);

  for (auto &particle : particles) // access by reference to avoid copying
  {
    if (fabs(yaw_rate) < SMALL_NUMBER) {
      particle.x += velocity * delta_t * cos(particle.theta);
      particle.y += velocity * delta_t * sin(particle.theta);
    } else {
      particle.y += velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
      particle.x += velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
      particle.theta += yaw_rate * delta_t;
    }
    particle.x     += N_x_init(gen);
    particle.y     += N_y_init(gen);
    particle.theta += N_theta_init(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
  for (auto &observation : observations) {
    double min_dist = DBL_MAX;

    for (int i = 0; i < predicted.size(); i++) {
      //calculate distance between landmark and observation
      double dist = evaluatePointsDistance(observation.x, predicted[i].x, observation.y, predicted[i].y);

      //find the smallest distance for this observation and remember the landmark as id
      if (dist < min_dist) {
        observation.id = i;
        min_dist = dist;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks) {
  weights.clear();

  for (auto &particle : particles) {
    //create temp variables
    vector<LandmarkObs> convertedObservations;
    vector<LandmarkObs> predicted_landmarks;
    double newWeight;

    //transform to the map coordinates system and put the new converted observation into the vector
    for (auto &observation : observations) {
      LandmarkObs convertedObservation;
      convertedObservation.id = FALSE_ID;
      convertedObservation.x = particle.x + observation.x * cos(particle.theta) - observation.y * sin(particle.theta);
      convertedObservation.y = particle.y + observation.x * sin(particle.theta) + observation.y * cos(particle.theta);
      convertedObservations.push_back(convertedObservation);
    }

    //find landmarks that the current particle can measure, i.e. landmarks distance to that < than sensor range
    for (auto &landmark : map_landmarks.landmark_list) {
      if (evaluatePointsDistance(particle.x, landmark.x_f, particle.y, landmark.y_f) > sensor_range) {
        continue;
      }
      LandmarkObs predictedLandmark;
      predictedLandmark.id = landmark.id_i;
      predictedLandmark.x  = landmark.x_f;
      predictedLandmark.y  = landmark.y_f;
      predicted_landmarks.push_back(predictedLandmark);
    }

    if ((convertedObservations.size() == 0 && predicted_landmarks.size() > 0) || (predicted_landmarks.size() == 0 && convertedObservations.size() > 0)) {
      newWeight = SMALL_NUMBER;
    } else {
      newWeight = 1;
      //associate observations with sensed landmarks
      dataAssociation(predicted_landmarks, convertedObservations);

      //find probability of all observations using gaussian distribution
      for (auto &observation: convertedObservations) {
        if (observation.id < 0) {
          continue;
        }
        LandmarkObs &landmark = predicted_landmarks[observation.id];
        newWeight *= evaluateMultiGaussian(observation.x, landmark.x, observation.y, landmark.y, std_landmark[0], std_landmark[1]);
      }
    }
    particle.weight = newWeight;
    weights.push_back(newWeight);
  }
}

void ParticleFilter::resample() {
  vector<Particle> sampledParticles;
  discrete_distribution<> N_weight(weights.begin(), weights.end());

  //Resample particles with replacement with probability proportional to their weight.
  for (int i = 0; i < num_particles; i++) {
    sampledParticles.push_back(particles[N_weight(gen)]);
  }
  particles = sampledParticles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}

double ParticleFilter::evaluatePointsDistance(const double x1, const double x2, const double y1, const double y2) {
  return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

double ParticleFilter::evaluateMultiGaussian(const double x1, const double x2, const double y1, const double y2, const double sigmaX, const double sigmaY) {
  return exp(-((pow(x1 - x2, 2) / (sigmaX * sigmaX)) + (pow(y1 - y2, 2) / (sigmaY * sigmaY))) / 2) / 2 / M_PI / sigmaX / sigmaY;
}
