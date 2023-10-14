#include <g2o/core/auto_differentiation.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_fixed_sized_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/slam2d/vertex_se2.h>

#include <Eigen/Core>
#include <vector>

using BlockSolver = g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1>>;
using LinearSolver = g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType>;

struct Parameters {
  double max_vel_x{0.3};
  double max_vel_x_backwards{0.3};
  double max_vel_theta{0.3};
  double acc_lim_x{0.1};
  double acc_lim_theta{0.1};
  double dt_ref{0.1};
  double dt_hysteresis{0.03};
  double weight_max_vel_x{1.0};
  double weight_max_vel_theta{1.0};
  double weight_acc_lim_x{1.0};
  double weight_acc_lim_theta{1.0};
  double weight_timeoptimal{10.0};
  double weight_kinematics_nh{1000.0};
  double weight_kinematics_forward_drive{1.0};
  double penalty_epsilon{0.1};
};

template <typename T>
inline T penaltyBoundToInterval(const T var, const double a, const double b,
                                const double epsilon) {
  const T lower = static_cast<T>(a + epsilon);
  if (var < lower) {
    return lower - var;
  }
  const T upper = static_cast<T>(b - epsilon);
  if (var <= upper) {
    return static_cast<T>(0.);
  }
  return var - upper;
  // const T penalty = static_cast<T>(1);
  // if (var < static_cast<T>(a)) {
  //   return penalty * (static_cast<T>(a) - var) + epsilon;
  // } else if (var < static_cast<T>(a + epsilon)) {
  //   return static_cast<T>(a + epsilon) - var;
  // }
  // if (var > static_cast<T>(b)) {
  //   return penalty * (var - static_cast<T>(b)) + epsilon;
  // } else if (var > static_cast<T>(b - epsilon)) {
  //   return var - static_cast<T>(b - epsilon);
  // }
  // return static_cast<T>(0.);
}

template <typename T>
inline T penaltyBoundFromBelow(const T var, const double a,
                               const double epsilon) {
  const T upper = static_cast<T>(a + epsilon);
  if (var >= upper) {
    return static_cast<T>(0.);
  }
  return upper - var;
}

inline double penaltyBoundFromBelowDerivative(const double var, const double a,
                                              const double epsilon) {
  if (var >= a + epsilon) {
    return 0.;
  } else {
    return -1;
  }
}

template <typename T>
inline T normalize_theta(const T theta) {
  if (theta > static_cast<T>(M_PI)) {
    return theta - static_cast<T>(2 * M_PI);
  } else if (theta < static_cast<T>(-M_PI)) {
    return theta + static_cast<T>(2 * M_PI);
  }
  return theta;
}

template <typename T>
inline T sign(const T v) {
  if (v > static_cast<T>(0)) {
    return static_cast<T>(1);
  } else if (v < static_cast<T>(0)) {
    return static_cast<T>(-1);
  }
  return static_cast<T>(0);
}

class VertexTimeDiff : public g2o::BaseVertex<1, double> {
 public:
  VertexTimeDiff() {}
  void setToOriginImpl() override { _estimate = 1.; }
  void oplusImpl(const double *update) override { _estimate += update[0]; }
  bool getEstimateData(double *est) const override {
    *est = _estimate;
    return true;
  }
  bool read(std::istream &) override { return true; }
  bool write(std::ostream &os) const override { return os.good(); }
};

class EdgeVelocity
    : public g2o::BaseFixedSizedEdge<2, Eigen::Vector2d, g2o::VertexSE2,
                                     g2o::VertexSE2, VertexTimeDiff> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeVelocity() {}
  void setParams(Parameters *params) { params_ = params; }
  template <typename T>
  bool operator()(const T *v1, const T *v2, const T *dt, T *error) const {
    typename g2o::VectorN<2, T>::ConstMapType t1(&v1[0]);
    typename g2o::VectorN<2, T>::ConstMapType t2(&v2[0]);
    const g2o::VectorN<2, T> diff = t2 - t1;
    T vel = diff.norm() / dt[0];
    const T &a1 = v1[2];
    const T &a2 = v2[2];
    vel *=
        sign(diff.x() * g2o::ceres::cos(a1) + diff.y() * g2o::ceres::sin(a1));
    const T angle_diff = normalize_theta(a2 - a1);
    const T omega = angle_diff / dt[0];
    error[0] =
        penaltyBoundToInterval(vel, -params_->max_vel_x_backwards,
                               params_->max_vel_x, params_->penalty_epsilon);
    error[1] = penaltyBoundToInterval(omega, -params_->max_vel_theta,
                                      params_->max_vel_theta,
                                      params_->penalty_epsilon);
    return true;
  }

  G2O_MAKE_AUTO_AD_FUNCTIONS_BY_GET
  bool read(std::istream &) override { return true; }
  bool write(std::ostream &os) const override { return os.good(); }

 private:
  Parameters *params_;
};

class EdgeTimeOptimal : public g2o::BaseUnaryEdge<1, double, VertexTimeDiff> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeTimeOptimal() { setMeasurement(0.); }
  void computeError() {
    _error[0] = static_cast<const VertexTimeDiff *>(_vertices[0])->estimate();
  }
  bool read(std::istream &) override { return true; }
  bool write(std::ostream &os) const override { return os.good(); }
  void linearizeOplus() { _jacobianOplusXi(0, 0) = 1; }
};

class EdgeAcceleration
    : public g2o::BaseFixedSizedEdge<2, Eigen::Vector2d, g2o::VertexSE2,
                                     g2o::VertexSE2, g2o::VertexSE2,
                                     VertexTimeDiff, VertexTimeDiff> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void setParams(Parameters *params) { params_ = params; }
  template <typename T>
  bool operator()(const T *v1, const T *v2, const T *v3, const T *dt1,
                  const T *dt2, T *error) const {
    typename g2o::VectorN<2, T>::ConstMapType t1(&v1[0]);
    typename g2o::VectorN<2, T>::ConstMapType t2(&v2[0]);
    typename g2o::VectorN<2, T>::ConstMapType t3(&v3[0]);
    const g2o::VectorN<2, T> diff1 = t2 - t1;
    const g2o::VectorN<2, T> diff2 = t3 - t2;
    T vel1 = diff1.norm() / dt1[0];
    T vel2 = diff2.norm() / dt2[0];
    const T &a1 = v1[2];
    const T &a2 = v2[2];
    const T &a3 = v3[2];
    vel1 *=
        sign(diff1.x() * g2o::ceres::cos(a1) + diff1.y() * g2o::ceres::sin(a1));
    vel2 *=
        sign(diff2.x() * g2o::ceres::cos(a2) + diff2.y() * g2o::ceres::sin(a2));
    const T acc_lin = (vel2 - vel1) * static_cast<T>(2.) / (dt1[0] + dt2[0]);
    error[0] =
        penaltyBoundToInterval(acc_lin, -params_->acc_lim_x, params_->acc_lim_x,
                               params_->penalty_epsilon);
    const T angle_diff1 = normalize_theta(a2 - a1);
    const T angle_diff2 = normalize_theta(a3 - a2);
    const T omega1 = angle_diff1 / dt1[0];
    const T omega2 = angle_diff2 / dt2[0];
    const T acc_rot =
        (omega2 - omega1) * static_cast<T>(2.) / (dt1[0] + dt2[0]);
    error[1] = penaltyBoundToInterval(acc_rot, -params_->acc_lim_theta,
                                      params_->acc_lim_theta,
                                      params_->penalty_epsilon);
    return true;
  }

  G2O_MAKE_AUTO_AD_FUNCTIONS_BY_GET
  bool read(std::istream &) override { return true; }
  bool write(std::ostream &os) const override { return os.good(); }

 private:
  Parameters *params_;
};

class EdgeAccelerationSingle
    : public g2o::BaseFixedSizedEdge<2, Eigen::Vector2d, g2o::VertexSE2,
                                     g2o::VertexSE2, VertexTimeDiff> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeAccelerationSingle() { resize(3); }
  void setParams(Parameters *params) { params_ = params; }
  void setVelocity(const Eigen::Vector2d &vel, bool is_start_vel) {
    setMeasurement(vel);
    is_start_vel_ = is_start_vel;
  }
  template <typename T>
  bool operator()(const T *v1, const T *v2, const T *dt, T *error) const {
    typename g2o::VectorN<2, T>::ConstMapType t1(&v1[0]);
    typename g2o::VectorN<2, T>::ConstMapType t2(&v2[0]);
    const g2o::VectorN<2, T> diff = t2 - t1;
    T vel = diff.norm() / dt[0];
    const T &a1 = v1[2];
    const T &a2 = v2[2];
    vel *=
        sign(diff.x() * g2o::ceres::cos(a1) + diff.y() * g2o::ceres::sin(a1));
    T acc_lin = (measurement().cast<T>().x() - vel) / dt[0];
    if (!is_start_vel_) {
      acc_lin = -acc_lin;
    }
    error[0] =
        penaltyBoundToInterval(acc_lin, -params_->acc_lim_x, params_->acc_lim_x,
                               params_->penalty_epsilon);
    const T angle_diff = normalize_theta(a2 - a1);
    const T omega = angle_diff / dt[0];
    T acc_rot = (measurement().cast<T>().y() - omega) / dt[0];
    if (!is_start_vel_) {
      acc_rot = -acc_rot;
    }
    error[1] = penaltyBoundToInterval(acc_rot, -params_->acc_lim_theta,
                                      params_->acc_lim_theta,
                                      params_->penalty_epsilon);
    return true;
  }

  G2O_MAKE_AUTO_AD_FUNCTIONS_BY_GET
  bool read(std::istream &) override { return true; }
  bool write(std::ostream &os) const override { return os.good(); }

 private:
  bool is_start_vel_{true};
  Parameters *params_;
};

class EdgeKinematicsDiffDrive
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSE2,
                                 g2o::VertexSE2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeKinematicsDiffDrive() {}
  template <typename T>
  bool operator()(const T *v1, const T *v2, T *error) const {
    typename g2o::VectorN<2, T>::ConstMapType t1(&v1[0]);
    typename g2o::VectorN<2, T>::ConstMapType t2(&v2[0]);
    const T &a1 = v1[2];
    const T &a2 = v2[2];
    const T cos1 = g2o::ceres::cos(a1);
    const T cos2 = g2o::ceres::cos(a2);
    const T sin1 = g2o::ceres::sin(a1);
    const T sin2 = g2o::ceres::sin(a2);
    const g2o::VectorN<2, T> diff = t2 - t1;
    error[0] =
        g2o::ceres::abs((cos1 + cos2) * diff.y() - (sin1 + sin2) * diff.x());
    const g2o::VectorN<2, T> dir{cos1, sin1};
    error[1] = penaltyBoundFromBelow(diff.dot(dir), 0, 0);
    return true;
  }

  G2O_MAKE_AUTO_AD_FUNCTIONS_BY_GET
  bool read(std::istream &) override { return true; }
  bool write(std::ostream &os) const override { return os.good(); }
};

void resize(const Parameters &params, std::vector<g2o::SE2> *poses,
            std::vector<double> *time_diffs) {
  bool updated = true;
  while (updated) {
    updated = false;
    for (int i = 0; i < time_diffs->size(); ++i) {
      const double dt = time_diffs->at(i);
      if (dt > params.dt_ref > params.dt_hysteresis) {
        if (dt > 2 * params.dt_ref) {
          const double new_dt = dt / 2;
          const auto &p0 = poses->at(i);
          const auto &p1 = poses->at(i + 1);
          const Eigen::Vector2d new_t =
              (p0.translation() + p1.translation()) / 2;
          const double new_angle =
              g2o::average_angle(p0.rotation().angle(), p1.rotation().angle());
          time_diffs->at(i) = new_dt;
          poses->insert(poses->begin() + i + 1,
                        g2o::SE2{new_t.x(), new_t.y(), new_angle});
          time_diffs->insert(time_diffs->begin() + i + 1, new_dt);
          --i;
          updated = true;
        } else {
          if (i < time_diffs->size() - 1) {
            time_diffs->at(i + 1) += dt - params.dt_ref;
          }
          time_diffs->at(i) = params.dt_ref;
        }
      } else if (dt < params.dt_ref - params.dt_hysteresis) {
        if (i < time_diffs->size() - 1) {
          time_diffs->at(i + 1) += dt;
          time_diffs->erase(time_diffs->begin() + i);
          poses->erase(poses->begin() + i + 1);
          i--;
        } else {
          time_diffs->at(i - 1) += dt;
          time_diffs->erase(time_diffs->begin() + i);
          poses->erase(poses->begin() + i);
        }
        updated = true;
      }
    }
  }
}

int main() {
  std::vector<g2o::SE2> poses;
  std::vector<double> time_diffs;

  Parameters params;

  g2o::SE2 start_pose{0, 0, 0};
  Eigen::Vector2d start_vel{0.2, 0};
  g2o::SE2 goal_pose{3, 0, -3.14};
  Eigen::Vector2d goal_vel{-0.2, 0};

  poses.push_back(start_pose);
  poses.push_back(goal_pose);
  time_diffs.push_back(
      (start_pose.translation() - goal_pose.translation()).norm() /
      params.max_vel_x);

  auto optimizer = g2o::SparseOptimizer();
  std::unique_ptr<LinearSolver> linear_solver(new LinearSolver());
  linear_solver->setBlockOrdering(true);
  std::unique_ptr<BlockSolver> block_solver(
      new BlockSolver(std::move(linear_solver)));
  g2o::OptimizationAlgorithmLevenberg *solver =
      new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
  optimizer.setAlgorithm(solver);
  optimizer.initMultiThreading();

  optimizer.setComputeBatchStatistics(true);

  for (int loop = 0; loop < 10; ++loop) {
    optimizer.clear();
    resize(params, &poses, &time_diffs);

    const int num_poses = static_cast<int>(poses.size());
    std::cerr << "loop:" << loop << ",poses:" << num_poses << std::endl;
    int id = 0;
    std::cerr << "add vertices ..." << std::endl;
    for (int i = 0; i < num_poses; ++i) {
      const auto &pose = poses.at(i);
      g2o::VertexSE2 *vse2 = new g2o::VertexSE2;
      vse2->setId(id);
      vse2->setEstimate(pose);
      if (i == 0 || i == num_poses - 1) {
        vse2->setFixed(true);
      }
      optimizer.addVertex(vse2);
      ++id;
    }
    for (const auto &dt : time_diffs) {
      VertexTimeDiff *vdt = new VertexTimeDiff;
      vdt->setId(id);
      vdt->setEstimate(dt);
      optimizer.addVertex(vdt);
      ++id;
    }
    std::cerr << "done" << std::endl;
    std::cerr << "add edges ..." << std::endl;
    for (int i = 0; i < num_poses - 1; ++i) {
      Eigen::Matrix<double, 2, 2> information;
      information(0, 0) = params.weight_max_vel_x;
      information(1, 1) = params.weight_max_vel_theta;
      information(0, 1) = 0.0;
      information(1, 0) = 0.0;
      EdgeVelocity *ev = new EdgeVelocity;
      ev->setVertex(0, optimizer.vertices().at(i));
      ev->setVertex(1, optimizer.vertices().at(i + 1));
      ev->setVertex(2, optimizer.vertices().at(num_poses + i));
      ev->setInformation(information);
      ev->setParams(&params);
      optimizer.addEdge(ev);
    }
    {
      Eigen::Matrix<double, 2, 2> information;
      information(0, 0) = params.weight_acc_lim_x;
      information(1, 1) = params.weight_acc_lim_theta;
      information(0, 1) = 0.0;
      information(1, 0) = 0.0;
      EdgeAccelerationSingle *eas = new EdgeAccelerationSingle;
      eas->setVertex(0, optimizer.vertices().at(0));
      eas->setVertex(1, optimizer.vertices().at(1));
      eas->setVertex(2, optimizer.vertices().at(num_poses + 0));
      eas->setVelocity(start_vel, true);
      eas->setInformation(information);
      eas->setParams(&params);
      optimizer.addEdge(eas);
      for (int i = 0; i < num_poses - 2; ++i) {
        EdgeAcceleration *ea = new EdgeAcceleration;
        ea->setVertex(0, optimizer.vertices().at(i));
        ea->setVertex(1, optimizer.vertices().at(i + 1));
        ea->setVertex(2, optimizer.vertices().at(i + 2));
        ea->setVertex(3, optimizer.vertices().at(num_poses + i));
        ea->setVertex(4, optimizer.vertices().at(num_poses + i + 1));
        ea->setInformation(information);
        ea->setParams(&params);
        optimizer.addEdge(ea);
      }
      eas = new EdgeAccelerationSingle;
      eas->setVertex(0, optimizer.vertices().at(num_poses - 2));
      eas->setVertex(1, optimizer.vertices().at(num_poses - 1));
      eas->setVertex(2, optimizer.vertices().at(num_poses + num_poses - 2));
      eas->setVelocity(goal_vel, false);
      eas->setInformation(information);
      eas->setParams(&params);
      optimizer.addEdge(eas);
    }
    for (int i = 0; i < num_poses - 1; ++i) {
      Eigen::Matrix<double, 1, 1> information;
      information.fill(params.weight_timeoptimal);
      EdgeTimeOptimal *et = new EdgeTimeOptimal;
      et->setVertex(0, optimizer.vertices().at(num_poses + i));
      et->setInformation(information);
      optimizer.addEdge(et);
    }
    for (int i = 0; i < num_poses - 1; i++) {
      Eigen::Matrix<double, 2, 2> information;
      information(0, 0) = params.weight_kinematics_nh;
      information(1, 1) = params.weight_kinematics_forward_drive;
      information(0, 1) = 0.0;
      information(1, 0) = 0.0;
      EdgeKinematicsDiffDrive *ek = new EdgeKinematicsDiffDrive;
      ek->setVertex(0, optimizer.vertices().at(i));
      ek->setVertex(1, optimizer.vertices().at(i + 1));
      ek->setInformation(information);
      optimizer.addEdge(ek);
    }
    std::cerr << "done" << std::endl;

    optimizer.setVerbose(true);

    optimizer.initializeOptimization();
    int it = optimizer.optimize(10);
    std::cerr << "iterate:" << it << std::endl;
    for (int i = 0; i < num_poses; ++i) {
      poses.at(i) =
          static_cast<const g2o::VertexSE2 *>(optimizer.vertices().at(i))
              ->estimate();
      if (i < num_poses - 1) {
        time_diffs.at(i) = static_cast<const VertexTimeDiff *>(
                               optimizer.vertices().at(num_poses + i))
                               ->estimate();
      }
    }
  }

  std::cerr << "finished" << std::endl;
  const int num_poses = static_cast<int>(poses.size());
  std::ofstream ofs_csv_file("traj.csv");
  for (int i = 0; i < num_poses; ++i) {
    const g2o::SE2 &pose =
        static_cast<const g2o::VertexSE2 *>(optimizer.vertices().at(i))
            ->estimate();
    double dt = 0.0;
    if (i < num_poses - 1) {
      dt = static_cast<const VertexTimeDiff *>(
               optimizer.vertices().at(num_poses + i))
               ->estimate();
    }
    ofs_csv_file << pose.translation().x() << "," << pose.translation().y()
                 << "," << pose.rotation().angle() << "," << dt << std::endl;
  }
  ofs_csv_file.close();
}
